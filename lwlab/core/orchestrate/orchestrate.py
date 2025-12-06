from typing import Any, Dict, Optional
from isaaclab_arena.orchestrator.orchestrator_base import OrchestratorBase
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.pose import Pose
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
import torch
from lwlab.utils.place_utils import env_utils as EnvUtils
import numpy as np
from scipy.spatial.transform import Rotation as R
from lwlab.utils.place_utils.env_utils import set_robot_to_position, sample_robot_base_helper
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn
import lwlab.utils.math_utils.transform_utils.torch_impl as Tt
from lwlab.core.models.fixtures.fixture import FixtureType
from lwlab.core.context import get_context
from lwlab.utils.fixture_utils import fixture_is_type
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from lwlab.utils.isaaclab_utils import NoDeepcopyMixin
from lwlab.utils.log_utils import get_code_version
from lwlab.utils.env import ExecuteMode


class LwLabBaseOrchestrator(OrchestratorBase, NoDeepcopyMixin):

    def __init__(self):
        # self.fixture_controllers = dict()
        self.context = get_context()
        self.scene = None
        self.embodiment = None
        self.task = None

    # def add_fixture_controller(self, fixture_controller):
    #     self.fixture_controllers[fixture_controller.name] = fixture_controller

    # def update_fixture_controllers(self, fixtures: Dict[str, Any]):
    #     for fixture_name, fixture_cfg in fixtures.items():
    #         if fixture_name in self.fixture_controllers:
    #             self.fixture_controllers[fixture_name].update(fixture_cfg)

    def orchestrate(self, embodiment: EmbodimentBase, scene: Scene, task: TaskBase) -> None:

        self.context = scene.context

        # Second stage: update scene, embodiment, task
        self.scene = scene
        self.embodiment = embodiment
        self.task = task

        scene.setup_env_config(self)
        task.setup_env_config(self)
        embodiment.setup_env_config(self)

        # set up kitchen references
        self.fixture_refs = self.task.fixture_refs

        # usd simplify
        if self.context.usd_simplify:
            from lwlab.utils.usd_utils import OpenUsd as usd
            new_stage = usd.usd_simplify(self.scene.lwlab_arena.stage, [ref.name for ref in self.fixture_refs.values()])
            self.scene.scene_type
            self.scene.scene_usd_path = self.scene.scene_usd_path.replace(".usd", "_simplified.usd")
            new_stage.GetRootLayer().Export(self.scene.scene_usd_path)
            # modify background
            self.scene.assets["Scene"].usd_path = self.scene.scene_usd_path
        # del self.scene.lwlab_arena

    def _reset_internal(self, env, env_ids):
        """
        Reset the event.
        """
        if self.task.context.task_backend == "robocasa":
            self.task._setup_scene(env, env_ids)
            self.reset_root_state(env=env, env_ids=env_ids)

    def update_state(self, env):
        for fixture_controller in self.fixture_refs.values():
            if isinstance(fixture_controller, IsaacFixture):
                fixture_controller.update_state(env)

    def get_ep_meta(self):
        """
        Combine the ep_meta of scene, embodiment, task.
        """
        ep_meta = self.scene.get_ep_meta()
        ep_meta.update(self.embodiment.get_ep_meta(self.task.env_instance))
        ep_meta.update(self.task.get_ep_meta())
        ep_meta["cache_usd_version"] = {"floorplan_version": ep_meta["floorplan_version"], "objects_version": ep_meta["objects_version"]}
        ep_meta["code_version"] = get_code_version()
        return ep_meta

    def _extract_cfg(self, cfg, key):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        return None

    def _apply_cfg(self, cfg, key, value):
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    def reset_root_state(self, env, env_ids=None):
        """
        reset the root state of objects and robot in the environment
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self.context.device, dtype=torch.int64)
        object_placements = EnvUtils.sample_object_placements(self, need_retry=False)
        object_placements, updated_obj_names = self._update_fxtr_obj_placement(object_placements, env_ids=env_ids)
        if self.task.resample_objects_placement_on_reset and self.task.fix_object_pose_cfg is None:
            reset_objs = object_placements.keys()
            # test asset part
            if self.context.execute_mode == ExecuteMode.TEST_OBJECT:
                self.task.visible_obj_idx += 1
                if self.task.visible_obj_idx >= len(self.task.objects):
                    self.task.visible_obj_idx = 0
                print(f"[INFO] Placing object {list(self.task.objects.keys())[self.task.visible_obj_idx]}")
                print(f"[INFO] Object sizes: {self.task.objects[list(self.task.objects.keys())[self.task.visible_obj_idx]].size}")
        else:
            reset_objs = updated_obj_names
        for obj_name in reset_objs:
            obj_pos, obj_quat_xyzw, _ = object_placements[obj_name]
            obj_pos_multienv = torch.tensor(obj_pos, device=self.context.device, dtype=torch.float32) + env.scene.env_origins[env_ids]
            obj_quat = Tt.convert_quat(torch.tensor(obj_quat_xyzw, device=self.context.device, dtype=torch.float32), to="wxyz")
            obj_quat = obj_quat.unsqueeze(0).repeat(obj_pos_multienv.shape[0], 1)
            root_pos_multienv = torch.concatenate([obj_pos_multienv, obj_quat], dim=-1)
            if obj_name in self.task._articulation_assets:
                self.fixture_refs[obj_name]._pos = obj_pos_multienv
                self.fixture_refs[obj_name]._rot = R.from_quat(obj_quat_xyzw).as_euler('xyz', degrees=False)
                env.scene.articulations[obj_name].write_root_pose_to_sim(
                    root_pos_multienv,
                    env_ids=env_ids
                )
            else:
                env.scene.rigid_objects[obj_name].write_root_pose_to_sim(
                    root_pos_multienv,
                    env_ids=env_ids
                )

        if self.task.resample_robot_placement_on_reset:
            # TODO:(ju.zheng) need to update this
            self.sample_robot_base(env, env_ids)
            set_robot_to_position(env, self.init_robot_base_pos, self.init_robot_base_ori, env_ids=env_ids)

    def _update_fxtr_obj_placement(self, object_placements, env_ids=None):
        updated_obj_names = []
        for obj_name, obj_placement in object_placements.items():
            updated_placement = list(obj_placement)
            obj_cfg = [cfg for cfg in self.task.object_cfgs if cfg["name"] == obj_name][0]
            ref_fixture = None
            if "fixture" in obj_cfg["placement"]:
                ref_fixture = obj_cfg["placement"]["fixture"]
            if isinstance(ref_fixture, str):
                ref_fixture = self.task.get_fixture(ref_fixture)
            # TODO: add other sliding fxtr types
            if fixture_is_type(ref_fixture, FixtureType.DRAWER):
                ref_rot_mat = Tn.euler2mat(np.array([0, 0, ref_fixture.rot]))
                updated_placement[0] = np.array(updated_placement[0]) + ref_fixture._regions["int"]["per_env_offset"][env_ids] @ ref_rot_mat.T
                updated_obj_names.append(obj_name)
            else:
                updated_placement[0] = np.array(updated_placement[0])[None, :].repeat(env_ids.shape[0], axis=0)
            object_placements[obj_name] = updated_placement
        return object_placements, updated_obj_names

    def sample_robot_base(self, env, env_ids=None):
        # set the robot here
        ep_meta = self.get_ep_meta()
        if "init_robot_base_pos" in ep_meta:
            assert "init_robot_base_ori" in ep_meta, "init_robot_base_ori is required when init_robot_base_pos is provided"
            self.init_robot_base_pos = ep_meta["init_robot_base_pos"]
            self.init_robot_base_ori = ep_meta["init_robot_base_ori"]
            if len(self.init_robot_base_ori) == 4:  # xyzw
                self.init_robot_base_ori = Tn.mat2euler(Tn.quat2mat(self.init_robot_base_ori)).tolist()
        else:
            robot_pos = sample_robot_base_helper(
                env=env,
                anchor_pos=self.embodiment.init_robot_base_pos_anchor,
                anchor_ori=self.embodiment.init_robot_base_ori_anchor,
                rot_dev=self.embodiment.robot_spawn_deviation_rot,
                pos_dev_x=self.embodiment.robot_spawn_deviation_pos_x,
                pos_dev_y=self.embodiment.robot_spawn_deviation_pos_y,
                env_ids=env_ids,
                execute_mode=self.context.execute_mode,
            )
            self.init_robot_base_pos = robot_pos
            self.init_robot_base_ori = self.embodiment.init_robot_base_ori_anchor
