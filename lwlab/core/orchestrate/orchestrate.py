from typing import Any, Dict, Optional
from isaac_arena.orchestrator.orchestrator_base import OrchestratorBase
from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.task_base import TaskBase
from isaac_arena.utils.pose import Pose
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
import torch
from lwlab.utils.place_utils import env_utils as EnvUtils
import numpy as np
from lwlab.utils.place_utils.env_utils import set_robot_to_position, sample_robot_base_helper, get_safe_robot_anchor
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn
import lwlab.utils.math_utils.transform_utils.torch_impl as Tt
from lwlab.core.models.fixtures.fixture import FixtureType
from lwlab.utils.fixture_utils import fixture_is_type


class PlacementStrategy:

    def compute_robot_pose(
        self, orchestrator
    ) -> Optional[Pose]:
        """
        Compute the pose of the robot in the scene.
        """
        self.get_robot_anchor()

    def compute_object_poses(
        self, orchestrator
    ) -> Dict[str, Pose]:
        """
        Compute the poses of the objects in the scene.
        """
        object_placements = EnvUtils.sample_object_placements(orchestrator, need_retry=False)
        return object_placements

    @staticmethod
    def get_robot_anchor(orchestrator):
        (
            robot_base_pos_anchor,
            robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(orchestrator)

        if hasattr(orchestrator, "robot_base_offset"):
            try:
                robot_base_pos_anchor += np.array(orchestrator.robot_base_offset["pos"])
                robot_base_ori_anchor += np.array(orchestrator.robot_base_offset["rot"])
            except KeyError:
                raise ValueError("offset value is not correct !! please make sure offset has key pos and rot !!")

        # Intercept the unsafe anchor and make it safe
        safe_anchor_pos, safe_anchor_ori = get_safe_robot_anchor(
            cfg=orchestrator.robot,
            unsafe_anchor_pos=robot_base_pos_anchor,
            unsafe_anchor_ori=robot_base_ori_anchor
        )

        return safe_anchor_pos, safe_anchor_ori


class LwLabBaseOrchestrator(OrchestratorBase):

    def __init__(self, placement_strategy: Optional[PlacementStrategy] = None):
        # self.fixture_controllers = dict()
        self.placement_strategy = placement_strategy
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
        embodiment.setup_env_config(self)
        scene.setup_env_config(self)
        task.setup_env_config(self)

        # set up kitchen references
        self.fixture_refs = self.task.fixture_refs

        # usd simplify
        if self.context.get("usd_simplify", False):
            from lwlab.utils.usd_utils import OpenUsd as usd
            new_stage = usd.usd_simplify(self.scene.lwlab_arena.stage, self.fixture_refs)
            self.scene.scene_type
            self.scene.usd_path = self.scene.usd_path.replace(".usd", "_simplified.usd")
            new_stage.GetRootLayer().Export(self.scene.usd_path)
            # modify background
            self.scene.assets[self.scene.scene_type].usd_path = self.scene.usd_path

        # init ref fixtures
        self._init_ref_fixtures()

        # add ref fixtureassets to arena
        self._add_ref_fixtures_to_arena()

        # place robot and objects
        self.place_robot_and_objects()

        # setup scene done terms
        self.setup_scene_done_terms()

        # setup scene event terms
        self.setup_scene_event_terms()

    def _init_ref_fixtures(self):
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                fixtr.setup_cfg(self)

    def _add_ref_fixtures_to_arena(self):
        # add ref fixtures to arena
        from isaac_arena.assets.object_reference import ObjectReference
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                self.scene.add_asset(ObjectReference(parent_asset=self.scene.assets[self.scene.scene_type]))

    def _reset_internal(self, env_ids, env):
        """
        Reset the event.
        """
        self.scene._setup_scene(env_ids)
        self.scene.reset_root_state(env=env, env_ids=env_ids)

    def init_scene(self, env):
        for fixture_controller in self.fixture_refs.values():
            if isinstance(fixture_controller, IsaacFixture):
                fixture_controller.setup_env(env)

    def update_state(self, env):
        for fixture_controller in self.fixture_refs.values():
            if isinstance(fixture_controller, IsaacFixture):
                fixture_controller.update_state(env)

    def check_success_caller(self, env):
        self.update_state(env)

        for checker in self.scene.checkers:
            self.scene.checkers_results[checker.type] = checker.check(env)

        def _check_success(env):
            return torch.tensor([False], device=env.device).repeat(env.num_envs)

        # at the begining of the episode, dont check success for stabilization
        success_check_result = self.task._check_success(env) if self.task.hasattr('_check_success') else _check_success(env)

        assert isinstance(success_check_result, torch.Tensor), f"_check_success must be a torch.Tensor, but got {type(success_check_result)}"
        assert len(success_check_result.shape) == 1 and success_check_result.shape[0] == env.num_envs, f"_check_success must be a torch.Tensor of shape ({env.num_envs},), but got {success_check_result.shape}"
        success_check_result &= (env.episode_length_buf >= self.scene.start_success_check_count)

        # success delay
        self.scene.success_flag &= (self.scene.success_cache < self.scene.success_count)
        self.scene.success_cache *= (self.scene.success_cache < self.scene.success_count)
        self.scene.success_flag |= success_check_result
        self.scene.success_cache += self.scene.success_flag.int()
        return self.scene.success_cache >= self.scene.success_count

    def setup_scene_done_terms(self):
        """
        Update the state.
        """
        termination_cfg = self.scene.get_termination_cfg()
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        termination_cfg.success = DoneTerm(func=self.check_success_caller)

    def setup_scene_event_terms(self):
        """
        setup the init_scene event.
        """
        events_cfg = self.scene.get_events_cfg()
        from isaaclab.managers import EventTermCfg as EventTerm
        events_cfg.init_scene = EventTerm(func=self.init_scene, mode="startup")

    def get_ep_meta(self):
        """
        Combine the ep_meta of scene, embodiment, task.
        """
        ep_meta = self.scene.get_ep_meta()
        ep_meta.update(self.embodiment.get_ep_meta())
        ep_meta.update(self.task.get_ep_meta())
        ep_meta["cache_usd_version"] = {"floorplan_version": ep_meta["floorplan_version"], "objects_version": ep_meta["objects_version"]}
        return ep_meta

    def place_robot_and_objects(self):
        """
        Place the robot and objects in the scene.
        """
        if not self.placement_strategy:
            return

        robot_pose = self.placement_strategy.compute_robot_pose(self)
        object_poses = self.placement_strategy.compute_object_poses(self)

        self._apply_robot_pose(embodiment_cfg, robot_pose)
        self._apply_object_poses(task_cfg, object_poses)

    def _apply_robot_pose(self, embodiment_cfg, robot_pose: Optional[Pose]):
        if robot_pose and embodiment_cfg and hasattr(embodiment_cfg, 'robot'):
            robot_cfg = embodiment_cfg.robot
            if hasattr(robot_cfg, 'init_state'):
                robot_cfg.init_state.pos = robot_pose.position_xyz
                robot_cfg.init_state.rot = robot_pose.rotation_wxyz

    def _apply_object_poses(self, task_cfg, object_poses: Dict[str, Pose]):
        for obj_name, obj_pose in object_poses.items():
            if hasattr(task_cfg, obj_name):
                obj_cfg = getattr(task_cfg, obj_name)
                if hasattr(obj_cfg, 'init_state'):
                    obj_cfg.init_state.pos = obj_pose.position_xyz
                    obj_cfg.init_state.rot = obj_pose.rotation_wxyz

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
        object_placements, updated_obj_names = self._update_fxtr_obj_placement(object_placements)
        if self.task.reset_objects_enabled and self.task.fix_object_pose_cfg is None:
            reset_objs = object_placements.keys()
        else:
            reset_objs = updated_obj_names
        for obj_name in reset_objs:
            obj_poses, obj_quat, _ = object_placements[obj_name]
            obj_pos = torch.tensor(obj_poses, device=self.context.device, dtype=torch.float32)[env_ids] + env.scene.env_origins[env_ids]
            obj_quat = Tt.convert_quat(torch.tensor(obj_quat, device=self.context.device, dtype=torch.float32), to="wxyz")
            obj_quat = obj_quat.unsqueeze(0).repeat(obj_pos.shape[0], 1)
            root_pos = torch.concatenate([obj_pos, obj_quat], dim=-1)
            env.scene.rigid_objects[obj_name].write_root_pose_to_sim(
                root_pos,
                env_ids=env_ids
            )
        env.sim.forward()

        if self.task.reset_robot_enabled:
            # TODO:(ju.zheng) need to update this
            self.sample_robot_base(env, env_ids)
            set_robot_to_position(env, self.init_robot_base_pos, self.init_robot_base_ori, keep_z=False, env_ids=env_ids)

    def _update_fxtr_obj_placement(self, object_placements):
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
                updated_placement[0] = np.array(updated_placement[0]) + ref_fixture._regions["int"]["per_env_offset"] @ ref_rot_mat.T
                updated_obj_names.append(obj_name)
            else:
                updated_placement[0] = np.array(updated_placement[0])[None, :].repeat(self.num_envs, axis=0)
            object_placements[obj_name] = updated_placement
        return object_placements, updated_obj_names

    def sample_robot_base(self, env, env_ids=None):
        # set the robot here
        if "init_robot_base_pos" in self._ep_meta:
            assert "init_robot_base_ori" in self._ep_meta, "init_robot_base_ori is required when init_robot_base_pos is provided"
            self.init_robot_base_pos = self._ep_meta["init_robot_base_pos"]
            self.init_robot_base_ori = self._ep_meta["init_robot_base_ori"]
            if len(self.init_robot_base_ori) == 4:  # xyzw
                self.init_robot_base_ori = Tn.mat2euler(Tn.quat2mat(self.init_robot_base_ori)).tolist()
        else:
            robot_pos = sample_robot_base_helper(
                env=env,
                anchor_pos=self.init_robot_base_pos_anchor,
                anchor_ori=self.init_robot_base_ori_anchor,
                rot_dev=self.robot_spawn_deviation_rot,
                pos_dev_x=self.robot_spawn_deviation_pos_x,
                pos_dev_y=self.robot_spawn_deviation_pos_y,
                env_ids=env_ids,
                execute_mode=self.execute_mode,
            )
            self.init_robot_base_pos = robot_pos
            self.init_robot_base_ori = self.init_robot_base_ori_anchor
