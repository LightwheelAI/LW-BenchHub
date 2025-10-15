# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from dataclasses import MISSING
from copy import deepcopy
from pathlib import Path
import shutil
import traceback
from collections import namedtuple

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from lwlab.core.tasks.base import TerminationsCfg

from ..base import BaseSceneEnvCfg
from lwlab.core.context import get_context
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
from lwlab.utils.env import ExecuteMode
from lwlab.utils.usd_utils import OpenUsd as usd
from lightwheel_sdk.loader import ENDPOINT
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn
import lwlab.utils.math_utils.transform_utils.torch_impl as Tt
from lwlab.utils.log_utils import get_error_logger
from lwlab.core.checks.checker_factory import get_checkers_from_cfg, form_checker_result
import lwlab.utils.place_utils.env_utils as EnvUtils
from lwlab.utils.place_utils.kitchen_object_utils import extract_failed_object_name, recreate_object
from lwlab.core.scenes.kitchen.kitchen_arena import KitchenArena
from lwlab.utils.errors import SamplingError
from lwlab.core.models.fixtures.fixture import FixtureType, Fixture
from lwlab.utils.fixture_utils import fixture_is_type
from lwlab.utils.place_utils.usd_object import USDObject
from lwlab.utils.place_utils.env_utils import set_robot_to_position, sample_robot_base_helper, get_safe_robot_anchor


from isaac_arena.scene.scene import Scene
from isaac_arena.assets.background import Background
from isaac_arena.assets.object_reference import ObjectReference
from isaac_arena.assets.object_base import ObjectType
from isaaclab.utils import configclass
from lwlab.utils.log_utils import copy_dict_for_json

'''
What second stage need to do:

1. Scene USD Change:
    - enable_fixtures (in KitchenArena)
    - removable_fixtures (in KitchenArena)
    - usd_simplify (in KitchenArena)
2. Placement:
    - object placement
    - robot placement
    - retry (init scene_cfg / task_cfg again)
3. 
'''


class LwLabScene(Scene):

    def __init__(self,
                 num_envs: int = 1,
                 device: str = "cpu",
                 excute_mode: ExecuteMode = ExecuteMode.TELEOP,
                 **kwargs,
                 ):
        self.context = get_context()
        self.scene_name = self.context.scene_name
        self.layout_id: int = None
        self.style_id: int = None
        self.fixture_refs = {}
        self.is_replay_mode = self.context.execute_mode in [ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE]
        self._setup_config()

    # first stage (just init from itself)
    def _setup_config(self):
        self.floorplan_version = None
        self._ep_meta = {}
        if self.context.replay_cfgs is not None and "ep_meta" in self.context.replay_cfgs:
            self.set_ep_meta(self.context.replay_cfgs["ep_meta"])
            if "cache_usd_version" in self._ep_meta:
                self.floorplan_version = self._ep_meta["cache_usd_version"]["floorplan_version"]

        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
            self.scene_type = self._ep_meta["scene_type"] if "scene_type" in self._ep_meta else "robocasakitchen"
        else:
            scene_name_split = self.scene_name.split("-")
            if len(scene_name_split) == 3:
                self.scene_type, layout_id, style_id = scene_name_split
            elif len(scene_name_split) == 2:
                self.scene_type, layout_id = scene_name_split
            elif len(scene_name_split) == 1:
                self.scene_type = scene_name_split[0]
            else:
                raise ValueError(f"Invalid scene name: {self.scene_name}")

        self.layout_id = int(layout_id) if layout_id is not None else None
        self.style_id = int(style_id) if style_id is not None else None

    # second stage (init from ArenaEnvironment)
    def setup_env_config(self, orchestrator):
        self.lwlab_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            exclude_layouts=orchestrator.task.exclude_layouts,
            scene_cfg=self,
            scene_type=self.scene_type
        )
        self.usd_path = self.lwlab_arena.usd_path
        self.layout_id = self.lwlab_arena.layout_id
        self.style_id = self.lwlab_arena.style_id
        self.scene_type = self.lwlab_arena.scene_type
        self.fixture_cfgs = self.lwlab_arena.get_fixture_cfgs()
        self.floorplan_version = self.lwlab_arena.version_id
        self.fxtr_placements = usd.get_fixture_placements(self.lwlab_arena.stage.GetPseudoRoot(), self.fixture_cfgs, self.fixtures)

        if self.lwlab_arena.layout_id in orchestrator.task.EXCLUDE_LAYOUTS:
            raise ValueError(f"Layout {self.lwlab_arena.layout_id} is excluded in task {self.task_name}")

        background = Background(
            name=self.scene_type,
            usd_path=self.lwlab_arena.usd_path,
            object_min_z=0.1,
        )

        # flush self.assets
        self.assets = {}
        self.add_asset(background)

    def set_ep_meta(self, meta):
        self._ep_meta = meta

    def get_ep_meta(self):
        ep_meta = {}
        ep_meta.update(deepcopy(self._ep_meta))
        ep_meta["scene_type"] = self.scene_type
        ep_meta["layout_id"] = (
            self.layout_id if isinstance(self.layout_id, dict) else int(self.layout_id)
        )
        ep_meta["style_id"] = (
            self.style_id if isinstance(self.style_id, dict) else int(self.style_id)
        )

        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        # export actual init pose if available in this episode, otherwise omit
        ep_meta["floorplan_version"] = self.floorplan_version
        return ep_meta


class RobocasaKitchenEnvCfg(BaseSceneEnvCfg):
    """Configuration for the robocasa kitchen environment."""
    fixtures: Dict[str, Fixture] = {}
    scene_name: str = MISSING
    scene_group: int = None
    enable_fixtures: Optional[List[str]] = None
    removable_fixtures: Optional[List[str]] = None

    style_id: int = None
    layout_id: int = None

    EXCLUDE_LAYOUTS = []

    # TODO: move these to cloud
    OVEN_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 8, 10, 11, 13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    DOUBLE_CAB_EXCLUDED_LAYOUTS = [32, 41, 59]

    DINING_COUNTER_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]

    ISLAND_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 8, 9, 10, 13, 18, 19, 22, 27, 30, 36, 40, 43, 46, 47, 49, 52, 53, 60]

    STOOL_EXCLUDED_LAYOUT = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]

    SHELVES_INCLUDED_LAYOUT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    DOUBLE_CAB_EXCLUDED_LAYOUTS = [32, 41, 59]

    def __post_init__(self):
        self.cache_usd_version = {}
        self._ep_meta = {}
        if hasattr(self, "replay_cfgs") and "ep_meta" in self.replay_cfgs:
            self.set_ep_meta(self.replay_cfgs["ep_meta"])
            if "cache_usd_version" in self._ep_meta:
                self.cache_usd_version = self._ep_meta["cache_usd_version"]
            if "hdf5_path" in self.replay_cfgs:
                self.hdf5_path = self.replay_cfgs["hdf5_path"]

        self.obj_instance_split = None
        self.fixture_refs = {}

        self.init_robot_base_ref = None
        self.deterministic_reset = False
        self.robot_spawn_deviation_pos_x = 0.15
        self.robot_spawn_deviation_pos_y = 0.05
        self.robot_spawn_deviation_rot = 0.0
        self.start_success_check_count = 10
        if self.execute_mode == ExecuteMode.TELEOP:
            self.success_count = int(1 / self.sim.dt / 2)
        else:
            self.success_count = 1
        self.success_cache = torch.tensor([0], device=self.device, dtype=torch.int32).repeat(self.num_envs)
        self.success_flag = torch.tensor([False], device=self.device, dtype=torch.bool).repeat(self.num_envs)
        self.rng = np.random.default_rng(self.seed)

        # Initialize robot base position and orientation attributes
        self.init_robot_base_pos = [0.0, 0.0, 0.0]
        self.init_robot_base_ori = [0.0, 0.0, 0.0, 1.0]

        # Initialize retry counts
        self.scene_retry_count = 0
        self.object_retry_count = 0

        if self.execute_mode in [ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE]:
            self.is_replay_mode = True
        else:
            self.is_replay_mode = False

        self._load_model()

        # run robocasa fixture initialization ahead of everything else
        super().__post_init__()
        self._init_fixtures()
        self._spawn_objects()

        def __setattr__(self, key, value):
            if key == "microwave":
                breakpoint()
            super().__setattr__(key, value)

        def init_kitchen(env, env_ids):
            self.env = env
            for fixtr in self.fixtures.values():
                if isinstance(fixtr, IsaacFixture):
                    fixtr.setup_env(env)

        def check_success_caller(env):
            self.update_state()
            for checker in self.checkers:
                self.checkers_results[checker.type] = checker.check(env)

            # at the begining of the episode, dont check success for stabilization
            success_check_result = self._check_success()
            assert isinstance(success_check_result, torch.Tensor), f"_check_success must be a torch.Tensor, but got {type(success_check_result)}"
            assert len(success_check_result.shape) == 1 and success_check_result.shape[0] == self.num_envs, f"_check_success must be a torch.Tensor of shape ({self.num_envs},), but got {success_check_result.shape}"
            success_check_result &= (env.episode_length_buf >= self.start_success_check_count)

            # success delay
            self.success_flag &= (self.success_cache < self.success_count)
            self.success_cache *= (self.success_cache < self.success_count)
            self.success_flag |= success_check_result
            self.success_cache += self.success_flag.int()
            return self.success_cache >= self.success_count

        def camera_pose_update(env):
            EnvUtils.set_camera_follow_pose(env, self.viewport_cfg["offset"], self.viewport_cfg["lookat"])
            return torch.tensor([False], device=self.device, dtype=torch.bool).repeat(self.num_envs)

        self.events.init_kitchen = EventTerm(func=init_kitchen, mode="startup")
        self.terminations.success = DoneTerm(func=check_success_caller)
        if self.first_person_view:
            self.terminations.camera_pose_update = DoneTerm(func=camera_pose_update, time_out=True)
        self.init_checkers_cfg()
        self.checkers = get_checkers_from_cfg(self.checkers_cfg)
        self.checkers_results = form_checker_result(self.checkers_cfg)

        # self.lwlab_arena.stage can not deepcopy
        del self.lwlab_arena

    def init_checkers_cfg(self):
        # checkers
        if self.execute_mode in (ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE):
            print("INFO: Running in Replay Mode. Using replay-specific checker config.")
            self.checkers_cfg = {
                "motion": {
                    "warning_on_screen": False
                },
                "kitchen_clipping": {
                    "warning_on_screen": False
                },
                "velocity_jump": {
                    "warning_on_screen": False
                }
            }
        else:
            print("INFO: Running in Teleop Mode. Using default checker config.")
            self.checkers_cfg = {
                "motion": {
                    "warning_on_screen": False
                },
                "kitchen_clipping": {
                    "warning_on_screen": False
                },
                "velocity_jump": {
                    "warning_on_screen": False
                },
                "start_object_move": {
                    "warning_on_screen": False
                }
            }

    def apply_object_init_offset(self, cfgs):
        if hasattr(self, "object_init_offset"):
            object_init_offset = getattr(self, "object_init_offset", [0.0, 0.0])
            for cfg in cfgs:
                if "placement" in cfg and "pos" in cfg["placement"]:
                    pos = cfg["placement"]["pos"]
                    cfg["placement"]["pos"] = ((object_init_offset[0] + pos[0]) if (isinstance(pos[0], float) or isinstance(pos[0], int)) else pos[0],
                                               (object_init_offset[1] + pos[1]) if (isinstance(pos[1], float) or isinstance(pos[1], int)) else pos[1])
        return cfgs

    def _setup_model(self):
        self._curr_gen_fixtures = self._ep_meta.get("gen_textures")
        layout_id = None
        style_id = None
        scene = "robocasakitchen"
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            layout_id = self._ep_meta["layout_id"]
            style_id = self._ep_meta["style_id"]
            self.scene_type = self._ep_meta["scene_type"] if "scene_type" in self._ep_meta else "robocasakitchen"
        else:
            # assert self.scene_name.startswith("robocasa"), "Only robocasa scenes are supported"
            # scene name is of the form robocasa-kitchen-<layout_id>
            scene_name_split = self.scene_name.split("-")
            if len(scene_name_split) == 3:
                self.scene_type, layout_id, style_id = scene_name_split
            elif len(scene_name_split) == 2:
                self.scene_type, layout_id = scene_name_split
            elif len(scene_name_split) == 1:
                self.scene_type = scene_name_split[0]
            else:
                raise ValueError(f"Invalid scene name: {self.scene_name}")
            if len(scene_name_split) in (2, 3):
                if int(layout_id) in self.EXCLUDE_LAYOUTS:
                    raise ValueError(f"Layout {layout_id} is excluded in task {self.task_name}")
            layout_id = int(layout_id) if layout_id is not None else None
            style_id = int(style_id) if style_id is not None else None

        self.lwlab_arena = KitchenArena(
            layout_id=layout_id,
            style_id=style_id,
            scene_cfg=self,
            scene=self.scene_type,
        )

        self.usd_path = self.lwlab_arena.usd_path
        self.layout_id = self.lwlab_arena.layout_id
        self.style_id = self.lwlab_arena.style_id
        self.scene_type = self.lwlab_arena.scene_type
        self.fixture_cfgs = self.lwlab_arena.get_fixture_cfgs()
        self.cache_usd_version.update({"floorplan_version": self.lwlab_arena.version_id})

    def _load_placement(self):
        objects_placement = {}
        import h5py
        ep_names = self.replay_cfgs["ep_names"]
        if len(ep_names) > 1:
            ep_name = ep_names[0]
        else:
            ep_name = ep_names[-1]
        with h5py.File(self.hdf5_path, 'r') as f:
            rigid_objects_path = f"data/{ep_name}/initial_state/rigid_object"

            # Check if rigid_objects_path exists in the file
            if rigid_objects_path not in f:
                return objects_placement

            rigid_objects_group = f[rigid_objects_path]

            for obj_name in rigid_objects_group.keys():
                if obj_name not in self.objects.keys():
                    continue
                pose_path = f"{rigid_objects_path}/{obj_name}"
                obj_group = f[pose_path]
                objects_placement[obj_name] = (
                    tuple(obj_group["root_pose"][0][0:3].tolist()), np.array([obj_group["root_pose"][0][4], obj_group["root_pose"][0][5], obj_group["root_pose"][0][6], obj_group["root_pose"][0][3]], dtype=np.float32), self.objects[obj_name]
                )
        return objects_placement

    def _load_model(self):
        # Reset scene retry count at start of load model
        self.object_retry_count = 0
        # clean cache when not in replay mode
        if self.execute_mode not in [ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE]:
            if not self.cache_usd_version.get("keep_placement", False):
                self.cache_usd_version = {}
        self._setup_model()
        if self.init_robot_base_ref is not None:
            for i in range(50):  # keep searching for valid environment
                init_fixture = self.get_fixture(self.init_robot_base_ref)
                if init_fixture is not None:
                    break
                self._setup_model()
        self.fxtr_placements = usd.get_fixture_placements(self.lwlab_arena.stage.GetPseudoRoot(), self.fixture_cfgs, self.fixtures)

        # setup references related to fixtures
        self._setup_kitchen_references()

        if self.usd_simplify:
            new_stage = usd.usd_simplify(self.lwlab_arena.stage, self.fixture_refs)
            dir_name = os.path.dirname(self.usd_path)
            base_name = os.path.basename(self.usd_path)
            new_path = os.path.join(dir_name, base_name.replace(".usd", "_simplified.usd"))
            new_stage.GetRootLayer().Export(new_path)
            self.usd_path = new_path

        self._create_objects()
        self.object_placements = EnvUtils.sample_object_placements(self)

        if self.fix_object_pose_cfg is not None:
            for obj_name, obj_placement in self.object_placements.items():
                if obj_name in self.fix_object_pose_cfg:
                    obj_pos = obj_placement[0]
                    obj_rot = obj_placement[1]
                    if "pos" in self.fix_object_pose_cfg[obj_name]:
                        obj_pos = self.fix_object_pose_cfg[obj_name]["pos"]
                    if "rot" in self.fix_object_pose_cfg[obj_name]:
                        obj_rot = self.fix_object_pose_cfg[obj_name]["rot"]
                    self.object_placements[obj_name] = (obj_pos, obj_rot, obj_placement[2])

        # replay mode also need this step, to make sure the same robot config step
        self.init_robot_base_pos_anchor, self.init_robot_base_ori_anchor = self.get_robot_anchor()
        self.robot_cfg.init_state.pos = self.init_robot_base_pos_anchor
        self.robot_cfg.init_state.rot = Tn.convert_quat(Tn.mat2quat(Tn.euler2mat(self.init_robot_base_ori_anchor)), to="wxyz")

    def set_ep_meta(self, meta):
        self._ep_meta = meta

    def _init_fixtures(self):
        # init fixtures for isaac
        # if self.usd_simplify:
        #     self.fixtures = self.fixture_refs
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                # try:
                fixtr.setup_cfg(self)
                # except Exception as e:
                #     print(f"Error setting up cfg of {fixtr.name}: {str(e)}")

    def _setup_scene(self, env_ids=None):
        pass

    def _get_obj_cfgs(self):  # direct copy from robocasa Kitchen
        """
        Returns a list of object configurations to use in the environment.
        The object configurations are usually environment-specific and should
        be implemented in the subclass.

        Returns:
            list: list of object configurations
        """

        return []

    def _create_objects(self):  # direct copy from robocasa Kitchen, except merge_objects func
        """
        Creates and places objects in the kitchen environment.
        Helper function called by _create_objects()
        """
        # add objects
        self.objects: Dict[str, USDObject] = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs: List[Dict[str, Any]] = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                object_version = None
                if self.cache_usd_version.get("objects_version", None) is not None:
                    for obj_version in self.cache_usd_version["objects_version"]:
                        if cfg["name"] in obj_version:
                            object_version = obj_version[cfg["name"]]
                            break
                        # TODO(geng.wang): temp code for task_1w_dev data, will be deleted in the future
                        else:
                            if "mjcf_path" in cfg.get("info", {}).keys():
                                xml_path = cfg.get("info", {})["mjcf_path"]
                                name = xml_path.split("/")[-2]
                                if name in obj_version:
                                    object_version = obj_version[name]
                                    break
                        ##
                model, info = EnvUtils.create_obj(self, cfg, version=object_version)
                cfg["info"] = info
                self.objects[model.task_name] = model
        else:
            self.object_cfgs = self._get_obj_cfgs()
            self.object_cfgs = self.apply_object_init_offset(self.object_cfgs)
            all_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.task_name] = model
                # self.model.merge_objects([model])
                try_to_place_in = cfg["placement"].get("try_to_place_in", None)
                object_ref = cfg["placement"].get("object", None)

                # place object in a container and add container as an object to the scene
                if try_to_place_in and (
                    "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": cfg["name"] + "_container",
                        "obj_groups": cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(cfg["placement"]),
                        "type": "object",
                    }

                    init_robot_here = cfg.get("init_robot_here", False)
                    if init_robot_here is True:
                        cfg["init_robot_here"] = False
                        container_cfg["init_robot_here"] = True

                    try_to_place_in_kwargs = cfg["placement"].get(
                        "try_to_place_in_kwargs", None
                    )
                    if try_to_place_in_kwargs is not None:
                        for k, v in try_to_place_in_kwargs.items():
                            container_cfg[k] = v

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    all_obj_cfgs.append(container_cfg)
                    model, info = EnvUtils.create_obj(self, container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.task_name] = model

                    # modify object config to lie inside of container
                    reset_regions = model.get_reset_regions()
                    if "int" in reset_regions:
                        int_region = reset_regions["int"]
                    else:
                        int_region = reset_regions[model.bounded_region_name]
                    cfg["placement"] = dict(
                        size=(int_region["size"][0] / 4, int_region["size"][1] / 4),
                        pos=int_region["offset"],
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                            ref_fixture=cfg["placement"]["fixture"],
                        ),
                    )
                elif (
                    object_ref
                    and "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    parent = object_ref
                    if parent in self.objects:
                        container_name = parent
                        container_obj = self.objects[parent]
                        container_size = container_obj.size
                        smaller_dim = min(container_size[0], container_size[1])
                        sampling_size = (smaller_dim * 0.5, smaller_dim * 0.5)
                        cfg["placement"] = {
                            "size": sampling_size,
                            "ensure_object_boundary_in_range": False,
                            "sample_args": {"reference": container_name},
                        }

                # append the config for this object
                all_obj_cfgs.append(cfg)

            self.object_cfgs = all_obj_cfgs

        # update cache_usd_version
        objects_version = []
        for cfg in self.object_cfgs:
            objects_version.append({cfg["name"]: cfg["info"]["obj_version"]})
        self.cache_usd_version.update({"objects_version": objects_version})

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(deepcopy(self._ep_meta))
        ep_meta["scene_type"] = self.scene_type
        ep_meta["layout_id"] = (
            self.layout_id if isinstance(self.layout_id, dict) else int(self.layout_id)
        )
        ep_meta["style_id"] = (
            self.style_id if isinstance(self.style_id, dict) else int(self.style_id)
        )
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]

        # serialize np arrays to lists
        for cfg in ep_meta["object_cfgs"]:
            if cfg.get("reset_region", None) is not None:
                if isinstance(cfg["reset_region"], dict):
                    cfg["reset_region"] = [cfg["reset_region"]]
                for region in cfg["reset_region"]:
                    for (k, v) in region.items():
                        if isinstance(v, np.ndarray):
                            region[k] = list(v)

        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""
        ep_meta["fixture_refs"] = dict(
            {k: v.name for (k, v) in self.fixture_refs.items()}
        )
        ep_meta["usd_simplify"] = self.usd_simplify
        ep_meta["LW_API_ENDPOINT"] = ENDPOINT
        # ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
        # ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)
        # export actual init pose if available in this episode, otherwise omit
        if hasattr(self, "init_robot_base_pos") and hasattr(self, "init_robot_base_ori") and self.init_robot_base_pos is not None and self.init_robot_base_ori is not None:
            ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
            ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)
        ep_meta["cache_usd_version"] = self.cache_usd_version
        ep_meta["sources"] = self.sources
        ep_meta["object_projects"] = self.object_projects
        ep_meta["seed"] = self.seed
        return ep_meta

    def get_fixture(self, id, ref=None, size=(0.2, 0.2), full_name_check=False, fix_id=None, full_depth_region=False) -> Fixture | None:
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

            full_depth_region (bool): if True, will only sample island counter regions that can be accessed

        Returns:
            Fixture: fixture object
        """
        from lwlab.core.models.fixtures import Fixture, FixtureType, fixture_is_type
        import lwlab.utils.fixture_utils as FixtureUtils

        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in self.fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                if full_name_check:
                    matches = [name for name in self.fixtures.keys() if name == id]
                else:
                    matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if FixtureUtils.is_fxtr_valid(self, self.fixtures[name], size)
                ]
            if (
                len(matches) > 1
                and any("island" in name for name in matches)
                and full_depth_region
            ):
                island_matches = [name for name in matches if "island" in name]
                if len(island_matches) >= 3:
                    depths = [self.fixtures[name].size[1] for name in island_matches]
                    sorted_indices = sorted(range(len(depths)), key=lambda i: depths[i])
                    min_depth = depths[sorted_indices[0]]
                    next_min_depth = (
                        depths[sorted_indices[1]] if len(depths) > 1 else min_depth
                    )
                    if min_depth < 0.8 * next_min_depth:
                        keep = [
                            i
                            for i in range(len(island_matches))
                            if i != sorted_indices[0]
                        ]
                        filtered_islands = [island_matches[i] for i in keep]
                        matches = [
                            name for name in matches if name not in island_matches
                        ] + filtered_islands

            if len(matches) == 0:
                return None
            # sample random key
            if fix_id is not None:
                key = matches[fix_id]
            else:
                key = self.rng.choice(matches)
            return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)

            # NOTE: I dont konw why error here?
            # assert isinstance(id, FixtureType)
            cand_fixtures: List[Fixture] = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = FixtureUtils.is_fxtr_valid(self, fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            if len(cand_fixtures) == 0:
                raise ValueError(f"No fixture found for {id} with size {size}")

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name: str, fn_kwargs: dict):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference

            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        if ref_name not in self.fixture_refs:
            self.fixture_refs[ref_name] = self.get_fixture(**fn_kwargs)
        return self.fixture_refs[ref_name]

    def _setup_kitchen_references(self):  # direct copy from robocasa Kitchen
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self._ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs = {
            k: self.get_fixture(v) for (k, v) in serialized_refs.items()
        }

    def update_state(self):
        """
        Updates the state of the environment.
        This involves updating the state of all fixtures in the environment.
        """
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                # try:
                fixtr.update_state(self.env)
                # except Exception as e:
                #     get_error_logger().error(f"Error updating state of {fixtr.name}: {str(e)}")

    def _check_success(self):
        return torch.tensor([False], device=self.env.device).repeat(self.env.num_envs)

    def get_metrics(self):
        """
        Get all metrics data for JSON export. This function integrates various types of metrics
        and can be extended to include additional metrics in the future.

        Returns:
            dict: Complete metrics data combining all available metrics
        """
        metrics_data = {}

        for checker in self.checkers:
            metrics_data[checker.type] = checker.get_metrics(self.checkers_results[checker.type])

        return metrics_data

    def get_warning_text(self):
        warning_text = ""
        for checker in self.checkers:
            if self.checkers_results[checker.type].get("warning_text"):
                warning_text += self.checkers_results[checker.type].get("warning_text")
                warning_text += "\n"
        return warning_text

    def _spawn_objects(self):
        for pos, rot, obj in self.object_placements.values():
            rot_mat = Tn.quat2mat(rot)
            rot = Tn.convert_quat(rot, to="wxyz")  # xyzw->wxyz
            obj_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Scene/{obj.task_name}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=obj.obj_path,
                    # TODO: huge bug!!! this value will be regarded as contact reporter force_threshold, so set it to False
                    activate_contact_sensors=False,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        sleep_threshold=0.0,  # disable sleep in CPU mode
                        stabilization_threshold=0.0,
                    ),
                ),
            )
            setattr(self.scene, obj.task_name, obj_cfg)
            obj_concact = ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Scene/{obj.task_name}/{obj.name}",
                update_period=0.0,
                history_length=6,
                debug_vis=False,
                force_threshold=0.0,
                filter_prim_paths_expr=[],
            )
            setattr(self.scene, f"{obj.task_name}_contact", obj_concact)

    def _reset_internal(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env_ids)

        for checker in self.checkers:
            checker.reset()

        # set up the scene (fixtures, variables, etc)
        self._setup_scene(env_ids)
        self.reset_root_state(env=self.env, env_ids=env_ids)

    def check_contact(self, geoms_1, geoms_2) -> torch.Tensor:
        """
        check if the two geoms are in contact
        """
        if self.env.common_step_counter == 1:
            if isinstance(geoms_1, str):
                geoms_1_sensor_path = f"{geoms_1}_contact"
            else:
                geoms_1_sensor_path = f"{geoms_1.task_name}_contact"

            if isinstance(geoms_2, str):
                geoms_2_sensor_path = geoms_2
            else:
                geoms_2_sensor_path = []
                geoms_2_prims = usd.get_prim_by_name(self.env.scene.stage.GetPseudoRoot(), geoms_2.name)
                for prim in geoms_2_prims:
                    geoms_2_sensor_path.append([str(cp.GetPrimPath()) for cp in usd.get_prim_by_types(prim, ["Mesh", "Cube", "Cylinder"])])

            geoms_1_contact_paths = self.env.scene.sensors[geoms_1_sensor_path].contact_physx_view.sensor_paths

            for env_id in range(self.env.num_envs):
                if isinstance(geoms_2, str):
                    filter_prim_paths_expr = [re.sub(r'env_\d+', f'env_{env_id}', geoms_2_sensor_path)]
                else:
                    filter_prim_paths_expr = geoms_2_sensor_path[env_id]
                self.env.cfg.contact_queues[env_id].add(
                    self.env.sim.physics_sim_view.create_rigid_contact_view(
                        geoms_1_contact_paths[env_id],
                        filter_patterns=filter_prim_paths_expr,
                        max_contact_data_count=200
                    )
                )
        elif self.env.common_step_counter:
            contact_views = [self.env.cfg.contact_queues[env_id].pop() for env_id in range(self.env.num_envs)]
            return torch.tensor(
                [max(abs(view.get_contact_data(self.env.physics_dt)[0])) > 0 for view in contact_views],
                device=self.env.device,
            )
        return torch.tensor([False], device=self.env.device).repeat(self.env.num_envs)

    def calculate_contact_force(self, geom) -> torch.Tensor:
        """
        calculate the contact force on the geom
        """
        if f"{geom}_contact" in self.env.scene.sensors:
            return torch.max(self.env.scene.sensors[f"{geom}_contact"].data.net_forces_w, dim=-1).values
        else:
            return torch.tensor([0.0], device=self.env.device).repeat(self.env.num_envs)

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)
        """
        return OU.get_obj_lang(self, obj_name=obj_name, get_preposition=get_preposition)

    def _update_fxtr_obj_placement(self, object_placements):
        updated_obj_names = []
        for obj_name, obj_placement in object_placements.items():
            updated_placement = list(obj_placement)
            obj_cfg = [cfg for cfg in self.object_cfgs if cfg["name"] == obj_name][0]
            ref_fixture = None
            if "fixture" in obj_cfg["placement"]:
                ref_fixture = obj_cfg["placement"]["fixture"]
            if isinstance(ref_fixture, str):
                ref_fixture = self.get_fixture(ref_fixture)
            # TODO: add other sliding fxtr types
            if fixture_is_type(ref_fixture, FixtureType.DRAWER):
                ref_rot_mat = Tn.euler2mat(np.array([0, 0, ref_fixture.rot]))
                updated_placement[0] = np.array(updated_placement[0]) + ref_fixture._regions["int"]["per_env_offset"] @ ref_rot_mat.T
                updated_obj_names.append(obj_name)
            else:
                updated_placement[0] = np.array(updated_placement[0])[None, :].repeat(self.num_envs, axis=0)
            object_placements[obj_name] = updated_placement
        return object_placements, updated_obj_names

    def get_robot_anchor(self):
        (
            robot_base_pos_anchor,
            robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(self)

        if hasattr(self, "robot_base_offset"):
            try:
                robot_base_pos_anchor += np.array(self.robot_base_offset["pos"])
                robot_base_ori_anchor += np.array(self.robot_base_offset["rot"])
            except KeyError:
                raise ValueError("offset value is not correct !! please make sure offset has key pos and rot !!")

        # Intercept the unsafe anchor and make it safe
        safe_anchor_pos, safe_anchor_ori = get_safe_robot_anchor(
            cfg=self,
            unsafe_anchor_pos=robot_base_pos_anchor,
            unsafe_anchor_ori=robot_base_ori_anchor
        )

        return safe_anchor_pos, safe_anchor_ori

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

    def reset_root_state(self, env, env_ids=None):
        """
        reset the root state of objects and robot in the environment
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self.device, dtype=torch.int64)
        object_placements = EnvUtils.sample_object_placements(self, need_retry=False)
        object_placements, updated_obj_names = self._update_fxtr_obj_placement(object_placements)
        if env.cfg.reset_objects_enabled and self.fix_object_pose_cfg is None:
            reset_objs = object_placements.keys()
        else:
            reset_objs = updated_obj_names
        for obj_name in reset_objs:
            obj_poses, obj_quat, _ = object_placements[obj_name]
            obj_pos = torch.tensor(obj_poses, device=self.device, dtype=torch.float32)[env_ids] + env.scene.env_origins[env_ids]
            obj_quat = Tt.convert_quat(torch.tensor(obj_quat, device=self.device, dtype=torch.float32), to="wxyz")
            obj_quat = obj_quat.unsqueeze(0).repeat(obj_pos.shape[0], 1)
            root_pos = torch.concatenate([obj_pos, obj_quat], dim=-1)
            env.scene.rigid_objects[obj_name].write_root_pose_to_sim(
                root_pos,
                env_ids=env_ids
            )
        env.sim.forward()

        if env.cfg.reset_robot_enabled:
            self.sample_robot_base(env, env_ids)
            set_robot_to_position(env, self.init_robot_base_pos, self.init_robot_base_ori, keep_z=False, env_ids=env_ids)
