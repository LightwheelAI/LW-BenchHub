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
from pathlib import Path
from dataclasses import MISSING
from copy import deepcopy
from pathlib import Path
import platform
from termcolor import colored

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from ..base import BaseSceneEnvCfg
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
from lwlab.utils.env import set_camera_follow_pose
from lwlab.utils.usd_utils import OpenUsd as usd
from lwlab.utils.env import ExecuteMode
from ..loader import floorplan_loader, object_loader, ENV_MODE

from lwlab.utils.log_utils import get_error_logger


class RobocasaKitchenEnvCfg(BaseSceneEnvCfg):
    """Configuration for the robocasa kitchen environment."""
    fixtures: Dict[str, Any] = {}
    scene_name: str = MISSING
    enable_fixtures: Optional[List[str]] = None

    style_id: int = None
    layout_id: int = None

    EXCLUDE_LAYOUTS = []

    OVEN_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 8, 10, 11, 13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    DOUBLE_CAB_EXCLUDED_LAYOUTS = [32, 41, 59]

    DINING_COUNTER_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]

    ISLAND_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 8, 9, 10, 13, 18, 19, 22, 27, 30, 36, 40, 43, 46, 47, 49, 52, 53, 60]

    STOOL_EXCLUDED_LAYOUT = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]

    SHELVES_INCLUDED_LAYOUT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __post_init__(self):
        assert self.scene_name.startswith("robocasa"), "Only robocasa scenes are supported"
        import lwlab.utils.place_utils.scene_registry as SceneRegistry
        # scene name is of the form robocasa-kitchen-<layout_id>
        scene_name_split = self.scene_name.split("-")
        layout_ids = SceneRegistry.unpack_layout_ids(None, "test")
        style_ids = SceneRegistry.unpack_style_ids(None, "test")
        if len(scene_name_split) == 3:
            _, layout_id, style_id = scene_name_split
            layout_id = int(layout_id)
            style_id = int(style_id)
            if layout_id not in layout_ids:
                raise ValueError(f"Layout {layout_id} is not in the list of available layouts")
            if style_id not in style_ids:
                raise ValueError(f"Style {style_id} is not in the list of available styles")
            layout_ids = [layout_id]
            style_ids = [style_id]
        elif len(scene_name_split) == 2:
            _, layout_id = scene_name_split
            layout_id = int(layout_id)
            if layout_id not in layout_ids:
                raise ValueError(f"Layout {layout_id} is not in the list of available layouts")
            layout_ids = [layout_id]
        elif len(scene_name_split) != 1:
            raise ValueError(f"Invalid scene name: {self.scene_name}")
        if len(scene_name_split) in (2, 3):
            if layout_id in self.EXCLUDE_LAYOUTS:
                raise ValueError(f"Layout {layout_id} is excluded")

        self.layout_and_style_ids = [(l, s) for l in layout_ids for s in style_ids]

        # remove excluded layouts
        self.layout_and_style_ids = [
            (l, s)
            for (l, s) in self.layout_and_style_ids
            if l not in self.EXCLUDE_LAYOUTS
        ]

        self._ep_meta = {}
        if hasattr(self, "replay_cfgs") and "ep_meta" in self.replay_cfgs:
            self.set_ep_meta(self.replay_cfgs["ep_meta"])
        self.sources = (
            "objaverse",
            "lightwheel"
        )
        self.obj_instance_split = None
        self.fixture_refs = {}
        self.init_robot_base_ref = None
        self.deterministic_reset = False
        # TODO: robot spawn deviation(now is 0.0)
        self.robot_spawn_deviation_pos_x = 0.15
        self.robot_spawn_deviation_pos_y = 0.05
        self.robot_spawn_deviation_rot = 0.0
        if self.execute_mode == ExecuteMode.TELEOP:
            self.success_count = int(1 / self.sim.dt / 2)
        else:
            self.success_count = 1
        self.success_cache = torch.tensor([0], device=self.device, dtype=torch.int32).repeat(self.num_envs)
        self.success_flag = torch.tensor([False], device=self.device, dtype=torch.bool).repeat(self.num_envs)
        self.rng = np.random.default_rng()  # TODO:seed

        # Initialize robot base position and orientation attributes
        self.init_robot_base_pos = [0.0, 0.0, 0.0]
        self.init_robot_base_ori = [0.0, 0.0, 0.0, 1.0]

        self._load_model()
        self.usd_path = self.lwlab_arena.usd_path
        # self.lwlab_arena.stage can not deepcopy
        del self.lwlab_arena

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
            try:
                check_result = self._check_success()
                self.success_flag &= (self.success_cache < self.success_count)
                self.success_cache *= (self.success_cache < self.success_count)
                self.success_flag |= check_result
                self.success_cache += self.success_flag.int()
                return self.success_cache >= self.success_count
            except Exception as e:
                print(f"Error checking success: {e}")
                return torch.tensor([False], device=self.device, dtype=torch.bool).repeat(self.num_envs)

        def camera_pose_update(env):
            set_camera_follow_pose(env, self.viewport_cfg["offset"], self.viewport_cfg["lookat"])
            return False

        self.events.init_kitchen = EventTerm(func=init_kitchen, mode="startup")
        self.terminations.success = DoneTerm(func=check_success_caller)
        if self.first_person_view:
            self.terminations.camera_pose_update = DoneTerm(func=camera_pose_update)

    def _setup_model(self):
        from lwlab.core.scenes.kitchen.kitchen_arena import KitchenArena
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            layout_id, style_id = self.rng.choice(self.layout_and_style_ids)
            self.layout_id = int(layout_id)
            self.style_id = int(style_id)
        self._curr_gen_fixtures = self._ep_meta.get("gen_textures")

        import time
        start_time = time.time()
        print(f"load scene {self.layout_id} {self.style_id}", end="...")
        self.lwlab_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            scene_cfg=self,
        )
        print(f" done in {time.time() - start_time:.2f}s")
        self.fixture_cfgs = self.lwlab_arena.get_fixture_cfgs()

    def _load_model(self):
        from lwlab.core.models.fixtures import Fixture
        import lwlab.utils.math_utils.transform_utils as T

        self._setup_model()

        if self.init_robot_base_ref is not None:
            for i in range(50):  # keep searching for valid environment
                init_fixture = self.get_fixture(self.init_robot_base_ref)
                if init_fixture is not None:
                    break
                self._setup_model()
        self.fxtr_placements, counter_fxtr_placements = usd.get_fixture_placements(self.lwlab_arena.stage.GetPseudoRoot(), self.fixture_cfgs, self.fixtures)
        for obj_pos, obj_quat, obj in self.fxtr_placements.values():
            assert isinstance(obj, Fixture)
            obj.set_euler(T.mat2euler(T.quat2mat(obj_quat)))
        for obj_pos, obj_quat, obj in counter_fxtr_placements.values():
            obj.set_euler(T.mat2euler(T.quat2mat(obj_quat)))

        from collections import namedtuple
        dummy_robot = namedtuple("dummy_robot", ["robot_model"])

        class DummyRobot:
            def set_base_xpos(self, pos):
                self.pos = tuple(pos)

            def set_base_ori(self, ori):
                self.ori = T.convert_quat(T.mat2quat(T.euler2mat(ori)), "wxyz")

        self.robots = [dummy_robot(DummyRobot())]

        # TODO: create objects from usd
        self._create_objects()
        import lwlab.utils.place_utils.env_utils as EnvUtils
        # try:
        self.placement_initializer = EnvUtils._get_placement_initializer(
            self, self.object_cfgs
        )
        # except Exception as e:
        #     print(
        #         "Could not create placement initializer for objects. Trying again with self._load_model()"
        #     )
        #     self._load_model()
        #     return
        object_placements = None

        object_placements = self.placement_initializer.sample(
            placed_objects=self.fxtr_placements
        )

        if object_placements is None:
            print("Could not place objects. Trying again with self._load_model()")
            self._load_model()
            return

        if self.fix_object_pose_cfg is not None:
            for obj_name, obj_placement in object_placements.items():
                if obj_name in self.fix_object_pose_cfg:
                    obj_pos = obj_placement[0]
                    obj_rot = obj_placement[1]
                    if "pos" in self.fix_object_pose_cfg[obj_name]:
                        obj_pos = self.fix_object_pose_cfg[obj_name]["pos"]
                    if "rot" in self.fix_object_pose_cfg[obj_name]:
                        obj_rot = self.fix_object_pose_cfg[obj_name]["rot"]
                    object_placements[obj_name] = (obj_pos, obj_rot, obj_placement[2])

        self.object_placements = object_placements

        (
            self.init_robot_base_pos_anchor,
            self.init_robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(self)

        if hasattr(self, "robot_base_offset"):
            try:
                self.init_robot_base_pos_anchor += np.array(self.robot_base_offset["pos"])
                self.init_robot_base_ori_anchor += np.array(self.robot_base_offset["rot"])
            except KeyError:
                raise ValueError("offset value is not correct !! please make sure offset has key pos and rot !!")

        # set the robot way out of the scene at the start, it will be placed correctly later
        self.robots[0].robot_model.set_base_xpos(self.init_robot_base_pos_anchor)
        self.robots[0].robot_model.set_base_ori(self.init_robot_base_ori_anchor)
        # will set again in _reset_internal func
        self.robot_pos = tuple(self.robots[0].robot_model.pos)
        self.robot_ori = tuple(self.robots[0].robot_model.ori)

    def set_ep_meta(self, meta):
        self._ep_meta = meta

    def _init_fixtures(self):
        # init fixtures for isaac
        if self.usd_simplify:
            self.fixtures = self.fixture_refs
        stage = usd.get_stage(self.usd_path)
        root_prim = stage.GetPseudoRoot()
        for fixtr in self.fixtures.values():
            if isinstance(fixtr, IsaacFixture):
                try:
                    fixtr.setup_cfg(self, root_prim)
                except Exception as e:
                    print(f"Error setting up cfg of {fixtr.name}: {str(e)}")

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
        import lwlab.utils.place_utils.env_utils as EnvUtils
        # add objects
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.task_name] = model
                # self.model.merge_objects([model])
        else:
            self.object_cfgs = self._get_obj_cfgs()
            addl_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.task_name] = model
                # self.model.merge_objects([model])
                try_to_place_in = cfg["placement"].get("try_to_place_in", None)

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

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    addl_obj_cfgs.append(container_cfg)
                    model, info = EnvUtils.create_obj(self, container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.task_name] = model
                    # self.model.merge_objects([model])

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(container_cfg["info"]["size"][0] / 2, container_cfg["info"]["size"][1] / 2),
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                        ),
                    )

            # prepend the new object configs in
            self.object_cfgs = addl_obj_cfgs + self.object_cfgs

            # # remove objects that didn't get created
            # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        def copy_dict_for_json(orig_dict):
            new_dict = {}
            for (k, v) in orig_dict.items():
                if isinstance(v, dict):
                    new_dict[k] = copy_dict_for_json(v)
                elif isinstance(v, IsaacFixture):
                    new_dict[k] = v.name
                else:
                    new_dict[k] = v
            return new_dict

        ep_meta.update(deepcopy(self._ep_meta))
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
                for (k, v) in cfg["reset_region"].items():
                    if isinstance(v, np.ndarray):
                        cfg["reset_region"][k] = list(v)

        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        ep_meta["gen_textures"] = self._curr_gen_fixtures or {}
        ep_meta["lang"] = ""
        ep_meta["fixture_refs"] = dict(
            {k: v.name for (k, v) in self.fixture_refs.items()}
        )
        ep_meta["usd_simplify"] = self.usd_simplify
        ep_meta["LWLAB_ENV_MODE"] = ENV_MODE
        # ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
        # ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)
        # export actual init pose if available in this episode, otherwise omit
        if hasattr(self, "init_robot_base_pos") and hasattr(self, "init_robot_base_ori") and self.init_robot_base_pos is not None and self.init_robot_base_ori is not None:
            ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
            ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)
        return ep_meta

    def get_fixture(self, id, ref=None, size=(0.2, 0.2), full_name_check=False, fix_id=None):
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

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
            cand_fixtures = []
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

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            print(cand_fixtures)
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name, fn_kwargs):
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
        for fixtr in self.fixtures.values():
            if isinstance(fixtr, IsaacFixture):
                try:
                    fixtr.update_state(self.env)
                except Exception as e:
                    get_error_logger().error(f"Error updating state of {fixtr.name}: {str(e)}")

    def _check_success(self):
        return torch.tensor([[False]], device=self.env.device).repeat(self.env.num_envs, 1)

    def _spawn_objects(self):
        for pos, rot, obj in self.object_placements.values():
            obj_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Scene/{obj.task_name}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=obj.obj_path,
                    activate_contact_sensors=True,
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
                force_threshold=10.0,
                filter_prim_paths_expr=[],
            )
            setattr(self.scene, f"{obj.task_name}_contact", obj_concact)

    def _reset_internal(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env_ids)

        # set up the scene (fixtures, variables, etc)
        self._setup_scene(env_ids)
        self.reset_root_state(env=self.env, env_ids=env_ids)

    def check_contact(self, geoms_1, geoms_2) -> torch.Tensor:
        """
        check if the two geoms are in contact
        """
        if self.env.common_step_counter > 1:
            contact_views = [self.env.cfg.contact_queues[env_id].pop() for env_id in range(self.env.num_envs)]
            return torch.tensor(
                [max(abs(view.get_contact_data(self.env.physics_dt)[0])) > 0 for view in contact_views],
                device=self.env.device,
            )
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
                prim_children = prim.GetAllChildren()
                prim_children_paths = [str(child.GetPrimPath()) for child in prim_children]
                geoms_2_sensor_path.append(prim_children_paths)

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
        return torch.tensor([False], device=self.env.device).repeat(self.env.num_envs)

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)
        """
        return OU.get_obj_lang(self, obj_name=obj_name, get_preposition=get_preposition)

    def reset_root_state(self, env, env_ids=None):
        """
        reset the root state of objects and robot in the environment
        """

        def place_objects(self):
            object_placements = None
            if self.placement_initializer is not None:
                try:
                    object_placements = self.placement_initializer.sample(
                        placed_objects=self.fxtr_placements
                    )
                except Exception as e:
                    print("Placement error for objects")
                    place_objects(self)
            return object_placements

        def place_robot(self):
            # set the robot here
            from lwlab.utils.env import set_robot_to_position, sample_robot_base, get_safe_robot_anchor
            if "init_robot_base_pos" in self._ep_meta:
                self.init_robot_base_pos = self._ep_meta["init_robot_base_pos"]
                # if user provides orientation, use it; otherwise fallback to anchor orientation
                if "init_robot_base_ori" in self._ep_meta and self._ep_meta["init_robot_base_ori"] is not None:
                    self.init_robot_base_ori = self._ep_meta["init_robot_base_ori"]
                else:
                    self.init_robot_base_ori = self.init_robot_base_ori_anchor
                # directly set pose using provided xyz (world) and wxyz (world)
                set_robot_to_position(self.env, self.init_robot_base_pos, self.init_robot_base_ori, keep_z=False, env_ids=env_ids)
            else:
                # Intercept the unsafe anchor and make it safe
                safe_anchor_pos, safe_anchor_ori = get_safe_robot_anchor(
                    env=self.env,
                    unsafe_anchor_pos=self.init_robot_base_pos_anchor,
                    unsafe_anchor_ori=self.init_robot_base_ori_anchor
                )

                robot_pos = sample_robot_base(
                    env=self.env,
                    anchor_pos=safe_anchor_pos,
                    anchor_ori=safe_anchor_ori,
                    rot_dev=self.robot_spawn_deviation_rot,
                    pos_dev_x=self.robot_spawn_deviation_pos_x,
                    pos_dev_y=self.robot_spawn_deviation_pos_y,
                    env_ids=env_ids,
                    execute_mode=self.execute_mode,
                )
                self.init_robot_base_pos = robot_pos
                self.init_robot_base_ori = self.init_robot_base_ori_anchor

        if env.cfg.reset_objects_enabled and self.fix_object_pose_cfg is None:
            if env_ids is None:
                env_ids = torch.arange(env.num_envs, device=self.device, dtype=torch.int64)
            for env_id in env_ids:
                object_placements = place_objects(self)
                for obj_name, obj_placement in object_placements.items():
                    obj_pos = torch.tensor(obj_placement[0], device=self.device, dtype=torch.float32) + env.scene.env_origins[env_id]
                    obj_rot = torch.tensor(obj_placement[1], device=self.device, dtype=torch.float32)
                    root_pos = torch.concatenate([obj_pos, obj_rot], dim=-1)
                    env.scene.rigid_objects[obj_name].write_root_pose_to_sim(
                        root_pos,
                        env_ids=torch.tensor([env_id], device=self.device, dtype=torch.int64)
                    )
            env.sim.forward()

        if env.cfg.reset_robot_enabled:
            place_robot(self)
