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
from ..loader.floorplan import floorplan_loader


class RobocasaKitchenEnvCfg(BaseSceneEnvCfg):
    """Configuration for the robocasa kitchen environment."""
    fixtures: Dict[str, Any] = {}
    scene_name: str = MISSING
    enable_fixtures: Optional[List[str]] = None

    style_id: int = None
    layout_id: int = None

    EXCLUDE_LAYOUTS = []

    TOASTEN_OVEN_EXCLUDED_LAYOUTS = [1, 2, 11, 32, 37, 40, 41, 44, 45, 52, 55, 57, 58]  # LIGHTWHEEL DEFINE

    OVEN_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 8, 10, 11, 13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    DOUBLE_CAB_EXCLUDED_LAYOUTS = [32, 41, 59]

    DINING_COUNTER_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]

    ISLAND_EXCLUDED_LAYOUTS = [1, 3, 5, 6, 8, 9, 10, 13, 18, 19, 22, 27, 30, 36, 40, 43, 46, 47, 49, 52, 53, 60]

    STOOL_EXCLUDED_LAYOUT = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]

    SHELVES_INCLUDED_LAYOUT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __post_init__(self):
        assert self.scene_name.startswith("robocasa"), "Only robocasa scenes are supported"
        import robocasa.models.scenes.scene_registry as SceneRegistry
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
        self.obj_registries = (
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

        import time
        start_time = time.time()
        print(f"load usd", end="...")
        self.usd_path = str(self._usd_future.result())
        del self._usd_future
        print(f" done in {time.time() - start_time:.2f}s")

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
                self.success_flag &= (self.success_cache <= self.success_count)
                self.success_cache *= (self.success_cache <= self.success_count)
                self.success_flag |= check_result
                self.success_cache += self.success_flag.int()
                return self.success_cache > self.success_count
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
        from robocasa.models.scenes.kitchen_arena import KitchenArena
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            layout_id, style_id = self.rng.choice(self.layout_and_style_ids)
            self.layout_id = int(layout_id)
            self.style_id = int(style_id)
        self._usd_future = floorplan_loader.acquire_usd(self.layout_id, self.style_id, cancel_previous_download=True)

        self._curr_gen_fixtures = self._ep_meta.get("gen_textures")

        import time
        start_time = time.time()
        print(f"load scene {self.layout_id} {self.style_id}", end="...")
        self.mujoco_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            rng=self.rng,
            enable_fixtures=self.enable_fixtures
        )
        print(f" done in {time.time() - start_time:.2f}s")
        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        self.fixture_cfgs = self.mujoco_arena.get_fixture_cfgs()
        # The character . cannot be parsed by isaacsim
        for cfg in self.fixture_cfgs:
            if "." in cfg["name"]:
                cfg["name"] = cfg["name"].replace(".", "_")
        self.fixtures = {cfg["name"]: cfg["model"] for cfg in self.fixture_cfgs}

    def _load_model(self):
        import robocasa.utils.env_utils as EnvUtils
        from robocasa.models.fixtures import Fixture
        import robosuite.utils.transform_utils as T
        import robocasa.macros as macros
        from robocasa.utils.errors import PlacementError

        self._setup_model()

        if self.init_robot_base_ref is not None:
            for i in range(50):  # keep searching for valid environment
                init_fixture = self.get_fixture(self.init_robot_base_ref)
                if init_fixture is not None:
                    break
                self._setup_model()

        try:
            fxtr_placement_initializer = EnvUtils._get_placement_initializer(
                self, self.fixture_cfgs, z_offset=0.0
            )
        except PlacementError as e:
            if macros.VERBOSE:
                print(
                    "Could not create placement initializer for objects. Trying again with self._load_model()"
                )
            self._load_model()
            return
        fxtr_placements = None
        for attempt in range(3):
            try:
                fxtr_placements = fxtr_placement_initializer.sample()
            except PlacementError as e:
                if macros.VERBOSE:
                    print("Placement error for fixtures")
                continue
            break
        if fxtr_placements is None:
            if macros.VERBOSE:
                print("Could not place fixtures. Trying again with self._load_model()")
            self._load_model()
            return
        self.fxtr_placements = fxtr_placements
        for obj_pos, obj_quat, obj in fxtr_placements.values():
            assert isinstance(obj, Fixture)
            obj.set_pos(obj_pos)
            # hacky code to set orientation
            obj.set_euler(T.mat2euler(T.quat2mat(T.convert_quat(obj_quat, "xyzw"))))

        from collections import namedtuple
        dummy_robot = namedtuple("dummy_robot", ["robot_model"])

        class DummyRobot:
            def set_base_xpos(self, pos):
                self.pos = tuple(pos)

            def set_base_ori(self, ori):
                self.ori = T.convert_quat(T.mat2quat(T.euler2mat(ori)), "wxyz")

        self.robots = [dummy_robot(DummyRobot())]

        # setup internal references related to fixtures
        self._setup_kitchen_references()
        # create and place objects
        self._create_objects()

        try:
            self.placement_initializer = EnvUtils._get_placement_initializer(
                self, self.object_cfgs
            )
        except PlacementError as e:
            if macros.VERBOSE:
                print(
                    "Could not create placement initializer for objects. Trying again with self._load_model()"
                )
            self._load_model()
            return
        object_placements = None
        for attempt in range(1):
            try:
                object_placements = self.placement_initializer.sample(
                    placed_objects=self.fxtr_placements
                )
            except PlacementError as e:
                if macros.VERBOSE:
                    print("Placement error for objects")
                continue
            break
        if object_placements is None:
            if macros.VERBOSE:
                print("Could not place objects. Trying again with self._load_model()")
            self._load_model()
            return
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
        stage = usd.get_stage(self.usd_path)
        root_prim = stage.GetPseudoRoot()
        for fixtr in self.fixtures.values():
            if isinstance(fixtr, IsaacFixture):
                try:
                    fixtr.setup_cfg(self, root_prim)
                except Exception as e:
                    print(f"Error setting up cfg of {fixtr.folder.split('/')[-1]}: {str(e)}")

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
        import robocasa.utils.env_utils as EnvUtils

        # add objects
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.name] = model
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
                self.objects[model.name] = model
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
                    self.objects[model.name] = model
                    # self.model.merge_objects([model])

                    # modify object config to lie inside of container
                    cfg["placement"] = dict(
                        size=(0.01, 0.01),
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
        # ep_meta["init_robot_base_pos"] = list(self.init_robot_base_pos)
        # ep_meta["init_robot_base_ori"] = list(self.init_robot_base_ori)
        return ep_meta

    def sample_object(
        self,
        groups,
        exclude_groups=None,
        graspable=None,
        microwavable=None,
        washable=None,
        cookable=None,
        freezable=None,
        dishwashable=None,
        split=None,
        obj_registries=None,
        max_size=(None, None, None),
        object_scale=None,
        rotate_upright=False,
    ):
        """
        Sample a kitchen object from the specified groups and within max_size bounds.

        Args:
            groups (list or str): groups to sample from or the exact xml path of the object to spawn

            exclude_groups (str or list): groups to exclude

            graspable (bool): whether the sampled object must be graspable

            washable (bool): whether the sampled object must be washable

            microwavable (bool): whether the sampled object must be microwavable

            cookable (bool): whether whether the sampled object must be cookable

            freezable (bool): whether whether the sampled object must be freezable

            dishwashable (bool): whether whether the sampled object must be dishwashable

            split (str): split to sample from. Split "train" specifies all but the last 4 object instances
                        (or the first half - whichever is larger), "test" specifies the rest, and None
                        specifies all.

            obj_registries (tuple): registries to sample from

            max_size (tuple): max size of the object. If the sampled object is not within bounds of max size,
                            function will resample

            object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


        Returns:
            dict: kwargs to apply to the MJCF model for the sampled object

            dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to,
            the category of the object the sampling split the object came from, and the groups the object was sampled from
        """
        from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
        return sample_kitchen_object(
            groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            dishwashable=dishwashable,
            rng=self.rng,
            obj_registries=(obj_registries or self.obj_registries),
            split=(split or self.obj_instance_split),
            max_size=max_size,
            object_scale=object_scale,
            rotate_upright=rotate_upright,
        )

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
        from robocasa.models.fixtures import Fixture, FixtureType, fixture_is_type
        import robocasa.models.fixtures.fixture_utils as FixtureUtils

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

            assert isinstance(id, FixtureType)
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
                    print(f"Error updating state of {fixtr.folder.split('/')[-1]}: {str(e)}")

    def _check_success(self):
        return torch.tensor([[False]], device=self.env.device).repeat(self.env.num_envs, 1)

    def _spawn_objects(self):
        pass

    def _reset_internal(self, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env_ids)

        # set up the scene (fixtures, variables, etc)
        self._setup_scene(env_ids)

        # set the robot here
        from lwlab.utils.env import set_robot_to_position, set_robot_base, get_safe_robot_anchor
        if "init_robot_base_pos" in self._ep_meta:
            self.init_robot_base_pos = self._ep_meta["init_robot_base_pos"]
            self.init_robot_base_ori = self._ep_meta["init_robot_base_ori"]
            set_robot_to_position(self.env, self.init_robot_base_pos)
        else:
            # Intercept the unsafe anchor and make it safe
            safe_anchor_pos, safe_anchor_ori = get_safe_robot_anchor(
                env=self.env,
                unsafe_anchor_pos=self.init_robot_base_pos_anchor,
                unsafe_anchor_ori=self.init_robot_base_ori_anchor
            )

            robot_pos = set_robot_base(
                env=self.env,
                anchor_pos=safe_anchor_pos,
                anchor_ori=safe_anchor_ori,
                rot_dev=self.robot_spawn_deviation_rot,
                pos_dev_x=self.robot_spawn_deviation_pos_x,
                pos_dev_y=self.robot_spawn_deviation_pos_y,
                env_ids=env_ids,
            )
            self.init_robot_base_pos = robot_pos
            self.init_robot_base_ori = self.init_robot_base_ori_anchor

    def check_contact(self, geoms_1, geoms_2, has_sensor=True) -> torch.Tensor:
        """
        check if the two geoms are in contact
        """
        if isinstance(geoms_1, str):
            geoms_1_sensor_path = f"{geoms_1}_contact"
        else:
            geoms_1_sensor_path = f"{geoms_1.name}_contact"

        if has_sensor:
            if isinstance(geoms_2, str):
                geoms_2_sensor_path = f"{geoms_2}_contact"
            else:
                geoms_2_sensor_path = f"{geoms_2.name}_contact"
            filter_prim_paths_expr = self.env.scene.sensors[geoms_2_sensor_path].contact_physx_view.sensor_paths
        else:
            filter_prim_paths_expr = [geoms_2]

        geoms_1_contact_paths = self.env.scene.sensors[geoms_1_sensor_path].contact_physx_view.sensor_paths
        contacts = []
        for env_id in range(len(geoms_1_contact_paths)):
            if has_sensor:
                filter_prim_paths_expr = [filter_prim_paths_expr[env_id]]
            else:
                filter_prim_paths_expr = [re.sub(r'env_\d+', f"env_{env_id}", expr) for expr in filter_prim_paths_expr]
            contacts.append(
                abs(max(self.env.sim.physics_sim_view.create_rigid_contact_view(geoms_1_contact_paths[env_id], filter_patterns=filter_prim_paths_expr, max_contact_data_count=1000).get_contact_data(self.env.physics_dt)[0])) > 0
            )
        return torch.tensor(contacts, device=self.env.device)

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)
        """
        return OU.get_obj_lang(self, obj_name=obj_name, get_preposition=get_preposition)
