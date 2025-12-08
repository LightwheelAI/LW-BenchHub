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

from dataclasses import MISSING

import lw_benchhub.utils.object_utils as OU
import lw_benchhub.utils.place_utils.env_utils as EnvUtils
from lw_benchhub.core.models.fixtures import fixture_is_type
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.models.fixtures import HingeCabinet, FridgeFrenchDoor, FridgeBottomFreezer, Drawer, Counter, Stove, Stovetop, HousingCabinet, SingleCabinet, Fridge, Wall, Floor
from lw_benchhub.core.scenes.kitchen.kitchen import LwScene
from lw_benchhub.core.tasks.base import LwTaskBase


class ManipulateDoor(LwTaskBase):
    """
    Class encapsulating the atomic manipulate door tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            door manipulation behavior for the task.

        fixture_id (str): The door fixture id to manipulate.
    """

    behavior: str = "open"
    fixture_id: FixtureType = MISSING

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the door tasks.
        """
        super()._setup_kitchen_references(scene)
        self.fxtr = self.register_fixture_ref("fxtr", dict(id=self.fixture_id))
        self.init_robot_base_ref = self.fxtr

    def get_ep_meta(self):
        """
        Get the episode metadata for the door tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        if (isinstance(self.fxtr, HingeCabinet)
            or isinstance(self.fxtr, FridgeFrenchDoor)
                or isinstance(self.fxtr, Drawer)):
            door_name = "doors"
        else:
            door_name = "door"
        ep_meta["lang"] = f"{self.behavior.capitalize()} the {self.fxtr.nat_lang} {door_name}"
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Reset the environment internal state for the door tasks.
        This includes setting the door state based on the behavior.
        """
        if self.behavior == "open":
            self.fxtr.close_door(env=env, env_ids=env_ids)
        elif self.behavior == "close":
            self.fxtr.open_door(env=env, env_ids=env_ids)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._setup_scene(env, env_ids)
        self._place_robot(scene=env.cfg.isaaclab_arena_env.scene)

    def _place_robot(self, scene: LwScene):
        pass

    def _check_success(self, env):
        """
        Check if the door manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        if self.behavior == "open":
            return self.fxtr.is_open(env=env)
        elif self.behavior == "close":
            return self.fxtr.is_closed(env=env)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the door tasks. This includes the object placement configurations.
        Place one object inside the door fixture and 1-4 distractors on the counter.
        """
        cfgs = []

        if not fixture_is_type(self.fxtr, FixtureType.DISHWASHER):
            cfg = dict(
                name="door_obj",
                obj_groups="all",
                graspable=True,
                placement=dict(
                    fixture=self.fxtr,
                    size=(0.30, 0.30),
                    pos=(None, -1.0),
                ),
            )
            if fixture_is_type(self.fxtr, FixtureType.OVEN):
                cfg["placement"]["try_to_place_in"] = "oven_tray"
                cfg["placement"]["size"] = (1.0, 0.45)
            cfgs.append(cfg)

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.fxtr),
                        sample_region_kwargs=dict(
                            ref=self.fxtr,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class ManipulateLowerDoor(ManipulateDoor):
    X_OFS = 0.2
    Y_OFS = -0.1
    robot_side = ""
    behavior: str = "open"
    robot_spawn_deviation_pos_x = 0.05

    def _place_robot(self, scene: LwScene):
        x_ofs = (self.fxtr.width / 2) + self.X_OFS
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the drawer
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            scene=scene,
            task=self,
            ref_fixture=self.fxtr,
            offset=(-x_ofs, self.Y_OFS)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            scene=scene,
            task=self,
            ref_fixture=self.fxtr,
            offset=(-x_ofs - TEST_OFS, self.Y_OFS)
        )

        # check if the robot will be in contact with any fixture or wall during initialization
        if not self.check_fxtr_contact(test_pos_left) and not self._point_outside_scene(
            test_pos_left
        ):
            # drawer is to the right of the robot
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        # compute where the robot placement if it is to the right of the drawer
        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = EnvUtils.compute_robot_base_placement_pose(
            scene=scene,
            task=self,
            ref_fixture=self.fxtr,
            offset=(x_ofs, self.Y_OFS)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the drawer
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            scene=scene,
            task=self,
            ref_fixture=self.fxtr,
            offset=(x_ofs + TEST_OFS, self.Y_OFS)
        )

        if not self.check_fxtr_contact(
            test_pos_right
        ) and not self._point_outside_scene(test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            return False
        random_index = self.rng.integers(len(inits))
        robot_base_pos, robot_base_ori, side = inits[random_index]
        self.drawer_side = side
        return True

    def check_fxtr_contact(self, pos):
        """
        Check if the point is in contact with any fixture

        Args:
            pos (tuple): The position of the point to check

        Returns:
            bool: True if the point is in contact with any fixture, False otherwise
        """
        fxtrs = [
            fxtr
            for fxtr in self.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, SingleCabinet)
            or isinstance(fxtr, HingeCabinet)
            or isinstance(fxtr, Fridge)
            or (isinstance(fxtr, Wall) and not isinstance(fxtr, Floor))
        ]

        for fxtr in fxtrs:
            # get bounds of fixture
            if OU.point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                return True
        return False

    def _point_outside_scene(self, pos):
        walls = [
            fxtr for (name, fxtr) in self.fixtures.items() if isinstance(fxtr, Floor)
        ]
        return not any(
            [
                OU.point_in_fixture(point=pos, fixture=wall, only_2d=True)
                for wall in walls
            ]
        )


class OpenDoor(ManipulateDoor):
    task_name: str = "OpenDoor"
    behavior: str = "open"


class CloseDoor(ManipulateDoor):
    task_name: str = "CloseDoor"
    behavior: str = "close"


class OpenCabinet(OpenDoor):
    task_name: str = "OpenCabinet"
    fixture_id = FixtureType.CABINET_WITH_DOOR


class CloseCabinet(CloseDoor):
    task_name: str = "CloseCabinet"
    fixture_id = FixtureType.CABINET_WITH_DOOR


class OpenMicrowave(OpenDoor):
    task_name: str = "OpenMicrowave"
    fixture_id = FixtureType.MICROWAVE


class CloseMicrowave(CloseDoor):
    task_name: str = "CloseMicrowave"
    fixture_id = FixtureType.MICROWAVE


class OpenFridge(OpenDoor):
    task_name: str = "OpenFridge"
    fixture_id = FixtureType.FRIDGE

    def _place_robot(self, scene: LwScene):
        if isinstance(self.fxtr, FridgeBottomFreezer):
            OFFSET = (-0.30, -0.30)
        else:
            OFFSET = (0, -0.30)

        (
            init_robot_base_pos_anchor,
            init_robot_base_ori_anchor,
        ) = EnvUtils.compute_robot_base_placement_pose(
            scene=scene,
            task=self,
            ref_fixture=self.fxtr,
            offset=OFFSET
        )
        if hasattr(self, "init_robot_base_pos_anchor"):
            self.init_robot_base_pos_anchor[:2] = init_robot_base_pos_anchor[:2]
            self.init_robot_base_ori_anchor[:2] = init_robot_base_ori_anchor[:2]
        else:
            self.init_robot_base_pos_anchor = init_robot_base_pos_anchor
            self.init_robot_base_ori_anchor = init_robot_base_ori_anchor
        return True


class CloseFridge(CloseDoor):
    task_name: str = "CloseFridge"
    fixture_id = FixtureType.FRIDGE

    def _place_robot(self, scene: LwScene):
        if isinstance(self.fxtr, FridgeBottomFreezer):
            OFFSET = (-0.30, -0.30)
        else:
            OFFSET = (0, -0.30)

        (
            init_robot_base_pos_anchor,
            init_robot_base_ori_anchor,
        ) = EnvUtils.compute_robot_base_placement_pose(
            scene=scene,
            task=self,
            ref_fixture=self.fxtr,
            offset=OFFSET
        )
        if hasattr(self, "init_robot_base_pos_anchor"):
            self.init_robot_base_pos_anchor[:2] = init_robot_base_pos_anchor[:2]
            self.init_robot_base_ori_anchor[:2] = init_robot_base_ori_anchor[:2]
        else:
            self.init_robot_base_pos_anchor = init_robot_base_pos_anchor
            self.init_robot_base_ori_anchor = init_robot_base_ori_anchor
        return True


class OpenDropDownDoor(ManipulateLowerDoor):
    behavior: str = "open"


class CloseDropDownDoor(ManipulateLowerDoor):
    behavior: str = "close"


class OpenOven(OpenDropDownDoor):
    task_name: str = "OpenOven"
    EXCLUDE_LAYOUTS = LwTaskBase.OVEN_EXCLUDED_LAYOUTS
    fixture_id = FixtureType.OVEN


class CloseOven(CloseDropDownDoor):
    task_name: str = "CloseOven"
    EXCLUDE_LAYOUTS = LwTaskBase.OVEN_EXCLUDED_LAYOUTS
    Y_OFS = -0.25
    fixture_id = FixtureType.OVEN


class OpenDishwasher(OpenDropDownDoor):
    # with lower x_ofs the base of the robots sometimes blocks the door from closing all the way
    task_name: str = "OpenDishwasher"
    X_OFS = 0.275
    fixture_id = FixtureType.DISHWASHER
    robot_spawn_deviation_pos_x = 0


class CloseDishwasher(CloseDropDownDoor):
    task_name: str = "CloseDishwasher"
    X_OFS = 0.25
    Y_OFS = -0.25
    fixture_id = FixtureType.DISHWASHER


class OpenToasterOvenDoor(LwTaskBase):
    """
    Class encapsulating the atomic toaster oven door tasks.

    Args:
        behavior (str): "open". Used to define the desired door manipulation
            behavior for the task
    """
    task_name: str = "OpenToasterOvenDoor"
    enable_fixtures = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the toaster oven door."
        return ep_meta

    def _check_success(self, env):
        return self.toaster_oven.is_open(env=env)


class CloseToasterOvenDoor(LwTaskBase):
    """
    Class encapsulating the atomic toaster oven door tasks.

    Args:
        behavior (str): "close". Used to define the desired door manipulation
            behavior for the task
    """
    task_name: str = "CloseToasterOvenDoor"
    enable_fixtures = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the toaster oven door."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.toaster_oven.open_door(env)

    def _check_success(self, env):
        return self.toaster_oven.is_closed(env=env)
