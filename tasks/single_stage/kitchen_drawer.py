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

import torch

import robocasa.utils.env_utils as EnvUtils
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.others import Floor, Wall

from tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures import *


class ManipulateDrawer(RobocasaKitchenEnvCfg, BaseTaskEnvCfg):
    """
    Class encapsulating the atomic manipulate drawer tasks.

    Args:
        behavior (str): "open" or "close". Used to define the desired
            drawer manipulation behavior for the task.

        drawer_id (str): The drawer fixture id to manipulate
    """

    robot_spawn_deviation_pos_x: float = 0.05
    drawer_id: FixtureType = FixtureType.TOP_DRAWER
    behavior: str = "open"

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self):
        x_ofs = (self.drawer.width / 2) + 0.20
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the drawer
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(-x_ofs - TEST_OFS, -0.10)
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
            self, ref_fixture=self.drawer, offset=(x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the drawer
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.drawer, offset=(x_ofs + TEST_OFS, -0.10)
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
        self.init_robot_base_pos_anchor = robot_base_pos
        self.init_robot_base_ori_anchor = robot_base_ori
        return True

    def _setup_scene(self, env_ids=None):
        """
        Reset the environment internal state for the drawer tasks.
        This includes setting the drawer state based on the behavior
        """
        if self.behavior == "open":
            self.drawer.close_door(env=self.env, env_ids=env_ids)
        elif self.behavior == "close":
            self.drawer.open_door(env=self.env, env_ids=env_ids)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._setup_scene(env_ids)

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the drawer tasks
        """
        super()._setup_kitchen_references()
        valid_drawer = False
        for i in range(7):
            self.drawer = self.get_fixture(id=self.drawer_id)
            if self._place_robot():
                valid_drawer = True
                break
        if not valid_drawer:
            self._load_model()
            return

        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer))
        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        """
        Get the episode metadata for the drawer tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.behavior.capitalize()} the {self.drawer_side} drawer."
        return ep_meta

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

    def _check_success(self):
        """
        Check if the drawer manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.drawer.get_door_state(env=self.env)

        success = torch.tensor([True], device=self.env.scene.device).repeat(self.env.num_envs)
        door_joint_pos = torch.stack(list(door_state.values()), dim=0)
        if self.behavior == "open":
            success = (door_joint_pos >= 0.95).all(dim=0)
        elif self.behavior == "close":
            success = (door_joint_pos <= 0.05).all(dim=0)
        return success


class OpenDrawer(ManipulateDrawer):
    """
    Class encapsulating the atomic open drawer task.
    """

    task_name: str = "OpenDrawer"
    behavior: str = "open"


class CloseDrawer(ManipulateDrawer):
    """
    Class encapsulating the atomic close drawer task.
    """

    task_name: str = "CloseDrawer"
    behavior: str = "close"
