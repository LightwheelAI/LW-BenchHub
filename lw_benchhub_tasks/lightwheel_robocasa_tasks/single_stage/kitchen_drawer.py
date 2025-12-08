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

import lw_benchhub.utils.object_utils as OU
import lw_benchhub.utils.place_utils.env_utils as EnvUtils
from lw_benchhub.core.models import fixtures
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.models.fixtures.others import Floor, Wall
from lw_benchhub.core.tasks.base import LwTaskBase


class ManipulateDrawer(LwTaskBase):
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

    def _place_robot(self, scene):
        x_ofs = (self.drawer.width / 2) + 0.25
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the drawer
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            scene, self, ref_fixture=self.drawer, offset=(-x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            scene, self, ref_fixture=self.drawer, offset=(-x_ofs - TEST_OFS, -0.10)
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
            scene, self, ref_fixture=self.drawer, offset=(x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the drawer
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            scene, self, ref_fixture=self.drawer, offset=(x_ofs + TEST_OFS, -0.10)
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

    def _setup_scene(self, env, env_ids=None):
        """
        Reset the environment internal state for the drawer tasks.
        This includes setting the drawer state based on the behavior
        """
        if self.behavior == "open":
            self.drawer.close_door(env=env, env_ids=env_ids)
        elif self.behavior == "close":
            self.drawer.open_door(env=env, env_ids=env_ids)
        # set the door state then place the objects otherwise objects initialized in opened drawer will fall down before the drawer is opened
        super()._setup_scene(env, env_ids)
        self.drawer.update_state(env)

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the drawer tasks
        """
        super()._setup_kitchen_references(scene)
        valid_drawer = False
        for i in range(7):
            self.drawer = self.get_fixture(id=self.drawer_id)
            if self._place_robot(scene=scene):
                valid_drawer = True
                break
        if not valid_drawer:
            raise ValueError("Failed to find suitable drawer to place robot")

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
            if isinstance(fxtr, fixtures.Counter)
            or isinstance(fxtr, fixtures.Stove)
            or isinstance(fxtr, fixtures.Stovetop)
            or isinstance(fxtr, fixtures.HousingCabinet)
            or isinstance(fxtr, fixtures.SingleCabinet)
            or isinstance(fxtr, fixtures.HingeCabinet)
            or isinstance(fxtr, fixtures.Fridge)
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

    def _check_success(self, env):
        """
        Check if the drawer manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        door_state = self.drawer.get_door_state(env=env)

        success = torch.tensor([True], device=env.device).repeat(env.num_envs)
        door_joint_pos = torch.stack(list(door_state.values()), dim=0)
        if self.behavior == "open":
            success = (door_joint_pos >= 0.99).all(dim=0)
        elif self.behavior == "close":
            success = (door_joint_pos <= 0.01).all(dim=0)
        return success


class OpenDrawer(ManipulateDrawer):
    """
    Class encapsulating the atomic open drawer task.
    """

    task_name: str = "OpenDrawer"
    behavior: str = "open"

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups="all",
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                        sample_region_kwargs=dict(
                            ref=self.drawer,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class CloseDrawer(ManipulateDrawer):
    """
    Class encapsulating the atomic close drawer task.
    """

    task_name: str = "CloseDrawer"
    behavior: str = "close"

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the drawer tasks. This includes the object placement configurations.
        Place the object inside the drawer and 1-4 distractors on the counter.

        Returns:
            list: List of object configurations.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="drawer_obj",
                obj_groups="all",
                graspable=True,
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, -0.75),
                    # offset=(0, -self.drawer.size[1] * 0.15),
                ),
            )
        )

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i+1}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.drawer),
                        sample_region_kwargs=dict(
                            ref=self.drawer,
                        ),
                        size=(1.0, 0.50),
                        pos=(None, -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )

        return cfgs


class SlideDishwasherRack(LwTaskBase):
    """
    Class encapsulating sliding dishwasher rack in or out atomic task.
    """
    task_name: str = "SlideDishwasherRack"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dishwasher = self.register_fixture_ref(
            "dishwasher", dict(id=FixtureType.DISHWASHER)
        )
        if "should_pull" in scene._ep_meta:
            self.should_pull = scene._ep_meta["should_pull"]
        else:
            self.should_pull = self.rng.random() > 0.5

        self.init_robot_base_ref = self.dishwasher

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "out" if self.should_pull else "in"
        ep_meta["lang"] = f"Fully slide the top dishwasher rack {direction}."
        ep_meta["should_pull"] = self.should_pull
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.dishwasher.open_door(env=env, env_ids=env_ids)

        if not self.should_pull:
            self.dishwasher.slide_rack(env=env, value=0.7, env_ids=env_ids)
        else:
            self.dishwasher.slide_rack(env=env, value=0.2, env_ids=env_ids)

    def _check_success(self, env):
        current_pos = self.dishwasher.get_state(env)["rack"]

        if self.should_pull:
            return current_pos >= 0.65
        else:
            return current_pos <= 0.10
