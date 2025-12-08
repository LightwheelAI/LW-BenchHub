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


class PreheatOven(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.OVEN]
    """
    Class encapsulating the atomic oven preheating task.
    """

    task_name: str = "PreheatOven"
    EXCLUDE_LAYOUTS = LwTaskBase.OVEN_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.init_robot_base_ref = self.oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Preheat the oven by turning the temperature knob."
        return ep_meta

    def _check_success(self, env):
        return torch.tensor(self.oven.get_state()["temperature"] >= 0.1, dtype=torch.bool)


class SlideOvenRack(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.OVEN]
    """
    Atomic task for sliding an oven rack.
    """

    task_name: str = "SlideOvenRack"
    EXCLUDE_LAYOUTS = LwTaskBase.OVEN_EXCLUDED_LAYOUTS

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self, scene):
        x_ofs = (self.oven.width / 2) + 0.25
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the oven
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.oven, offset=(-x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            scene=scene, task=self, ref_fixture=self.oven, offset=(-x_ofs - TEST_OFS, -0.10)
        )

        # check if the robot will be in contact with any fixture or wall during initialization
        if not self.check_fxtr_contact(test_pos_left) and not self._point_outside_scene(
            test_pos_left
        ):
            # oven is to the right of the robot
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        # compute where the robot placement if it is to the right of the oven
        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = EnvUtils.compute_robot_base_placement_pose(
            scene=scene, task=self, ref_fixture=self.oven, offset=(x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the oven
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            scene=scene, task=self, ref_fixture=self.oven, offset=(x_ofs + TEST_OFS, -0.10)
        )

        if not self.check_fxtr_contact(
            test_pos_right
        ) and not self._point_outside_scene(test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            return False
        random_index = self.rng.integers(len(inits))
        robot_base_pos, robot_base_ori, side = inits[random_index]
        self.oven_side = side
        if hasattr(self, "init_robot_base_pos_anchor"):
            self.init_robot_base_pos_anchor[:2] = robot_base_pos[:2]
            self.init_robot_base_ori_anchor[:2] = robot_base_ori[:2]
        else:
            self.init_robot_base_pos_anchor = robot_base_pos
            self.init_robot_base_ori_anchor = robot_base_ori
        return True

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.init_robot_base_ref = self.oven
        if "rack_level" in scene._ep_meta:
            self.should_pull = scene._ep_meta["should_pull"]
            self.rack_level = scene._ep_meta["rack_level"]
        else:
            self.should_pull = self.rng.random() > 0.5
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "out" if self.should_pull else "in"
        if self.oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta["lang"] = f"Fully slide the {rack_pos} oven rack {direction}."
        else:
            ep_meta["lang"] = f"Fully slide the oven rack {direction}."
        ep_meta["should_pull"] = self.should_pull
        ep_meta["rack_level"] = self.rack_level
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

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.oven.open_door(env=env, env_ids=env_ids)

        if not self.should_pull:
            self.oven.slide_rack(env, rack_level=self.rack_level, env_ids=env_ids)
        else:
            self.oven.slide_rack(env, value=0.50, rack_level=self.rack_level, env_ids=env_ids)

    def _check_success(self, env):
        oven_state = self.oven.get_state(rack_level=self.rack_level)

        movable_keys = [k for k in oven_state if k.startswith("rack")]
        if not movable_keys:
            return torch.tensor(False, dtype=torch.bool)

        key = movable_keys[0]
        current_pos = oven_state[key]

        if current_pos is None:
            return torch.tensor(False, dtype=torch.bool)

        if self.should_pull:
            return torch.tensor(current_pos >= 0.85, dtype=torch.bool)
        else:
            return torch.tensor(current_pos <= 0.01, dtype=torch.bool)
