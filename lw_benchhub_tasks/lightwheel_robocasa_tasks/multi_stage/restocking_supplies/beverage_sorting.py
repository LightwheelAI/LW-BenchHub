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

import numpy as np

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class BeverageSorting(LwTaskBase):
    """
    Beverage Sorting: composite task for Restocking Supplies activity.

    Simulates the task of sorting beverages.

    Steps:
        Sort all alcoholic drinks to one cabinet, and non-alcoholic drinks to the
        other.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER]
    task_name: str = "BeverageSorting"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        if "cabinet1" in self.fixture_refs:
            self.cab1 = self.fixture_refs["cabinet1"]
            self.cab2 = self.fixture_refs["cabinet2"]
            self.counter = self.fixture_refs["counter"]
        else:
            while True:
                self.cab1 = self.get_fixture(FixtureType.CABINET)

                valid_cab_config_found = False
                for _ in range(20):  # 20 attempts
                    # sample until 2 different cabinets are selected
                    self.cab2 = self.get_fixture(FixtureType.CABINET)
                    cab1_rot = self.cab1.rot % (2 * np.pi)
                    cab2_rot = self.cab2.rot % (2 * np.pi)
                    if self.cab2 != self.cab1 and np.abs(cab1_rot - cab2_rot) < 0.05:
                        valid_cab_config_found = True
                        break

                if valid_cab_config_found:
                    break

            self.fixture_refs["cabinet1"] = self.cab1
            self.fixture_refs["cabinet2"] = self.cab2
            self.counter = self.register_fixture_ref(
                "counter", dict(id=FixtureType.COUNTER, size=(0.5, 0.5), ref=self.cab1)
            )

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Sort all alcoholic drinks to one cabinet, and non-alcoholic drinks to the other."
        return ep_meta

    def _setup_scene(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab1.open_door(min=0.85, max=0.9, env=env, env_ids=env_ids)
        self.cab2.open_door(min=0.85, max=0.9, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="alcohol1",
                obj_groups="alcohol",
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.5, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="alcohol2",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    ref_obj="alcohol1",
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="non_alcohol1",
                obj_groups="drink",
                exclude_obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    ref_obj="alcohol1",
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.5, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="non_alcohol2",
                obj_groups="drink",
                exclude_obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    ref_obj="alcohol1",
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab1),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        gripper_far = OU.gripper_obj_far(env, obj_name="alcohol1") & \
            OU.gripper_obj_far(env, obj_name="alcohol2") & \
            OU.gripper_obj_far(env, obj_name="non_alcohol1") & \
            OU.gripper_obj_far(env, obj_name="non_alcohol2")

        # Two possible arrangements
        alcohol_in_cab_1 = OU.obj_inside_of(env, "alcohol1", self.cab1) & \
            OU.obj_inside_of(env, "alcohol2", self.cab1) & \
            OU.obj_inside_of(env, "non_alcohol1", self.cab2) &\
            OU.obj_inside_of(env, "non_alcohol2", self.cab2)

        alcohol_in_cab_2 = OU.obj_inside_of(env, "alcohol1", self.cab2) & \
            OU.obj_inside_of(env, "alcohol2", self.cab2) & \
            OU.obj_inside_of(env, "non_alcohol1", self.cab1) &\
            OU.obj_inside_of(env, "non_alcohol2", self.cab1)

        # return False otherwise
        return gripper_far & (alcohol_in_cab_1 | alcohol_in_cab_2)
