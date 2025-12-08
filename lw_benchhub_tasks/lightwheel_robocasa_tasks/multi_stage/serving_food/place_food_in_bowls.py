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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class PlaceFoodInBowls(LwTaskBase):
    """
    Place Food In Bowls: composite task for Serving Food activity.

    Simulates the task of placing food in bowls.

    Steps:
        Pick up two bowls and place them on the counter.
        Then, pick up two food items and place them in the bowls.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bowls are
            picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "PlaceFoodInBowls"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food1 = self.get_obj_lang("food1")
        food2 = self.get_obj_lang("food2")
        ep_meta["lang"] = (
            "Pick both bowls and place them on the counter. "
            f"Then pick the {food1} and place it in one bowl and pick the {food2} and place it in the other bowl."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.4),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.4),
                    pos=(1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food1",
                obj_groups="food_set1",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -0.5),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food2",
                obj_groups="food_set1",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -0.5),
                    offset=(0.07, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.20),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_food1_far = OU.gripper_obj_far(env, obj_name="food1")
        gripper_food2_far = OU.gripper_obj_far(env, obj_name="food2")
        food1_in_receptacle1 = OU.check_obj_in_receptacle(env, "food1", "receptacle1")
        food1_in_receptacle2 = OU.check_obj_in_receptacle(env, "food1", "receptacle2")
        food2_in_receptacle1 = OU.check_obj_in_receptacle(env, "food2", "receptacle1")
        food2_in_receptacle2 = OU.check_obj_in_receptacle(env, "food2", "receptacle2")

        receptacles_on_counter = OU.check_obj_fixture_contact(
            env, "receptacle1", self.counter
        ) & OU.check_obj_fixture_contact(env, "receptacle2", self.counter)

        # make sure food are in different bowls
        food_in_bowls = (food1_in_receptacle1 & food2_in_receptacle2) | (
            food1_in_receptacle2 & food2_in_receptacle1
        )

        return gripper_food1_far & gripper_food2_far & food_in_bowls & receptacles_on_counter
