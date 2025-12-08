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


class AfterwashSorting(LwTaskBase):
    """
    Afterwash Sorting: composite task for Washing Fruits And Vegetables activity.

    Simulates the task of sorting washed fruits and vegetables.

    Steps:
        Pick the foods of the same kind from the sink and place them in one bowl.
        Place the food of a different kind in the other bowl. Then, turn off the
        sink.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "AfterwashSorting"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food1_name = OU.get_obj_lang(self, "food1")
        food2_name = OU.get_obj_lang(self, "food2")
        food3_name = OU.get_obj_lang(self, "food3")
        ep_meta["lang"] = (
            f"Pick the {food1_name}s and {food2_name}s from the sink and place them in one bowl. "
            f"Place the {food3_name} in the other bowl. Then turn off the sink faucet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.sink.set_handle_state(mode="on", env=env)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="food1",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.2),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food2",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.2),
                    pos=(1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="food3",
                obj_groups=["vegetable", "fruit"],
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.2, 0.2),
                    pos=(-1.0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl2",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        handle_state = self.sink.get_handle_state(env=env)
        water_on = handle_state["water_on"]

        food1_in_bowl1 = OU.check_obj_in_receptacle(env, "food1", "bowl1")
        food1_in_bowl2 = OU.check_obj_in_receptacle(env, "food1", "bowl2")
        food2_in_bowl1 = OU.check_obj_in_receptacle(env, "food2", "bowl1")
        food2_in_bowl2 = OU.check_obj_in_receptacle(env, "food2", "bowl2")
        food3_in_bowl1 = OU.check_obj_in_receptacle(env, "food3", "bowl1")
        food3_in_bowl2 = OU.check_obj_in_receptacle(env, "food3", "bowl2")

        food12_in_bowl_1 = food1_in_bowl1 & food2_in_bowl1
        food12_in_bowl_2 = food1_in_bowl2 & food2_in_bowl2

        return ~water_on & (
            (food12_in_bowl_1 & food3_in_bowl2) | (food12_in_bowl_2 & food3_in_bowl1))
