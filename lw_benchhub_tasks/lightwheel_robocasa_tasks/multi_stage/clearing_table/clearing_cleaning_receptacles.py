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


class ClearingCleaningReceptacles(LwTaskBase):
    """
    Clearing Cleaning Receptacles: composite task for Clearing Table activity.

    Simulates the process of clearing receptacles from the dining table and
    cleaning them in the sink.

    Steps:
        Pick the receptacles from the dining table and place them in the sink.
        Then, turn on the water.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK, FixtureType.STOOL]
    task_name: str = "ClearingCleaningReceptacles"
    EXCLUDE_LAYOUTS: list = LwTaskBase.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        # dining_table is a sufficiently large counter closest to the stools
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("receptacle1")
        obj_name_2 = self.get_obj_lang("receptacle2")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_name_1} and {obj_name_2} and place them in the sink. Then turn on the water."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.sink.set_handle_state(mode="off", env=env)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="receptacle",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.8, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="receptacle",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.8, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        receptacle1_in_sink = OU.obj_inside_of(env, "receptacle1", self.sink)
        receptacle2_in_sink = OU.obj_inside_of(env, "receptacle2", self.sink)

        handle_state = self.sink.get_handle_state(env=env)
        water_on = handle_state["water_on"]

        return receptacle1_in_sink & receptacle2_in_sink & water_on
