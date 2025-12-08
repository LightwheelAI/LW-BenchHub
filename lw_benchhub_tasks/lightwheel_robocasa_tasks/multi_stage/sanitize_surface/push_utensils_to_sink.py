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


class PushUtensilsToSink(LwTaskBase):
    """
    Push Utensils To Sink: composite task for Sanitize Surface activity.

    Simulates the task of pushing (since utensils are difficult to grasp)
    utensils into the sink.

    Steps:
        Push the utensils into the sink.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "PushUtensilsToSink"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj1_name = self.get_obj_lang("utensil1")
        obj2_name = self.get_obj_lang("utensil2")

        ep_meta["lang"] = f"Push the {obj1_name} and {obj2_name} into the sink."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="utensil1",
                obj_groups=["utensil"],
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                    offset=(0.07, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil2",
                obj_groups=["utensil"],
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(1.0, 0.30),
                    pos=(0.0, 0.0),
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
        utensil1_in_sink = OU.obj_inside_of(env, "utensil1", self.sink)
        utensil2_in_sink = OU.obj_inside_of(env, "utensil2", self.sink)
        gripper_utensil1_far = OU.gripper_obj_far(env, obj_name="utensil1")
        gripper_utensil2_far = OU.gripper_obj_far(env, obj_name="utensil2")
        return utensil1_in_sink & utensil2_in_sink & gripper_utensil1_far & gripper_utensil2_far
