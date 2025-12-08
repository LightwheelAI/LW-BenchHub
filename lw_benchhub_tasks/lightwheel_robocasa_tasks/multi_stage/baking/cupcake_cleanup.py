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


class CupcakeCleanup(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]

    """
    Cupcake Cleanup: composite task for Baking activity.

    Simulates the task of cleaning up after baking cupcakes.

    Steps:
        Move the cupcake off the tray onto the counter, and place the bowl used for
        mixing into the sink.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "CupcakeCleanup"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Move the fresh-baked cupcake off the tray onto the counter, "
            "and place the bowl used for mixing into the sink."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cupcake",
                obj_groups="cupcake",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.4)
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    try_to_place_in="tray",
                    try_to_place_in_kwargs=dict(
                        object_scale=0.6,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.65, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_far = OU.gripper_obj_far(env, "cupcake") & OU.gripper_obj_far(env, "bowl")
        bowl_in_sink = OU.obj_inside_of(env, "bowl", self.sink)
        cupcake_on_counter = OU.check_obj_fixture_contact(env, "cupcake", self.counter)

        return gripper_far & bowl_in_sink & cupcake_on_counter
