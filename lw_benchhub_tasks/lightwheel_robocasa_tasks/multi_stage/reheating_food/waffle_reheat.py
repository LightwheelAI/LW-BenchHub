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


class WaffleReheat(LwTaskBase):
    """
    Waffle Reheat: composite task for Reheating Food activity.

    Simulates the task of reheating a waffle.

    Steps:
        Open the microwave. Place the bowl with waffle inside the microwave, then
        close the microwave door and turn it on.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.MICROWAVE]
    task_name: str = "WaffleReheat"
    # exclude layout 8 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [8]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Open the microwave, place the bowl with waffle inside the microwave, "
            "then close the microwave door and turn it on."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="waffle",
                obj_groups="waffle",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                    try_to_place_in="bowl",
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        gripper_far = OU.gripper_obj_far(env, "waffle")
        waffle_in_bowl = OU.check_obj_in_receptacle(env, "waffle", "waffle_container")
        bowl_in_microwave = OU.obj_inside_of(env, "waffle_container", self.microwave)
        microwave_on = self.microwave.get_state()["turned_on"]
        return waffle_in_bowl & bowl_in_microwave & microwave_on & gripper_far
