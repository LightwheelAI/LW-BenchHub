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


class DessertUpgrade(LwTaskBase):
    """
    Dessert Upgrade: composite task for Serving Food activity.

    Simulates the task of serving dessert.

    Steps:
        Move the dessert items from the plate to the tray.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER_NON_CORNER]
    task_name: str = "DessertUpgrade"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(1.0, 0.4))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Move the dessert items from the plate to the tray."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="tray",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(1, 0.5),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert1",
                obj_groups="sweets",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0, -1),
                    try_to_place_in="plate",
                ),
            )
        )

        cfgs.append(
            dict(
                name="dessert2",
                obj_groups="sweets",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(1, 0.4),
                    pos=(0, -1),
                    try_to_place_in="plate",
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        sweets_on_tray = OU.check_obj_in_receptacle(
            env, "dessert1", "receptacle"
        ) & OU.check_obj_in_receptacle(env, "dessert2", "receptacle")

        return sweets_on_tray & OU.gripper_obj_far(env, "receptacle")
