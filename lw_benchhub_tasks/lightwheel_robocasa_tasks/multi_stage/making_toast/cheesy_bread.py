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


class CheesyBread(LwTaskBase):
    """
    Cheesy Bread: composite task for Making Toast activity.

    Simulates the task of making cheesy bread.

    Steps:
        Start with a slice of bread already on a plate and a wedge of cheese on the
        counter. Pick up the wedge of cheese and place it on the slice of bread to
        prepare a simple cheese on bread dish.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER_NON_CORNER]
    task_name: str = "CheesyBread"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER_NON_CORNER, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish."

        return ep_meta

    def _setup_scene(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread_flat",
                object_scale=1.5,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.7),
                    pos=(0, -1.0),
                    try_to_place_in="cutting_board",
                ),
            )
        )
        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
                init_robot_here=True,
                placement=dict(
                    ref_obj="bread_container",
                    fixture=self.counter,
                    size=(1.0, 0.3),
                    pos=(0, -1.0),
                ),
            )
        )

        # Distractor on the counter
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(fixture=self.counter, size=(1.0, 0.20), pos=(0, 1.0)),
            )
        )
        return cfgs

    def _check_success(self, env):
        # Bread is still on the cutting board, and cheese is on top
        return OU.check_obj_in_receptacle(env, "bread", "bread_container") &\
            OU.gripper_obj_far(env, obj_name="cheese") &\
            OU.check_contact(env, self.objects["cheese"], self.objects["bread"])
