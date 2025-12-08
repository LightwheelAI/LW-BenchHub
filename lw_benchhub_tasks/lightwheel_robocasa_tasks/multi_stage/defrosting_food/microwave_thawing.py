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


class MicrowaveThawing(LwTaskBase):
    """
    Microwave Thawing: composite task for Defrosting Food activity.

    Simulates the task of defrosting food in a microwave.

    Steps:
        Pick the food from the counter and place it in the microwave.
        Then turn on the microwave.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.MICROWAVE]
    task_name: str = "MicrowaveThawing"
    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.microwave)
        )
        self.distr_counter = self.register_fixture_ref(
            "distractor_counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.microwave,
            ),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.microwave.close_door(env=env, env_ids=env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name = self.get_obj_lang()
        ep_meta["lang"] = (
            f"Pick the {obj_name} from the counter and place it in the microwave. "
            "Then turn on the microwave."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups="food",
                graspable=True,
                microwavable=True,
                freezable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.50, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        obj_in_microwave = OU.obj_inside_of(env, "obj", self.microwave)

        button_pressed = self.microwave.get_state()["turned_on"]
        gripper_obj_far = OU.gripper_obj_far(env)

        return obj_in_microwave & gripper_obj_far & button_pressed
