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

import torch

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class SteamInMicrowave(LwTaskBase):
    """
    Steam In Microwave: composite task for Steaming Food activity.

    Simulates the task of steaming a vegetable in a microwave.

    Steps:
        Pick the vegetable from the sink and place it in the bowl. Then pick the
        bowl and place it in the microwave. Then close the microwave door and press
        the start button.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.MICROWAVE, FixtureType.SINK]
    task_name: str = "SteamInMicrowave"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_name = OU.get_obj_lang(self, "vegetable")
        ep_meta["lang"] = (
            f"Pick the {vegetable_name} from the sink and place it in the bowl. "
            "Then pick the bowl and place it in the microwave. "
            "Then close the microwave door and press the start button."
        )

        return ep_meta

    def _setup_scene(self, env, env_ids):
        super()._setup_scene(env, env_ids)
        self.sink.set_handle_state(mode="off", env=env)
        self.microwave.set_door_state(min=0.90, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.3, 0.2),
                    pos=(0.0, 1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter_0",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.40),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        vegetable_in_bowl = OU.check_obj_in_receptacle(env, "vegetable", "bowl")
        bowl_in_microwave = OU.obj_inside_of(env, "bowl", self.microwave)

        door_state = self.microwave.get_door_state(env=env)
        door_closed = torch.ones(list(door_state.values())[0].shape[0], device=env.device)
        for joint_p in door_state.values():
            door_closed = torch.logical_and(door_closed, joint_p < 0.05)

        button_pressed = self.microwave.get_state()["turned_on"]

        return vegetable_in_bowl & bowl_in_microwave & door_closed & button_pressed
