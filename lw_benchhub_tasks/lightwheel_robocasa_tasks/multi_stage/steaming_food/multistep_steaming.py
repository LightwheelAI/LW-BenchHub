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


class MultistepSteaming(LwTaskBase):
    """
    Multistep Steaming: composite task for Steaming Food activity.

    Simulates the task of steaming a vegetable.

    Steps:
        Turn on the sink faucet. Then move the vegetable from the counter to the sink.
        Turn off the sink. Move the vegetable from the sink to the pot next to the
        stove. Finally, move the pot to the stove burner.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK, FixtureType.STOVE]
    task_name: str = "MultistepSteaming"
    knob_id: str = "random"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.water_was_turned_on = torch.zeros(self.context.num_envs, dtype=torch.bool, device=self.context.device)
        self.vegetable_was_in_sink = torch.zeros(self.context.num_envs, dtype=torch.bool, device=self.context.device)

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.stove_counter = self.register_fixture_ref(
            "stove_counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable_name = OU.get_obj_lang(self, "vegetable1")
        ep_meta["lang"] = (
            "Turn on the sink faucet. "
            f"Then move the {vegetable_name} from the counter to the sink. "
            f"Turn off the sink. Move the {vegetable_name} from the sink to the pot next to the stove. "
            f"Finally, move the pot to the burner."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids):
        super()._setup_scene(env, env_ids)
        # Reset task progress flags
        self.water_was_turned_on = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.vegetable_was_in_sink = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        # Set initial fixture states
        self.sink.set_handle_state(mode="off", env=env)

        # Select knob for burner placement
        valid_knobs = list(self.stove.get_knobs_state(env=env).keys())
        if self.knob_id == "random":
            self.knob = self.rng.choice(valid_knobs)
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob_id
        self.stove.set_knob_state(mode="off", knob=self.knob, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.stove_counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.6, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.3, 0.3),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.4, 0.4),
                    pos=("ref", None),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        handle_state = self.sink.get_handle_state(env=env)
        water_on = handle_state["water_on"]

        # Track if water was turned on
        self.water_was_turned_on = torch.logical_or(self.water_was_turned_on, water_on)

        # Check object positions
        vegetable_in_sink = OU.obj_inside_of(env, "vegetable1", self.sink)
        vegetable_in_pot = OU.check_obj_in_receptacle(env, "vegetable1", "pot")
        pot_on_stove = OU.check_obj_fixture_contact(env, "pot", self.stove)

        # Track if vegetable was in sink when water was on
        vegetable_washed = vegetable_in_sink & water_on
        self.vegetable_was_in_sink = torch.logical_or(self.vegetable_was_in_sink, vegetable_washed)

        gripper_far = OU.gripper_obj_far(env, "pot")

        return (
            self.water_was_turned_on &
            self.vegetable_was_in_sink &
            ~water_on &  # water is now off
            pot_on_stove &
            vegetable_in_pot &
            gripper_far
        )
