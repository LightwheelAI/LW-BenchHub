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


class QuickThaw(LwTaskBase):
    """
    Quick Thaw: composite task for Defrosting Food activity.

    Simulates the task of defrosting meat on a stove.

    Steps:
        Pick the meat from the counter and place it in a pot on a burner.
        Then turn on the burner.

    Args:
        knob_id (str): The id of the knob who's burner the pot will be placed on.
            If "random", a random knob is chosen.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOVE]
    task_name: str = "QuickThaw"
    knob_id = "random"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        # Pick a knob/burner on a stove and a counter close to it
        valid_knobs = [k for (k, v) in self.stove.knob_joints.items() if v is not None]

        if not valid_knobs:
            valid_knobs = list(self.stove.knob_joints.keys())
        if not valid_knobs:
            valid_knobs = self.stove.valid_locations
        if not valid_knobs:
            raise RuntimeError("No valid knobs found on the stove fixture")

        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=FixtureType.STOVE)
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        knob_desc = getattr(self, "knob", self.knob_id if self.knob_id != "random" else "selected")
        ep_meta["lang"] = (
            "Frozen meat rests on a plate on the counter. "
            "Retrieve the meat and place it in a pot on a burner. Then turn the burner on."
        )
        ep_meta["knob"] = self.knob
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.stove.set_knob_state(mode="off", knob=self.knob, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.30),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        # place the pot on the specific burner we chose earlier
        cfgs.append(
            dict(
                name="container",
                obj_groups=("pot"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    size=(0.02, 0.02),
                    rotation=[(-3 * torch.pi / 8, -torch.pi / 4), (torch.pi / 4, 3 * torch.pi / 8)],
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_value = knobs_state[self.knob]  # (num_envs,)
        knob_on = (0.35 <= torch.abs(knob_value)) & (torch.abs(knob_value) <= 2 * torch.pi - 0.35)
        meat_in_container = OU.check_obj_in_receptacle(env, "meat", "container")
        gripper_far = OU.gripper_obj_far(env, obj_name="meat")
        return knob_on & meat_in_container & gripper_far
