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


class SimmeringSauce(LwTaskBase):
    """
    Simmering Sauce: composite task for Reheating Food activity.

    Simulates the task of simmering a sauce.

    Steps:
        Place the pan on a specific burner on the stove, then place the tomato and
        the onion in the pan and turn on the burner.

    Args:
        knob_id (str): The id of the knob who's burner the pan will be placed on.
            If "random", a random knob is chosen.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOVE]
    task_name: str = "SimmeringSauce"
    knob_id: str = "random"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.5, 0.4))
        )
        self.init_robot_base_ref = self.stove

        if "refs" in scene._ep_meta:
            self.knob = scene._ep_meta["refs"]["knob"]
        else:
            valid_knobs = self.stove.valid_locations
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Place the pan on the {self.knob.replace('_', ' ')} burner on the stove. "
            f"Then place the tomato and the onion in the pan and turn on the {self.knob.replace('_', ' ')} burner."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                placement=dict(
                    fixture=self.counter,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(ref=self.stove, top_size=(0.50, 0.40)),
                    size=(0.35, 0.4),
                    pos=("ref", -0.4),
                ),
            )
        )

        cfgs.append(
            dict(
                name="tomato",
                obj_groups="tomato",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.45, 0.4),
                    pos=("ref", 0.2),
                ),
            )
        )

        cfgs.append(
            dict(
                name="onion",
                obj_groups="onion",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.45, 0.4),
                    pos=("ref", 0.2),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        locs = self.stove.check_obj_location_on_stove(env, "pan", need_knob_on=True)
        pan_on_stove = torch.tensor([(loc is not None and len(loc) > 0 and loc[0] == self.knob) for loc in locs], device=env.device)
        tomato_in_pan = OU.check_obj_in_receptacle(env, "tomato", "pan")
        onion_in_pan = OU.check_obj_in_receptacle(env, "onion", "pan")
        return pan_on_stove & tomato_in_pan & onion_in_pan
