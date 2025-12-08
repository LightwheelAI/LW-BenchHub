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

import numpy as np
import torch

from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class SetupFrying(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER, FixtureType.STOVE]

    """
    Setup Frying: composite task for Frying activity.

    Simulates the task of setting up the frying pan on the stove.

    Steps:
        Place the pan on the stove burner and turn the burner on.

    Args:
        cab_id (str): The id of the cabinet where the pan is placed.
    """

    task_name: str = "SetupFrying"
    knob_id: str = "random"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.stove)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

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
            f"Pick the pan from the cabinet and place it on the {self.knob.replace('_', ' ')} burner on the stove. "
            f"Then turn on the {self.knob.replace('_', ' ')} burner for the pan."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups="pan",
                object_scale=0.8,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.40),
                    pos=(0, -1.0),
                    offset=(0.0, -0.1),
                    # apply a custom rotation for the pan so that it fits better in the cabinet
                    # (if the handle sticks out it may not fit)
                    rotation=(3 * np.pi / 8, 4 * np.pi / 8),
                ),
            )
        )

        # distractors
        for i in range(2):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.cab,
                        ),
                        size=(0.50, 0.50),
                        pos=(0.0, -1.0),
                    ),
                )
            )
        other_knobs = [k for k in self.stove.valid_locations if k != self.knob]
        distr_knob = self.rng.choice(other_knobs)
        cfgs.append(
            dict(
                name="distr_stove",
                obj_groups=("kettle_non_electric"),
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[distr_knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the pan is placed on a stove burner and the burner is turned on.
        """
        pan_loc = self.stove.check_obj_location_on_stove(env, "pan", need_knob_on=True)
        pan_on_stove = torch.tensor([(loc is not None and len(loc) > 0 and loc[0] == self.knob) for loc in pan_loc], device=env.device)
        return pan_on_stove
