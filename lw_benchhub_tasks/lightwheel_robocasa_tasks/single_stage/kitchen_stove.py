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


class ManipulateStoveKnob(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.STOVE]
    """
    Class encapsulating the atomic manipulate stove knob tasks.

    Args:
        knob_id (str): The stove knob id to manipulate. If set to "random", a random knob will be selected.

        behavior (str): "turn_on" or "turn_off". Used to define the desired
            stove knob manipulation behavior for the task.
    """
    behavior: str = "turn_on"
    knob_id: str = "random"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the stove knob tasks
        This includes the stove and the stove knob to manipulate, and the burner to place the cookware on.
        """
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        if "task_refs" in scene._ep_meta:
            self.knob = scene._ep_meta["task_refs"]["knob"]
            self.cookware_burner = scene._ep_meta["task_refs"]["cookware_burner"]
        else:
            valid_knobs = self.stove.valid_locations
            if self.knob_id == "random":
                self.knob = self.rng.choice(valid_knobs)
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob
            self.cookware_burner = (
                self.knob
                if self.rng.uniform() <= 0.50
                else self.rng.choice(valid_knobs)
            )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the stove knob tasks.
        This includes the language description of the task and the task references.
        """
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"{self.behavior.replace('_', ' ').capitalize()} the {self.knob.replace('_', ' ')} burner of the stove."
        ep_meta["task_refs"] = dict(
            knob=self.knob,
            cookware_burner=self.cookware_burner,
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Reset the environment internal state for the stove knob tasks.
        This includes setting the stove knob state based on the behavior.
        """
        super()._setup_scene(env, env_ids)

        if self.behavior == "turn_on":
            self.stove.set_knob_state(
                mode="off", knob=self.knob, env=env, env_ids=env_ids
            )
        elif self.behavior == "turn_off":
            self.stove.set_knob_state(mode="on", knob=self.knob, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the stove knob tasks.
        This includes the object placement configurations.
        Place the cookware on the stove burner.

        Returns:
            list: List of object configurations
        """
        cfgs = []

        cfgs.append(
            dict(
                name="cookware",
                obj_groups=("cookware"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.cookware_burner],
                    ),
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the stove knob manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_value = knobs_state[self.knob]

        knob_on = (0.35 <= torch.abs(knob_value)) & (torch.abs(knob_value) <= 2 * torch.pi - 0.35)

        if self.behavior == "turn_on":
            success = knob_on
        elif self.behavior == "turn_off":
            success = ~ knob_on

        return success


class TurnOnStove(ManipulateStoveKnob):
    task_name: str = "TurnOnStove"
    behavior: str = "turn_on"


class TurnOffStove(ManipulateStoveKnob):
    task_name: str = "TurnOffStove"
    behavior: str = "turn_off"
