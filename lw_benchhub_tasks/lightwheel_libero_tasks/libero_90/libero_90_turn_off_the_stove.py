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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class L90K8TurnOffTheStove(LwTaskBase):
    task_name: str = 'L90K8TurnOffTheStove'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["stovetop", "mokapot_1", "mokapot_2"]
    removable_fixtures: list[str] = ["mokapot_1", "mokapot_2"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn off the stove."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.mokapot_1 = self.register_fixture_ref("mokapot_1", dict(id=FixtureType.MOKA_POT))
        self.mokapot_2 = self.register_fixture_ref("mokapot_2", dict(id=FixtureType.MOKA_POT))
        self.init_robot_base_ref = self.dining_table

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.stove.set_knob_state(env, knob="center", mode="on")

    def _check_success(self, env):
        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        knobs_state = self.stove.get_knobs_state(env=env)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_off = (abs_knob < lower)
            knob_success = knob_success | knob_off
        return knob_success & OU.gripper_obj_far(env, self.stove.name, 0.3)
