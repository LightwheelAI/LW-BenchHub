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

from isaaclab.envs import ManagerBasedRLEnv

from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class Blender(Fixture):
    fixture_types = [FixtureType.BLENDER]
    _BLENDER_LID_POS_THRESH = 0.05

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self.blender_lid = None
        self.power_button_name = "power_button_main"

        self._joint_names = {
            "knob_speed": "knob_speed_joint",
            "pitcher": "pitcher_joint",
            "blade": "blade_joint",
            "power": "power_button_joint",
        }

    def set_speed_dial_knob(self, env, knob_val):
        """
        Sets the speed of the blender

        Args:
            knob_val (float): normalized value between 0 and 1 (max speed)
        """
        self._speed_dial_knob_value = torch.clip(
            torch.tensor(knob_val, device=env.device), 0.0, 1.0)
        jn = self._joint_names["knob_speed"]
        self.set_joint_state(
            env=env,
            min=self._speed_dial_knob_value,
            max=self._speed_dial_knob_value,
            joint_names=[jn],
        )

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env
        device = env.device
        num_envs = env.num_envs

        self._lid_on_blender = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._turned_on = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._button_contact_prev_timestep = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def get_reset_region_names(self):
        return {"anchor", "int"}

    # TODO: need to add env_nums
    def get_lid_closed_pos(self, env):
        prims = usd.get_all_prims(env.scene.stage)
        anchor_prim = next((p for p in prims if "reg_anchor" in p.GetPath().pathString), None)
        pos_blender = env.scene.articulations[self.name].data.root_pos_w
        trans_attr = anchor_prim.GetAttribute("xformOp:translate").Get()
        anchor_offset = torch.tensor([trans_attr[0], trans_attr[1], trans_attr[2]], dtype=torch.float32, device=env.device)
        pos_anchor = pos_blender + anchor_offset
        return pos_anchor

    def get_curr_lid_pos(self, env):
        self.blender_lid = next((k for k in env.scene.rigid_objects if k.startswith(self.name) and k.endswith("_lid")), None)
        if self.blender_lid is None:
            return None
        return env.scene.rigid_objects[self.blender_lid].data.root_pos_w

    def update_state(self, env):
        curr_lid_pos = self.get_curr_lid_pos(env)
        if curr_lid_pos is None:
            self._lid_on_blender = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        else:
            closed_lid_pos = self.get_lid_closed_pos(env)
            dist = torch.norm(curr_lid_pos - closed_lid_pos, dim=-1)
            self._lid_on_blender = (dist < self._BLENDER_LID_POS_THRESH)
        blender_articulation = env.scene.articulations[self.name]
        body_names_blender = blender_articulation.body_names
        has_physical_button = any("button" in name.lower() for name in body_names_blender)
        button_pressed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if has_physical_button:
            joint_names = blender_articulation.joint_names
            power_joint_name = self._joint_names["power"]
            if power_joint_name in joint_names:
                power_joint_idx = joint_names.index(power_joint_name)
                joint_pos = blender_articulation.data.joint_pos[:, power_joint_idx]
                # A small positive value indicates the button is pressed.
                press_threshold = 0.0001
                button_pressed = joint_pos >= press_threshold
        switch_state = self._button_contact_prev_timestep & (~button_pressed)

        self._turned_on[~self._lid_on_blender] = False

        self._turned_on[switch_state & self._lid_on_blender] = ~self._turned_on[switch_state & self._lid_on_blender]

        self._button_contact_prev_timestep = button_pressed

    def get_state(self):
        state = dict(
            lid_on_blender=self._lid_on_blender,
            lid_not_on_blender=~self._lid_on_blender,
            turned_on=self._turned_on,
        )
        return state

    def get_power_button(self, env):
        prims = usd.get_all_prims(env.scene.stage)
        power_button_prim = next((p for p in prims if "power_button_main" in p.GetPath().pathString), None)
        if power_button_prim:
            return power_button_prim.GetName()
        return None
