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

from .fixture import Fixture
from .fixture_types import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.utils.usd_utils import OpenUsd as usd
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
import torch


class Blender(Fixture):
    fixture_types = [FixtureType.BLENDER]
    _BLENDER_LID_POS_THRESH = 0.04

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._lid_on_blender = torch.tensor([False], dtype=torch.bool).repeat(num_envs)
        self._turned_on = torch.tensor([False], dtype=torch.bool).repeat(num_envs)
        self._button_contact_prev_timestep = torch.tensor([False], dtype=torch.bool).repeat(num_envs)
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

    def get_reset_region_names(self):
        return {"anchor", }

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
            self._lid_on_blender = torch.tensor([False], dtype=torch.bool).repeat(env.num_envs)
        else:
            closed_lid_pos = self.get_lid_closed_pos(env)
            dist = torch.norm(curr_lid_pos - closed_lid_pos, dim=-1)
            self._lid_on_blender = (dist < self._BLENDER_LID_POS_THRESH)

        button_pressed = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        for gripper_name in [name for name in list(env.scene.sensors.keys()) if "gripper" in name and "contact" in name]:
            button_pressed |= env.cfg.check_contact(gripper_name.replace("_contact", ""), self.power_button_name)
        # since the state updates very often and the same button is used for turning on/off
        # we look at the release of the button to determine the state. If we look at the press then
        # the state will flicker between on and off
        switch_state = self._button_contact_prev_timestep & (~button_pressed)

        if not self._lid_on_blender:
            self._turned_on = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        else:
            self._turned_on[switch_state] = ~self._turned_on[switch_state]
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
