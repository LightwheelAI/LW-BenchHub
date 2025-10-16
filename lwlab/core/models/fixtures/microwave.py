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
from .fixture import Fixture
from functools import cached_property
from .fixture_types import FixtureType
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv

from lwlab.utils.usd_utils import OpenUsd as usd
from lwlab.utils import object_utils as OU


class Microwave(Fixture):
    fixture_types = [FixtureType.MICROWAVE]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._turned_on = torch.tensor([False], dtype=torch.bool, device=self.device).repeat(self.num_envs)
        self._door_open = torch.tensor([False], device=self.device).repeat(self.num_envs)
        self.force_threshold = 0.1
        self.pos_threshold = 0.02
        self.start_button_name = f"{self.name}_start_button"
        self.stop_button_name = f"{self.name}_stop_button"

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def is_open(self, env, th=0.9):
        return super().is_open(env, joint_names=self.door_joint_names, th=th)

    def is_closed(self, env, th=0.005):
        return super().is_closed(env, joint_names=self.door_joint_names, th=th)

    @property
    def door_joint_names(self):
        return ["microjoint"]

    def get_state(self):
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    @property
    def door_name(self):
        return f"{self.microwave_name}_door"

    def update_state(self, env: ManagerBasedRLEnv):
        start_button_pressed = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        stop_button_pressed = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        for gripper_name in [name for name in list(self._env.scene.sensors.keys()) if "gripper" in name and "contact" in name]:
            start_button_pressed |= OU.check_contact(env, gripper_name.replace("_contact", ""), str(self.button_infos["start_button"][0].GetPrimPath()))
            stop_button_pressed |= OU.check_contact(env, gripper_name.replace("_contact", ""), str(self.button_infos["stop_button"][0].GetPrimPath()))
        door_open = self.is_open(env)
        self._turned_on = ~door_open & (
            (self._turned_on & ~stop_button_pressed) |
            (~self._turned_on & start_button_pressed)
        )

    def gripper_button_far(self, env: ManagerBasedRLEnv, button, th=0.3):
        assert button in ["start_button", "stop_button"]
        ee_pos = env.scene["ee_frame"].data.target_pos_w  # (env_num, ee_num, 3)
        button_prims = self.button_infos[button]
        button_pos = []
        for button_prim in button_prims:
            button_pos.append(torch.tensor(button_prim.GetAttribute("xformOp:translate").Get(), dtype=torch.float32, device=env.device).reshape(-1, 3))
        button_pos = torch.stack(button_pos, dim=0)  # (env_num, 1, 3)

        dist = torch.norm(button_pos - ee_pos, dim=-1)  # (env_num, ee_num)
        dist = torch.min(dist, dim=-1).values  # (env_num, )

        return dist > th

    @cached_property
    def button_infos(self):
        button_infos = {}
        for name in ["start_button", "stop_button"]:
            for prim_path in self.prim_paths:
                buttons_prim = usd.get_prim_by_prefix(self._env.sim.stage.GetObjectAtPath(prim_path), name, only_xform=False)
                for button in buttons_prim:
                    if button is not None and button.IsValid():
                        if button.GetName() not in button_infos:
                            button_infos[button.GetName()] = [button]
                        else:
                            button_infos[button.GetName()].append(button)
        for name in ["start_button", "stop_button"]:
            assert name in button_infos.keys(), f"Button {name} not found!"
        return button_infos
