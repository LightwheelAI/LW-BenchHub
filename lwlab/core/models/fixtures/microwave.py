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

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from robocasa.models.fixtures.microwave import Microwave as RoboCasaMicrowave

from lwlab.utils.usd_utils import OpenUsd as usd


class Microwave(Fixture, RoboCasaMicrowave):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
        self._turned_on = torch.tensor([False], dtype=torch.bool, device=cfg.device).repeat(cfg.num_envs)
        self._door_open = torch.tensor([False], device=cfg.device).repeat(cfg.num_envs)
        self.force_threshold = 0.1
        self.pos_threshold = 0.02
        self.start_button_name = f"{self.fixture_name}_start_button"
        self.stop_button_name = f"{self.fixture_name}_stop_button"

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
        for gripper_name in ["left_gripper", "right_gripper"]:
            start_button_pressed |= env.cfg.check_contact(gripper_name, str(self.button_infos["start_button"][0].GetPrimPath()), has_sensor=False)
            stop_button_pressed |= env.cfg.check_contact(gripper_name, str(self.button_infos["stop_button"][0].GetPrimPath()), has_sensor=False)
        door_open = self.is_open(env)
        self._turned_on = ~door_open & (
            (self._turned_on & ~stop_button_pressed) |
            (~self._turned_on & start_button_pressed)
        )

    def gripper_button_far(self, env: ManagerBasedRLEnv, button, th=0.3):
        assert button in ["start_button", "stop_button"]
        button_idx = env.scene.articulations[self.name].body_names.index(button)
        button_pos = env.scene.articulations[self.name].data.body_com_pos_w[:, button_idx:button_idx + 1, :]  # (env_num, 1, 3)
        ee_pos = env.scene["ee_frame"].data.target_pos_w  # (env_num, ee_num, 3)

        dist = torch.norm(button_pos - ee_pos, dim=-1)  # (env_num, ee_num)
        dist = torch.min(dist, dim=-1)  # (env_num,)

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
