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

import re
from functools import cached_property

import torch

from isaaclab.envs import ManagerBasedRLEnv

from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


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

    def update_state(self, env: ManagerBasedRLEnv, threshold=0.01):
        device = env.device
        num_envs = env.num_envs
        button_prims = self.button_infos
        start_button_pressed = torch.tensor([False], dtype=torch.bool, device=device).repeat(num_envs)
        stop_button_pressed = torch.tensor([False], dtype=torch.bool, device=device).repeat(num_envs)
        robot_keys = [
            k for k in env.scene.articulations.keys()
            if "robot" in k.lower()
        ]
        if not robot_keys:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        robot_key = robot_keys[0]
        robot_articulation = env.scene.articulations[robot_key]
        body_names = robot_articulation.body_names

        microwave_keys = [k for k in env.scene.articulations.keys() if "microwave" in k.lower()]
        if not microwave_keys:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        microwave_key = microwave_keys[0]
        microwave_articulation = env.scene.articulations[microwave_key]
        joint_names = microwave_articulation.joint_names
        body_names_microwave = microwave_articulation.body_names
        has_physical_button = any("button" in name.lower() for name in body_names_microwave)
        if has_physical_button:
            button_press_states = {}

            start_button_prims = button_prims["start_button"]
            for button_prim in start_button_prims:
                prim_path = str(button_prim.GetPath())

                match = re.search(r'(Button[0-9]+)', prim_path, re.IGNORECASE)
                if not match:
                    continue

                button_name = match.group(1)
                button_joint_name = f"{button_name}_joint"

                button_joint_name_lower = button_joint_name.lower()
                joint_lookup = [jn for jn in joint_names if jn.lower() == button_joint_name_lower]
                if not joint_lookup:
                    continue

                button_joint_name = joint_lookup[0]
                button_joint_index = joint_names.index(button_joint_name)

                joint_pos = microwave_articulation.data.joint_pos[:, button_joint_index]
                press_threshold = torch.tensor([0.0001], device=device).repeat(num_envs)
                button_pressed = (joint_pos >= press_threshold)

                button_press_states["start_button"] = button_pressed
                start_button_pressed |= button_pressed

            stop_button_prims = button_prims["stop_button"]
            for button_prim in stop_button_prims:
                prim_path = str(button_prim.GetPath())

                match = re.search(r'(Button[0-9]+)', prim_path, re.IGNORECASE)
                if not match:
                    continue

                button_name = match.group(1)
                button_joint_name = f"{button_name}_joint"

                button_joint_name_lower = button_joint_name.lower()
                joint_lookup = [jn for jn in joint_names if jn.lower() == button_joint_name_lower]
                if not joint_lookup:
                    continue

                button_joint_name = joint_lookup[0]
                button_joint_index = joint_names.index(button_joint_name)

                joint_pos = microwave_articulation.data.joint_pos[:, button_joint_index]
                press_threshold = torch.tensor([0.0001], device=device).repeat(num_envs)
                button_pressed = (joint_pos >= press_threshold)

                button_press_states["stop_button"] = button_pressed
                stop_button_pressed |= button_pressed
        else:
            gripper_site_pos = env.scene["ee_frame"].data.target_pos_w  # (num_envs, num_ee, 3)
            start_button_pos_raw, _, _ = usd.get_prim_pos_rot_in_world(button_prims["start_button"][0])
            stop_button_pos_raw, _, _ = usd.get_prim_pos_rot_in_world(button_prims["stop_button"][0])
            start_button_pos = torch.tensor(start_button_pos_raw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            stop_button_pos = torch.tensor(stop_button_pos_raw, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            dist_start = torch.norm(gripper_site_pos - start_button_pos, dim=-1)  # (num_envs, num_ee)
            dist_stop = torch.norm(gripper_site_pos - stop_button_pos, dim=-1)  # (num_envs, num_ee)
            start_button_pressed = torch.any(dist_start < 0.1, dim=-1)  # (num_envs,)
            stop_button_pressed = torch.any(dist_stop < 0.1, dim=-1)  # (num_envs,)

            # Check contact forces if sensors exist
            def check_gripper_force(sensor_name, threshold=0.01):
                if sensor_name in env.scene.sensors:
                    force = env.scene.sensors[sensor_name]._data.net_forces_w[:, 0, :]  # (num_envs, 3)
                    return torch.norm(force, dim=-1) > threshold  # (num_envs,)
                return torch.zeros(num_envs, dtype=torch.bool, device=device)

            press_force = check_gripper_force("left_gripper_contact") | check_gripper_force("right_gripper_contact")
            start_button_pressed &= press_force
            stop_button_pressed &= press_force

        door_open = self.is_open(env)
        self._turned_on = ~door_open & (
            (self._turned_on & ~stop_button_pressed) |
            (~self._turned_on & start_button_pressed)
        )

    def gripper_button_far(self, env: ManagerBasedRLEnv, button, th=0.3):
        assert button in ["start_button", "stop_button"]
        ee_pos = env.scene["ee_frame"].data.target_pos_w  # (env_num, ee_num, 3)
        button_prims = self.button_infos[button]
        button_pos_list = []
        for prim in button_prims:
            pos, _, _ = usd.get_prim_pos_rot_in_world(prim)
            button_pos_list.append(pos)  # just append raw (3,)
        # shape = (num_buttons, 3)
        button_pos = torch.tensor(button_pos_list, dtype=torch.float32, device=env.device)
        button_pos = button_pos.unsqueeze(0).repeat(env.num_envs, 1, 1)
        diff = button_pos[:, :, None, :] - ee_pos[:, None, :, :]  # broadcast
        dist = torch.norm(diff, dim=-1)  # (num_envs, num_buttons, ee_num)
        # take min among buttons & ee
        dist_min = dist.min(dim=-1).values.min(dim=-1).values  # (env_num,)
        return dist_min > th

    @cached_property
    def button_infos(self):
        button_infos = {"start_button": [], "stop_button": []}
        stage = self._env.sim.stage

        for prim_path in self.prim_paths:
            root_prim = stage.GetObjectAtPath(prim_path)
            if not root_prim or not root_prim.IsValid():
                continue

            for button_name in button_infos.keys():
                found_prims = usd.get_prim_by_name(root_prim, button_name, only_xform=False)
                for prim in found_prims:
                    if prim and prim.IsValid():
                        button_infos[button_name].append(prim)

        missing = [k for k, v in button_infos.items() if not v]
        if missing:
            raise ValueError(f"Missing buttons: {missing} in microwave {self.name}")

        return button_infos
