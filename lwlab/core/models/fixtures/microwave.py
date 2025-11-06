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
from pxr import UsdGeom, Usd, Gf
import re
import lwlab.utils.math_utils.transform_utils.numpy_impl as T
import numpy as np
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
            for cube_name in [name for name in robot_articulation.body_names if "cube" in name]:
                cube_name_index = body_names.index(cube_name.replace("_contact", ""))
                cube_pos = robot_articulation.data.body_com_pos_w[:, cube_name_index, :]  # (env_num, 3)

                # ---- Start button ----
                start_button_prims = button_prims["start_button"]
                start_button_pos = []
                for button_prim in start_button_prims:
                    xform = UsdGeom.Xformable(button_prim)
                    world_mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    world_pos = world_mat.ExtractTranslation()
                    start_button_pos.append(torch.tensor(world_pos, dtype=torch.float32, device=device))
                start_button_pos = torch.stack(start_button_pos, dim=0).unsqueeze(0).repeat(num_envs, 1, 1)

                dist_start = torch.norm(cube_pos.unsqueeze(1) - start_button_pos, dim=-1)
                pressed_start = torch.any(dist_start < threshold, dim=-1)
                start_button_pressed |= pressed_start

                # ---- Stop button ----
                stop_button_prims = button_prims["stop_button"]
                stop_button_pos = []
                for button_prim in stop_button_prims:
                    xform = UsdGeom.Xformable(button_prim)
                    world_mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    world_pos = world_mat.ExtractTranslation()
                    stop_button_pos.append(torch.tensor(world_pos, dtype=torch.float32, device=device))
                stop_button_pos = torch.stack(stop_button_pos, dim=0).unsqueeze(0).repeat(num_envs, 1, 1)

                dist_stop = torch.norm(cube_pos.unsqueeze(1) - stop_button_pos, dim=-1)
                pressed_stop = torch.any(dist_stop < threshold, dim=-1)
                stop_button_pressed |= pressed_stop

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
            pos, _ = usd.get_prim_pos_rot_in_world(button_prim)
            button_pos.append(torch.tensor(pos, dtype=torch.float32, device=env.device))
        button_pos = torch.stack(button_pos, dim=0)  # (env_num, 1, 3)

        dist = torch.norm(button_pos - ee_pos, dim=-1)  # (env_num, ee_num)
        dist = torch.min(dist, dim=-1).values  # (env_num, )

        return dist > th

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
