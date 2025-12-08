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

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class Oven(Fixture):
    fixture_types = [FixtureType.OVEN]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._door = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
        self._timer = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
        self._temperature = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
        self._rack = {}

        self._joint_names = {
            "door": f"door_joint",
            "timer": f"knob_time_joint",
            "temperature": f"knob_temp_joint",
            "rack": f"rack0_joint",
        }

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def open_door(self, env, env_ids=None):
        self._door[env_ids] = 1.0
        self.set_joint_state(
            min=1.0, max=1.0, env=env, joint_names=[self._joint_names["door"]], env_ids=env_ids
        )

    def close_door(self, env, env_ids=None):
        self._door[env_ids] = 0.0
        self.set_joint_state(
            min=0.0, max=0.0, env=env, joint_names=[self._joint_names["door"]], env_ids=env_ids
        )

    def set_timer(self, env, value, env_ids=None):
        value = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=env.device)
        self._timer[env_ids] = torch.clip(value, 0.0, 1.0)
        for env_id in env_ids:
            self.set_joint_state(
                min=self._timer[env_id],
                max=self._timer[env_id],
                env=env,
                joint_names=[self._joint_names["timer"]],
                env_ids=torch.tensor([env_id])
            )

    def set_temperature(self, env, value, env_ids=None):
        value = value if isinstance(value, torch.Tensor) else torch.tensor(value, device=env.device)
        self._temperature[env_ids] = torch.clip(value, 0.0, 1.0)
        for env_id in env_ids:
            self.set_joint_state(
                min=self._temperature[env_id],
                max=self._temperature[env_id],
                env=env,
                joint_names=[self._joint_names["temperature"]],
                env_ids=torch.tensor([env_id])
            )

    def slide_rack(self, env, value=1.0, rack_level=0, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=env.device)
        for level in [rack_level, 0]:
            joint = f"rack{level}_joint"
            if joint in env.scene.articulations[self.name].data.joint_names:
                name = f"rack{level}"
                if name not in self._rack:
                    self._rack[name] = torch.tensor([value], device=env.device).repeat(env.num_envs)
                else:
                    self._rack[name][env_ids] = value
                self.set_joint_state(min=value, max=value, env=env, env_ids=env_ids, joint_names=[joint])
                return

        raise ValueError(
            f"No rack found for level {rack_level}, and fallback to 0 also failed."
        )

    def update_state(self, env):
        for name in ["door", "timer", "temperature"]:
            jn = self._joint_names[name]
            if jn in env.scene.articulations[self.name].data.joint_names:
                val = self.get_joint_state(env, [jn])[jn]
                setattr(self, f"_{name}", val)

        self._rack.clear()
        for joint_name in env.scene.articulations[self.name].data.joint_names:
            if "rack" in joint_name:
                val = self.get_joint_state(env, [joint_name])[joint_name]
                key = joint_name.replace("_joint", "")
                self._rack[key] = val
                if key.endswith("rack0"):
                    self._rack["rack0"] = val
                elif key.endswith("rack1"):
                    self._rack["rack1"] = val

    def check_rack_contact(self, env, object_name, rack_level=0):
        """
        Checks whether the specified object is in contact with the oven rack at the given level.

        Args:
            env: The simulation environment.
            object_name (str): Name of the object to check for contact.
            rack_level (int): Which rack level to check (default is 0).

        Returns:
            bool: True if contact exists between object and rack, False otherwise.
        """
        for level in [rack_level, 0]:
            joint = f"rack{level}_joint"
            if joint in env.scene.articulations[self.name].data.joint_names:
                contact_name = joint.replace("_joint", "")
                break
        else:
            raise RuntimeError(f"No rack found for level {rack_level}")
        return OU.check_contact(env, object_name, str(self.rack_infos[contact_name][0].GetPrimPath()))

    def get_state(self, rack_level=0):
        state = {}
        for key in [f"rack{rack_level}", "rack0"]:
            if key in self._rack:
                state[key] = self._rack[key]
                break
        else:
            raise ValueError(f"No rack state available for level {rack_level}")

        state["door"] = self._door
        state["timer"] = self._timer
        state["temperature"] = self._temperature
        return state

    def has_multiple_rack_levels(self):
        """
        Returns True if the oven has multiple rack levels, False if only one.
        """
        rack_levels = set()
        for key in self._regions:
            m = re.fullmatch(r"rack(\d+)", key)
            if m:
                idx = int(m.group(1))
                rack_levels.add(idx)
        return len(rack_levels) > 1

    @cached_property
    def rack_infos(self):
        rack_infos = {}
        for name in ["rack0", "rack1"]:
            for prim_path in self.prim_paths:
                prims = usd.get_prim_by_suffix(self._env.sim.stage.GetObjectAtPath(prim_path), name)
                for prim in prims:
                    if prim is not None and prim.IsValid():
                        if name not in rack_infos:
                            rack_infos[name] = [prim]
                        else:
                            rack_infos[name].append(prim)
        assert "rack0" in rack_infos, f"Oven rack0 not found!"
        return rack_infos

    def get_reset_region_names(self):
        return ("rack0", "rack1")

    def get_reset_regions(self, env=None, rack_level=0, z_range=None):
        rack_regions = []

        for key, reg in self._regions.items():
            m = re.fullmatch(r"rack(\d+)", key)
            if not m:
                continue
            idx = int(m.group(1))
            p0, px, py, pz = reg["p0"], reg["px"], reg["py"], reg["pz"]
            hx, hy, hz = (px[0] - p0[0]) / 2, (py[1] - p0[1]) / 2, (pz[2] - p0[2]) / 2
            center = p0 + np.array([hx, hy, hz])
            entry = (idx, key, p0, px, py, pz)
            rack_regions.append(entry)

        rack_regions.sort(key=lambda x: x[0])
        region = (
            rack_regions[1]
            if rack_level == 1 and len(rack_regions) > 1
            else rack_regions[0]
            if rack_regions
            else None
        )

        if region:
            level = region[0]
            self._joint_names["rack"] = f"rack{level}_joint"
            self._rack[f"rack{level}"] = torch.zeros(self.num_envs, device=self.device)
        else:
            raise ValueError(f"No rack reset regions found for rack_level {rack_level}")

        idx, key, p0, px, py, pz = region
        offset = (
            float(np.mean((p0[0], px[0]))),
            float(np.mean((p0[1], py[1]))),
            float(p0[2]),
        )
        size = (float(px[0] - p0[0]), float(py[1] - p0[1]))
        height = float(pz[2] - p0[2])
        return {key: {"offset": offset, "size": size, "height": height}}
