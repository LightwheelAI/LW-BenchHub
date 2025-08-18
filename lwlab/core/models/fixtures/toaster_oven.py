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
import time
from functools import cached_property

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv

from .fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd


class ToasterOven(Fixture):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
        self._door = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)
        self._doneness = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)
        self._function = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)
        self._temperature = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)
        self._time = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)
        self._rack = {}
        self._tray = {}

        self._door_target = [None] * cfg.num_envs
        self._last_time_update = [None] * cfg.num_envs
        self._joint_names = {
            "door": f"door_joint",
            "doneness": f"knob_doneness_joint",
            "function": f"knob_function_joint",
            "temperature": f"knob_temp_joint",
            "time": f"knob_time_joint",
            "rack": f"rack0_joint",
            "tray": f"tray0_joint",
        }

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def set_doneness(self, env, val, env_ids=None):
        val = val if isinstance(val, torch.Tensor) else torch.tensor(val, device=env.device)
        self._doneness[env_ids] = torch.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["doneness"]], env_ids=env_ids
        )

    def set_function(self, env, val, env_ids=None):
        val = val if isinstance(val, torch.Tensor) else torch.tensor(val, device=env.device)
        self._function[env_ids] = torch.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["function"]], env_ids=env_ids
        )

    def set_temperature(self, env, val, env_ids=None):
        val = val if isinstance(val, torch.Tensor) else torch.tensor(val, device=env.device)
        self._temperature[env_ids] = torch.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["temperature"]], env_ids=env_ids
        )

    def set_time(self, env, val, env_ids=None):
        val = val if isinstance(val, torch.Tensor) else torch.tensor(val, device=env.device)
        self._time[env_ids] = torch.clip(val, 0.0, 1.0)
        self.set_joint_state(
            env=env, min=val, max=val, joint_names=[self._joint_names["time"]], env_ids=env_ids
        )

    def open_door(self, env, min=1.0, max=1.0, env_ids=None):
        """
        helper function to open the door. calls set_door_state function
        """
        super().open_door(env=env, min=min, max=max, env_ids=env_ids)

    def slide_rack(self, env, value=1.0, rack_level=0, env_ids=None):
        """
        Slides the rack/tray at the specified level, with fallback to level 0 if the target level doesn't exist.

        Args:
            env: The environment object
            value (float): normalized value between 0 (closed) and 1 (open)
            rack_level (int): which level to target (0 = bottom, 1 = top)
        """
        door_pos = self.get_joint_state(env, [self._joint_names["door"]])[
            self._joint_names["door"]
        ]
        for env_id in env_ids:
            if door_pos[env_id] <= 0.99:
                self.open_door(env, env_ids=torch.tensor([env_id]))

        # Try rack{level}, fallback to rack0
        for level in [rack_level, 0]:
            joint = f"rack{level}_joint"
            if joint in env.scene.articulations[self.name].data.joint_names:
                name = f"rack{level}"
                if name not in self._rack:
                    self._rack[name] = torch.tensor([value], device=env.device).repeat(env.num_envs)
                else:
                    self._rack[name][env_ids] = value
                self.set_joint_state(env=env, min=value, max=value, joint_names=[joint], env_ids=env_ids)
                return name

        raise ValueError(
            f"No rack or tray found for level {rack_level}, and fallback to level 0 also failed."
        )

    def get_joint_state(self, env, joint_names, env_ids=None):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps door names to a percentage of how open they are
        """
        joint_state = dict()
        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, joint_idx]
            init_joint_qpos = env.scene.articulations[self.name].data.default_joint_pos[:, joint_idx]
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]
            norm_qpos = abs(joint_qpos - init_joint_qpos) / (joint_max - joint_min)
            joint_state[j_name] = norm_qpos

        return joint_state

    def set_joint_state(self, min, max, env, joint_names, env_ids=None, rng=None):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is
        Args:
            min (float): minimum percentage of how open the door is
            max (float): maximum percentage of how open the door is
            env (ManagerBasedRLEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        rng = self.rng if rng is None else rng
        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]
            desired_min = joint_min + (joint_max - joint_min) * min
            desired_max = joint_min + (joint_max - joint_min) * max
            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[rng.uniform(float(desired_min), float(desired_max))]]).to(env.device),
                torch.tensor([joint_idx]).to(env.device),
                torch.tensor(env_ids).to(env.device) if env_ids is not None else None
            )

    def update_state(self, env):
        if not isinstance(self._rack, dict):
            self._rack = {}
        if not isinstance(self._tray, dict):
            self._tray = {}

        # Clear canonical keys to avoid stale values
        for k in ["rack0", "rack1"]:
            self._rack.pop(k, None)
        for k in ["tray0", "tray1"]:
            self._tray.pop(k, None)

        # Check every joint in the model for matching rack/tray joints
        for joint_name in env.scene.articulations[self.name].data.joint_names:
            if "rack" in joint_name:
                val = self.get_joint_state(env, [joint_name])[joint_name]
                key = joint_name.replace("_joint", "")
                self._rack[key] = val
                if key.endswith("rack0"):
                    self._rack["rack0"] = val
                elif key.endswith("rack1"):
                    self._rack["rack1"] = val
            elif "tray" in joint_name:
                val = self.get_joint_state(env, [joint_name])[joint_name]
                key = joint_name.replace("_joint", "")
                self._tray[key] = val
                if key.endswith("tray0"):
                    self._tray["tray0"] = val
                elif key.endswith("tray1"):
                    self._tray["tray1"] = val

        # Update non-rack/tray joints from predefined names
        for name, jn in self._joint_names.items():
            if name in ["rack", "tray"]:
                continue  # already handled above
            if jn in env.scene.articulations[self.name].data.joint_names:
                val = self.get_joint_state(env, [jn])[jn]
                for env_id in range(env.num_envs):
                    if name == "time":
                        if val[env_id] > 0:
                            if self._last_time_update[env_id] is None:
                                self._last_time_update[env_id] = time.time()
                            else:
                                elapsed = time.time() - self._last_time_update[env_id]
                                self._last_time_update[env_id] = time.time()
                                new_val = val[env_id] - elapsed / 3000
                                new_val = torch.where(new_val < 0, 0, new_val)
                                self.set_joint_state(
                                    env=env, min=new_val, max=new_val, joint_names=[jn], env_ids=[env_id]
                                )
                        else:
                            self._last_time_update[env_id] = None
                setattr(self, f"_{name}", val)

    def check_rack_contact(self, env, obj_name, rack_level=0):
        """
        Checks whether object is touching the specified rack or tray level.
        If level 1 is requested but only level 0 exists, falls back to level 0.

        Args:
            env: The simulation environment
            obj_name (str): Name of the object to check contact with
            rack_level (int): 0 for bottom, 1 for top (falls back to 0 if top does not exist)
        """

        for level in [rack_level, 0]:
            joint = f"rack{level}_joint"
            if joint in env.scene.articulations[self.name].data.joint_names:
                contact_name = joint.replace("_joint", "")
                break
        else:
            for level in [rack_level, 0]:
                joint = f"tray{level}_joint"
                if joint in env.scene.articulations[self.name].data.joint_names:
                    contact_name = joint.replace("_joint", "")
                    break
            else:
                raise RuntimeError(
                    f"No rack or tray found at level {rack_level}, and fallback to level 0 failed."
                )
        if "rack" in contact_name:
            contact_path = self.rack_infos[contact_name][0].GetPrimPath()
        elif "tray" in contact_name:
            contact_path = self.tray_infos[contact_name][0].GetPrimPath()
        else:
            raise ValueError(f"Invalid contact name: {contact_name}")
        return env.check_contact(obj_name, str(contact_path))

    def get_state(self, env, rack_level=0):
        """
        Returns the state of the toaster oven.

        Args:
            env: the environment
            rack_level (int): 0 (bottom) or 1 (top). If 1 is not available, falls back to 0.

        Returns:
            dict: current state of the selected rack or tray level
        """
        state = {}

        target_keys = [f"rack{rack_level}", f"tray{rack_level}"]
        fallback_keys = ["rack0", "tray0"]

        for key in target_keys + fallback_keys:
            if key in self._rack:
                state[key] = self._rack.get(key, None)
                break
            elif key in self._tray:
                state[key] = self._tray.get(key, None)
                break
        else:
            raise ValueError(
                f"No rack or tray state available for rack_level={rack_level}"
            )

        for name in ["door", "doneness", "function", "temperature", "time"]:
            state[name] = getattr(self, f"_{name}", None)

        return state

    @property
    def nat_lang(self):
        return "toaster oven"

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
        assert "rack0" in rack_infos, f"Toaster oven rack0 not found!"
        return rack_infos

    @cached_property
    def tray_infos(self):
        tray_infos = {}
        for name in ["tray0", "tray1"]:
            for prim_path in self.prim_paths:
                prims = usd.get_prim_by_suffix(self._env.sim.stage.GetObjectAtPath(prim_path), name)
                for prim in prims:
                    if prim is not None and prim.IsValid():
                        if name not in tray_infos:
                            tray_infos[name] = [prim]
                        else:
                            tray_infos[name].append(prim)
        assert "tray0" in tray_infos, f"Toaster oven tray0 not found!"
        return tray_infos
