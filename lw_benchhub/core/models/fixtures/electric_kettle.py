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

import time

import torch

from .fixture import Fixture
from .fixture_types import FixtureType


class ElectricKettle(Fixture):
    fixture_types = [FixtureType.ELECTRIC_KETTLE]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        joint_names = list(self._joint_infos.keys())
        self._joint_names = {
            "lid": next(iter([name for name in joint_names if "lid_joint" in name.lower() and "button" not in name.lower()]), None),
            "switch": next(iter([name for name in joint_names if "switch_joint" in name.lower()]), None),
            "lid_button": next(iter([name for name in joint_names if "button_lid_joint" in name.lower()]), None),
        }
        self._turned_on = torch.tensor([False], device=self.device).repeat(self.num_envs)
        self._num_steps_on = torch.tensor([0], device=self.device).repeat(self.num_envs)
        self._cooldown_time = torch.tensor([0], device=self.device).repeat(self.num_envs)
        self._last_lid_update = [None] * self.num_envs
        self._target_lid_angle = [None] * self.num_envs
        self._lid_speed = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
        self._lid = torch.tensor([0.0], device=self.device).repeat(self.num_envs)

    def init_state(self, env):
        self._turned_on = torch.tensor([False], device=env.device).repeat(env.num_envs)
        self._num_steps_on = torch.tensor([0], device=env.device).repeat(env.num_envs)
        self._cooldown_time = torch.tensor([0], device=env.device).repeat(env.num_envs)
        self._last_lid_update = [None] * env.num_envs
        self._target_lid_angle = [None] * env.num_envs
        self._lid_speed = torch.tensor([0.0], device=env.device).repeat(env.num_envs)
        self._lid = torch.tensor([0.0], device=env.device).repeat(env.num_envs)

    def set_joint_state(self, min, max, env, joint_names, env_ids=None):
        assert torch.all((0 <= min) & (min <= max) & (max <= 1)), "min and max must satisfy 0 <= min <= max <= 1"

        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]

            # Compute desired qpos from normalized min/max
            if joint_min >= 0:
                qpos_min = joint_min + (joint_max - joint_min) * min
                qpos_max = joint_min + (joint_max - joint_min) * max
            else:
                # For joints with negative ranges (e.g., -0.02 to 0.02)
                qpos_min = joint_max - (joint_max - joint_min) * max
                qpos_max = joint_max - (joint_max - joint_min) * min

            # Choose deterministic midpoint for stability (avoid bouncing)
            desired_qpos = 0.5 * (qpos_min + qpos_max)

            env.scene.articulations[self.name].write_joint_position_to_sim(
                desired_qpos.unsqueeze(1),
                torch.tensor([joint_idx]).to(env.device),
                env_ids=torch.tensor(env_ids, device=env.device) if env_ids is not None else None)

    def set_lid(self, env, lid_val=1.0, gradual=False):
        """
        Sets the state of the lid

        Args:
            lid_val (float): normalized value between 0 (closed) and 1 (open)
            gradual (bool): if True, the lid will move smoothly to its target position over time
        """
        lid_val = torch.tensor([lid_val], device=env.device).repeat(env.num_envs)
        joint_name = self._joint_names["lid"]
        self._lid = torch.clip(lid_val, 0.0, 1.0)

        if gradual:
            self._target_lid_angle = self._lid.tolist()
            self._last_lid_update = [time.time()] * env.num_envs
            self._lid_speed = torch.tensor([1.5], device=env.device).repeat(env.num_envs)
        else:
            self.set_joint_state(
                env=env,
                min=self._lid,
                max=self._lid,
                joint_names=[joint_name],
            )

    def update_state(self, env):
        """
        Update the state of the electric kettle
        """
        # Update all joint states
        for name, jn in self._joint_names.items():
            state = self.get_joint_state(env, [jn])[jn]
            state = torch.clip(state, 0.0, 1.0)
            setattr(self, f"_{name}", state)

        mask = (self._lid < 0.9) & (self._lid_speed == 0.5)
        indices = torch.where(mask)[0].tolist()
        for idx in indices:
            self._target_lid_angle[idx] = None
            self._last_lid_update[idx] = None

        for env_id in range(len(self._lid)):
            if self._target_lid_angle[env_id] is not None and self._last_lid_update[env_id] is not None:
                joint_name = self._joint_names["lid"]
                current_angle = self.get_joint_state(env, [joint_name])[joint_name][env_id: env_id + 1]
                time_elapsed = time.time() - self._last_lid_update[env_id]
                angle_change = min(
                    time_elapsed * self._lid_speed[env_id],
                    abs(self._target_lid_angle[env_id] - current_angle),
                )
                if self._target_lid_angle[env_id] < current_angle:
                    angle_change = -angle_change
                new_angle = current_angle + angle_change
                new_angle = torch.clip(new_angle, 0.0, 1.0)
                self.set_joint_state(
                    env=env, min=new_angle, max=new_angle, joint_names=[joint_name], env_ids=[env_id]
                )

                if abs(new_angle - self._target_lid_angle[env_id]) < 0.001:
                    self._target_lid_angle[env_id] = None
                    self._last_lid_update[env_id] = None
                else:
                    self._last_lid_update[env_id] = time.time()
            if self._lid_button[env_id] > 0.1 and self._lid[env_id] <= 0.01:
                self.set_lid(env, 1.0, gradual=True)

            # Handle switch/power state
            switch_open_val = torch.tensor([1.0], device=env.device).repeat(env.num_envs)

            if self._switch[env_id] >= 0.95 and not self._turned_on[env_id]:
                self._turned_on[env_id] = True
                self._num_steps_on[env_id] += 1
                self.set_joint_state(
                    env=env,
                    min=switch_open_val,
                    max=switch_open_val,
                    joint_names=[self._joint_names["switch"]],
                )
            elif self._turned_on[env_id] and self._num_steps_on[env_id] < 500:
                self._num_steps_on[env_id] += 1
                self.set_joint_state(
                    env=env,
                    min=switch_open_val,
                    max=switch_open_val,
                    joint_names=[self._joint_names["switch"]],
                )
            elif self._turned_on[env_id] and self._num_steps_on[env_id] >= 500:
                self._turned_on[env_id] = False
                self._cooldown_time[env_id] += 1
                self._num_steps_on[env_id] = 0

            if self._cooldown_time[env_id] > 0 and self._cooldown_time[env_id] < 10:
                self._cooldown_time[env_id] += 1
                new_switch_state = self._switch[env_id] - 0.1
                if new_switch_state < 0.0:
                    new_switch_state = 0.0
                new_switch_state = torch.clip(torch.tensor([new_switch_state], device=env.device), 0.0, 1.0)
                self.set_joint_state(
                    env=env,
                    min=new_switch_state,
                    max=new_switch_state,
                    joint_names=[self._joint_names["switch"]],
                )
                self._switch[env_id] = new_switch_state
            elif self._cooldown_time[env_id] >= 10:
                self._cooldown_time[env_id] = 0

            # ensures lid stays open, and doesn't close on its own
            if self._lid[env_id] > 0.90:
                if self._target_lid_angle[env_id] is None:
                    self._target_lid_angle[env_id] = 1.0
                    self._last_lid_update[env_id] = time.time()
                    self._lid_speed[env_id] = 0.5

    def get_state(self, env):
        """
        Returns a dictionary representing the state of the electric kettle.
        """
        st = {}
        for name, jn in self._joint_names.items():
            st[name] = getattr(self, f"_{name}", None)
        st["turned_on"] = self._turned_on
        return st
