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

from typing import Any

import numpy as np
import torch

from lw_benchhub.core.mdp.config import Config
from lw_benchhub.core.mdp.helpers.KF import ESEKF, IMUEKF, IMUKF
from lw_benchhub.core.mdp.helpers.policy_unified import LocoLowLevelPolicy, SquatLowLevelPolicy


kf = IMUKF()
ekf = IMUEKF()
esekf = ESEKF(dt=1. / 50.)

torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)


class Controller:
    def __init__(self, config: Config, args: Any) -> None:
        self.config = config

        # Initializing process variables
        self.qj = np.zeros(config.num_dof, dtype=np.float32)
        self.dqj = np.zeros(config.num_dof, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.obs = np.zeros(config.num_obs, dtype=np.float32)

        # Initializing process variables
        self.last_low_action = np.zeros(config.num_dof, dtype=np.float32)
        self.kps = config.kps
        self.kds = config.kds

        self.counter = 0
        self.transition_count = 0

    def set_transition_count(self):
        self.transition_count = self.config.transition_time

    def reset(self):

        self.counter = 0
        self.transition_count = 0
        self.qj = np.zeros(self.config.num_dof, dtype=np.float32)
        self.dqj = np.zeros(self.config.num_dof, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.action = np.zeros(self.config.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.config.num_obs, dtype=np.float32)
        self.last_low_action = np.zeros(self.config.num_dof, dtype=np.float32)


class Controller_loco(Controller):
    def __init__(self, config: Config, args: Any) -> None:
        self.config = config
        super().__init__(config, args)
        self.loco_low_level_policy = LocoLowLevelPolicy(self.config)
        self.loco_cmd = np.zeros(3, dtype=np.float32)
        self.stance_command = False
        self.transition_count = 0
        self.last_policy_target_dof_pos = np.zeros(config.num_dof, dtype=np.float32)

    def compute_loco_cmd(self, cmd_raw):
        if cmd_raw is None:
            return
        cmd = cmd_raw * self.config.max_cmd
        cmd[1] = cmd[1] + 0.1
        cmd_mask = np.abs(cmd) >= self.config.cmd_clip
        self.loco_cmd = cmd * cmd_mask
        if self.stance_command:
            self.loco_cmd *= 0

    def run(self, cmd_raw, gravity_orientation, omega, qj_obs, dqj_obs, target_dof_pos):
        self.compute_loco_cmd(cmd_raw)
        self.loco_low_level_policy.gait_planner.update_gait_phase(self.stance_command)
        self.obs, self.action, target_dof_pos[self.config.action_idx] = self.loco_low_level_policy.inference(
            self.loco_cmd, gravity_orientation, omega, qj_obs[self.config.dof_idx], dqj_obs[self.config.dof_idx])
        return target_dof_pos

    def reset(self):
        super().reset()
        self.loco_low_level_policy.reset()

    def reset_gait(self):
        self.loco_low_level_policy.reset_gait()


class Controller_squat(Controller):
    def __init__(self, config: Config, args: Any) -> None:
        self.config = config
        super().__init__(config, args)
        self.squat_low_level_policy = SquatLowLevelPolicy(self.config)
        self.squat_cmd = np.array([0.75, 0.], dtype=np.float32)

    def compute_squat_cmd(self, cmd_raw):
        if cmd_raw is None:
            return
        d_hp = cmd_raw * self.config.max_cmd / 250
        self.squat_cmd += d_hp
        default_h = 0.75
        default_p = 0.0
        self.squat_cmd[0] = np.clip(self.squat_cmd[0], default_h - self.config.max_cmd[0], default_h)
        self.squat_cmd[1] = np.clip(self.squat_cmd[1], default_p, default_p + self.config.max_cmd[1])

    def run(self, cmd_raw, gravity_orientation, omega, qj_obs, dqj_obs, target_dof_pos):
        self.compute_squat_cmd(cmd_raw)
        self.obs, self.action, target_dof_pos[self.config.action_idx] = self.squat_low_level_policy.inference(
            self.squat_cmd, gravity_orientation, omega, qj_obs[self.config.dof_idx], dqj_obs[self.config.dof_idx])
        return target_dof_pos

    def reset(self):
        super().reset()
        self.squat_cmd = np.array([0.75, 0.], dtype=np.float32)
        self.squat_low_level_policy.reset()
