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

from collections.abc import Sequence
from dataclasses import field
from typing import Callable, List, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from lw_benchhub.core.mdp.actions.controller import Controller_loco, Controller_squat
from lw_benchhub.core.mdp.config import Config
from lw_benchhub.core.mdp.helpers.KF import IMUKF
from lw_benchhub.core.mdp.helpers.rotation_helper import get_gravity_orientation


class LegPositionAction(ActionTerm):
    cfg: "LegPositionActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "LegPositionActionCfg", env: ManagerBasedEnv):
        self.decimation = env.cfg.decimation
        self.apply_action_count = 0
        self.last_target_dof_pos = None
        super().__init__(cfg, env)
        self.joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._body_name = self.cfg.body_name
        self.loco_joint_ids = [9, 13, 17, 21, 25, 33, 10, 14, 18, 22, 26, 34]
        self.squat_joint_ids = [9, 13, 17, 21, 25, 33,
                                10, 14, 18, 22, 26, 34,
                                6, 3, 0,
                                1, 4, 7, 11, 15, 19, 23,
                                2, 5, 8, 12, 16, 20, 24]

        loco_config_path = f"lw_benchhub/core/mdp/configs/{cfg.loco_config}"
        self.loco_config = Config(loco_config_path)
        squat_config_path = f"lw_benchhub/core/mdp/configs/{cfg.squat_config}"
        self.squat_config = Config(squat_config_path)

        self.loco_controller = Controller_loco(self.loco_config, None)
        self.squat_controller = Controller_squat(self.squat_config, None)

        self._current_cmd = [0.0, 0.0, 0.0]
        self.gravity_kf = IMUKF(process_noise=1e-4, measurement_noise=1e-2)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.ang_vel = np.zeros(3)

        self.loco_qj = np.zeros(12, dtype=np.float32)
        self.loco_dqj = np.zeros(12, dtype=np.float32)
        self.squat_qj = np.zeros(29, dtype=np.float32)
        self.squat_dqj = np.zeros(29, dtype=np.float32)

        self.count_cmd_zero = 0
        self.target_dof_pos = self.loco_config.default_angles.copy()

        self.policy_mode = "squat"  # 'loco' or 'squat'

        self.reset()

    def update_imu_data(self, quaternion: np.ndarray, gyroscope: np.ndarray) -> None:
        self.quat = quaternion
        self.ang_vel = gyroscope

    def update_joint_state(self, qj: np.ndarray, dqj: np.ndarray) -> None:
        self.qj = qj
        self.dqj = dqj

    def set_mode(self, policy_mode: str, stance_mode: bool = False) -> None:
        self.policy_mode = policy_mode
        self.stance_mode = stance_mode

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = torch.zeros(
            actions.shape[0], actions.shape[1],
            device=actions.device, dtype=actions.dtype
        )

        self._raw_actions[:] = actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_ = list(env_ids) if env_ids is not None else list(range(self.num_envs))

        joint_count = len(self.loco_joint_ids)

        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)

        processed_actions = torch.zeros(len(env_ids_), joint_count, device=self.device)

        if env_ids is None:
            self._processed_actions = processed_actions
        else:
            self._processed_actions[env_ids, :] = processed_actions

        self.loco_controller.reset()
        self.squat_controller.reset()

        self.gravity_kf = IMUKF(process_noise=1e-4, measurement_noise=1e-2)
        self.policy_mode = "squat"
        self.stance_mode = False

        self.loco_qj = np.zeros(12, dtype=np.float32)
        self.loco_dqj = np.zeros(12, dtype=np.float32)
        self.squat_qj = np.zeros(29, dtype=np.float32)
        self.squat_dqj = np.zeros(29, dtype=np.float32)

        self.target_dof_pos = self.loco_config.default_angles.copy()

    def refresh_prop(self):
        joint_pos = self._asset.data.joint_pos[:, self.loco_joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self.loco_joint_ids]

        self.loco_qj = joint_pos.cpu().numpy()[0]
        self.loco_dqj = joint_vel.cpu().numpy()[0]
        squat_joint_pos = self._asset.data.joint_pos[:, self.squat_joint_ids]
        squat_joint_vel = self._asset.data.joint_vel[:, self.squat_joint_ids]
        self.squat_qj = squat_joint_pos.cpu().numpy()[0]
        self.squat_dqj = squat_joint_vel.cpu().numpy()[0]

        body_idx = self._asset.find_bodies("pelvis")[0]
        quat = self._asset.data.body_link_quat_w[:, body_idx]
        ang_vel = self._asset.data.body_link_ang_vel_w[:, body_idx]

        quat_np = quat.cpu().numpy()[0][0]  # [w, x, y, z]
        ang_vel_np = ang_vel.cpu().numpy()[0].flatten()  # [wx, wy, wz]

        quat_xyzw = np.array([quat_np[1], quat_np[2], quat_np[3], quat_np[0]])
        r = R.from_quat(quat_xyzw)

        ang_vel_root = r.inv().apply(ang_vel_np)

        self.quat = quat_np
        self.ang_vel = ang_vel_root

    def check_mode_switch(self, cmd_raw):

        if cmd_raw[3]:
            if self.policy_mode != "squat":
                self.policy_mode = "squat"
                self.squat_controller.set_transition_count()
                self.loco_controller.reset_gait()
                return True
        else:
            if self.policy_mode != "loco":
                self.policy_mode = "loco"
                self.loco_controller.set_transition_count()
                return True
        return False

    def transition_squat(self):
        if self.squat_controller.transition_count > 0:
            gravity_orientation = get_gravity_orientation(self.quat)
            gravity_orientation = self.gravity_kf.update(gravity_orientation)
            cmd_raw = None
            other_policy_target_dof_pos = self.loco_controller.run(
                cmd_raw, gravity_orientation, self.ang_vel, self.loco_qj, self.loco_dqj, self.target_dof_pos.copy())
            alpha = self.squat_controller.transition_count / self.squat_config.transition_time
            squat_policy_target_dof_pos = self.target_dof_pos.copy()
            self.target_dof_pos = alpha * other_policy_target_dof_pos + (1 - alpha) * squat_policy_target_dof_pos
            self.squat_controller.transition_count -= 1
            return True
        return False

    def apply_actions(self) -> None:
        cmd_raw = np.zeros(4, dtype=np.float32)
        for i in range(min(4, self._raw_actions.shape[1])):
            cmd_raw[i] = self._raw_actions[0, i].item()
        self.refresh_prop()
        if self.check_mode_switch(cmd_raw):
            self.transition_squat()
        if self.policy_mode == "loco":
            current_qj = self.loco_qj
            current_dqj = self.loco_dqj
        elif self.policy_mode == "squat":
            current_qj = self.squat_qj
            current_dqj = self.squat_dqj

        gravity_orientation = get_gravity_orientation(self.quat)
        gravity_orientation = self.gravity_kf.update(gravity_orientation)

        self.loco_controller.stance_command = self.stance_mode
        self._setup_policy_control(cmd_raw, gravity_orientation, current_qj, current_dqj)
        self.apply_action_count += 1

    def _setup_policy_control(self, cmd_raw: np.ndarray, gravity_orientation: np.ndarray, current_qj: np.ndarray, current_dqj: np.ndarray) -> None:
        if self.apply_action_count % self.decimation == 0:
            if self.policy_mode == "squat":
                if self.squat_controller.transition_count > 0:
                    self.transition_squat()
                else:
                    self.target_dof_pos = self.squat_controller.run(
                        [cmd_raw[0], 0],
                        gravity_orientation,
                        self.ang_vel,
                        current_qj,
                        current_dqj,
                        self.target_dof_pos.copy()
                    )
            elif self.policy_mode == "loco":
                self.target_dof_pos = self.loco_controller.run(
                    cmd_raw[0:3],
                    gravity_orientation,
                    self.ang_vel,
                    current_qj,
                    current_dqj,
                    self.target_dof_pos.copy()
                )
            del self.last_target_dof_pos
            self.last_target_dof_pos = self.target_dof_pos.copy()
        else:
            self.target_dof_pos = self.last_target_dof_pos
        self._processed_actions[:] = torch.from_numpy(self.target_dof_pos).to(self._processed_actions.device)
        zero_pos = torch.zeros(len(self.squat_joint_ids), dtype=torch.float32, device=self._asset.device)
        if self.policy_mode == "loco":
            self._asset.set_joint_position_target(zero_pos, self.squat_joint_ids)
        self._asset.set_joint_position_target(self._processed_actions, self.loco_joint_ids)

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions


@configclass
class LegPositionActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = LegPositionAction
    num_dof: int = 12
    joint_names: List[str] = field(default_factory=list)
    body_name: str = ""
    scale: float = 1.0
    loco_config: Optional[str] = None
    squat_config: Optional[str] = None
    post_process_fn: Optional[Callable] = None
