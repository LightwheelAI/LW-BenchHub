# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from lw_benchhub.utils.math_utils.transform_utils import convert_quat, make_pose

from .wbc_policy.config.configs import BaseConfig
from .wbc_policy.policy.g1_wbc_upperbody_controller import G1WBCUpperbodyController
from .wbc_policy.utils.g1 import instantiate_g1_robot_model

USE_P_CONTROL = False


class G1Action(ActionTerm):

    cfg: "G1ActionCfg"

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "G1ActionCfg", env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)

        # Avoid indexing across all joints for efficiency
        if (self._num_joints == self._asset.num_joints and
                not self.cfg.preserve_order):
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros([self.num_envs, self._num_joints], device=self.device)
        self._target_robot_joints_mujoco = None

        config = BaseConfig()

        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        self.robot_model = instantiate_g1_robot_model(waist_location=waist_location)

        self.current_upper_body_pose = self.robot_model.get_initial_upper_body_pose()

        self.upperbody_controller = G1WBCUpperbodyController(
            robot_model=self.robot_model,
            body_active_joint_groups=["arms"],
            control_hands=False,
        )

    # Properties.
    @property
    def num_joints(self) -> int:
        """Get the number of joints."""
        return self._num_joints

    @property
    def target_robot_joints_mujoco(self) -> torch.Tensor:
        """Get the target robot joints mujoco tensor."""
        return self._target_robot_joints_mujoco

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        return 14

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    # """
    # Operations.
    # """
    def save_check_point(self):
        return self.upperbody_controller.body_ik_solver.save_configuration_q()

    def load_check_point(self, q):
        self.upperbody_controller.body_ik_solver.load_configuration_q(q)

    def compute_upperbody_joint_positions(self, body_data):
        if self.upperbody_controller.in_warmup:
            for _ in range(50):
                target_robot_joints = self.upperbody_controller.inverse_kinematics(body_data, only_body=True)

            self.upperbody_controller.in_warmup = False
        else:
            target_robot_joints = self.upperbody_controller.inverse_kinematics(body_data, only_body=True)

        return target_robot_joints

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """
        # Store the raw actions
        self._raw_actions[:] = actions[:, :self.action_dim]

        # Extract arm poses directly from actions tensor
        # Handle both single and multi-environment cases
        if actions.dim() == 2:  # Multi-environment case: [num_envs, action_dim]
            left_arm_pos = actions[:, :3]  # [num_envs, 3]
            left_arm_quat = actions[:, 3:7]  # [num_envs, 4]
            right_arm_pos = actions[:, 7:10]  # [num_envs, 3]
            right_arm_quat = actions[:, 10:14]  # [num_envs, 4]
        else:  # Single environment case: [action_dim]
            left_arm_pos = actions[:3].unsqueeze(0)  # [1, 3]
            left_arm_quat = actions[3:7].unsqueeze(0)  # [1, 4]
            right_arm_pos = actions[7:10].unsqueeze(0)  # [1, 3]
            right_arm_quat = actions[10:14].unsqueeze(0)  # [1, 4]

        # Convert quaternions from wxyz (IsaacLab) to xyzw (scipy) format
        # Handle batch quaternion conversion
        if left_arm_quat.dim() == 2:  # Multi-environment case: [num_envs, 4]
            left_arm_quat_xyzw = torch.zeros_like(left_arm_quat)
            right_arm_quat_xyzw = torch.zeros_like(right_arm_quat)
            for i in range(left_arm_quat.shape[0]):
                left_arm_quat_xyzw[i] = convert_quat(left_arm_quat[i], to="xyzw")
                right_arm_quat_xyzw[i] = convert_quat(right_arm_quat[i], to="xyzw")
        else:  # Single environment case: [4]
            left_arm_quat_xyzw = convert_quat(left_arm_quat, to="xyzw")
            right_arm_quat_xyzw = convert_quat(right_arm_quat, to="xyzw")

        # Create rotation matrices and poses (minimize CPU-GPU transfers)
        left_rotmat = R.from_quat(left_arm_quat_xyzw.cpu().numpy()).as_matrix()
        right_rotmat = R.from_quat(right_arm_quat_xyzw.cpu().numpy()).as_matrix()

        # Process each environment separately
        num_envs = left_rotmat.shape[0] if left_rotmat.ndim == 3 else 1
        all_target_joints = []
        for env_idx in range(num_envs):
            # Get data for this environment
            if left_rotmat.ndim == 3:
                env_left_rotmat = left_rotmat[env_idx]
                env_right_rotmat = right_rotmat[env_idx]
                env_left_pos = left_arm_pos[env_idx]
                env_right_pos = right_arm_pos[env_idx]
            else:
                env_left_rotmat = left_rotmat
                env_right_rotmat = right_rotmat
                env_left_pos = left_arm_pos[0]
                env_right_pos = right_arm_pos[0]

            # Create poses for this environment
            left_arm_pose = make_pose(env_left_pos, torch.tensor(env_left_rotmat, device=self.device))
            right_arm_pose = make_pose(env_right_pos, torch.tensor(env_right_rotmat, device=self.device))

            # Run PINK IK to get target upper body joint positions for this environment
            body_data = {
                "left_wrist_yaw_link": left_arm_pose.cpu().numpy(),
                "right_wrist_yaw_link": right_arm_pose.cpu().numpy()
            }
            target_robot_joints_mujoco = self.compute_upperbody_joint_positions(body_data)
            all_target_joints.append(target_robot_joints_mujoco)

        # Convert to tensor and store
        if num_envs > 1:
            self._processed_actions = torch.tensor(
                np.array(all_target_joints),
                device=self.device,
                dtype=torch.float32
            )
        else:
            self._processed_actions = torch.tensor(
                all_target_joints[0],
                device=self.device,
                dtype=torch.float32
            ).unsqueeze(0)

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
        self.upperbody_controller.body_ik_solver.reset()


@configclass
class G1ActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = G1Action
    """Specifies the action term class type for G1 WBC action."""

    preserve_order: bool = True
    joint_names: list[str] = MISSING
