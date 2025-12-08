# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .wbc_policy.config.configs import BaseConfig
from .wbc_policy.policy.g1_wbc_upperbody_controller import G1WBCUpperbodyController
from .wbc_policy.policy.wbc_policy_factory import get_wbc_policy
from .wbc_policy.utils.g1 import instantiate_g1_robot_model

USE_P_CONTROL = False
from .wbc_policy.run_policy import prepare_observations, postprocess_actions, convert_sim_joint_to_wbc_joint

from dataclasses import MISSING
from lw_benchhub.data import LW_BENCHHUB_DATA_PATH


class G1DecoupledWBCAction(ActionTerm):

    cfg: "G1DecoupledWBCActionCfg"

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "G1DecoupledWBCActionCfg", env: ManagerBasedEnv):
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
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        self._processed_actions = torch.zeros([self.num_envs, self._num_joints], device=self.device)
        self._target_robot_joints_mujoco = None

        self._navigate_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._torso_orientation_rpy_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # Note: below are for mimic x WBC only
        self._is_navigating = False
        self._navigation_goal_reached = False
        self._default_base_height_cmd = 0.75

        # self._wbc_version = "homie"
        self._wbc_version = "homie_v2"
        config = BaseConfig()
        if self._wbc_version == 'homie':
            config.wbc_version = "homie"
            config.wbc_model_path = str(LW_BENCHHUB_DATA_PATH / "ckpts/nv_wbc_v0828/gear_decoupled_homie_wbc_v1.onnx")

        elif self._wbc_version == 'homie_v2':
            ckpt_path = LW_BENCHHUB_DATA_PATH / "ckpts/nv_wbc_v0904/homie_v2/"
            config.wbc_version = "homie_v2"
            config.wbc_model_path = ",".join([str(ckpt_path / "stand.onnx"), str(ckpt_path / "walk.onnx")])
        else:
            raise ValueError(f"Invalid WBC version: {self._wbc_version}")

        wbc_config = config.load_wbc_yaml()
        waist_location = "lower_and_upper_body" if config.enable_waist else "lower_body"
        self.robot_model = instantiate_g1_robot_model(waist_location=waist_location)

        self.current_upper_body_pose = self.robot_model.get_initial_upper_body_pose()
        self.wbc_policy = get_wbc_policy("g1", self.robot_model, wbc_config, config.upper_body_joint_speed)

        self._wbc_goal = {
            "target_upper_body_pose": np.tile(self.current_upper_body_pose, (self.num_envs, 1)),
            # lin_vel_cmd_x, lin_vel_cmd_y, ang_vel_cmd
            "navigate_cmd": np.tile(np.array([[0., 0., 0.]]), (self.num_envs, 1)),
            # base_height_cmd: 0.75 as pelvis height
            "base_height_command": np.tile(np.array([self._default_base_height_cmd]), (self.num_envs, 1)),
            # toggle_policy_action: 0 for disable policy, 1 for enable policy
            # Note: always enabled in the policy
            "toggle_policy_action": np.tile(np.array([0]), (self.num_envs, 1)),
        }
        if self._wbc_version == "homie_v2":
            self._wbc_goal["torso_orientation_rpy_cmd"] = np.array([[0., 0., 0.]]).reshape([self.num_envs, -1])

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
    def is_navigating(self) -> bool:
        """Get the is navigating tensor."""
        return self._is_navigating

    @property
    def navigate_cmd(self):
        return self._navigate_cmd

    @property
    def torso_orientation_rpy_cmd_dim(self) -> int:
        """Dimension of torso orientation rpy command."""
        return 3

    @property
    def navigation_goal_reached(self) -> bool:
        """Get the navigation goal reached tensor."""
        return self._navigation_goal_reached

    @property
    def navigate_cmd_dim(self) -> int:
        """Dimension of navigation command."""
        return 3

    @property
    def base_height_cmd_dim(self) -> int:
        """Dimension of base height command."""
        return 1

    @property
    def torso_orientation_rpy_cmd(self):
        return self._torso_orientation_rpy_cmd

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        # 7 for left arm pos/quat, 7 for right arm pos/quat, 3 for navigate_cmd, 1 for base_height_cmd, 3 for torso_orientation_rpy_cmd
        return 7 + 7 + 3 + 1 + 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def get_wbc_version(self):
        return self._wbc_version

    @property
    def get_wbc_policy(self):
        return self.wbc_policy

    @property
    def get_wbc_goal(self):
        return self._wbc_goal

    def set_wbc_goal(self, navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd=None):
        self._wbc_goal["navigate_cmd"] = navigate_cmd
        self._wbc_goal["base_height_command"] = base_height_cmd
        if self._wbc_version == "homie_v2" and torso_orientation_rpy_cmd is not None:
            self._wbc_goal["torso_orientation_rpy_cmd"] = torso_orientation_rpy_cmd

    # """
    # Constants.
    # """

    with open(os.path.join(os.path.dirname(__file__), "wbc_policy/config/loco_manip_g1_joints_order_43dof.yaml"), "r") as f:
        wbc_g1_joints_order = yaml.safe_load(f)
    # """
    # Operations.
    # """

    def compute_upperbody_joint_positions(self, body_data):
        if self.upperbody_controller.in_warmup:
            for _ in range(50):
                target_robot_joints = self.upperbody_controller.inverse_kinematics(body_data)

            self.upperbody_controller.in_warmup = False
        else:
            target_robot_joints = self.upperbody_controller.inverse_kinematics(body_data)

        return target_robot_joints

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """

        # Store the raw actions
        self._raw_actions[:] = actions[:, :self.action_dim]

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()

        '''
        **************************************************
        WBC lower body commands
        **************************************************
        '''
        # Extract navigate_cmd, base_height_cmd from actions
        navigate_cmd = actions_clone[:, -7:-4]
        base_height_cmd = actions_clone[:, -4]
        torso_orientation_rpy_cmd = actions_clone[:, -3:]

        self._navigate_cmd = torch.tensor(navigate_cmd)
        self._torso_orientation_rpy_cmd = torch.tensor(torso_orientation_rpy_cmd)

        self.set_wbc_goal(navigate_cmd, base_height_cmd, torso_orientation_rpy_cmd)
        self.wbc_policy.set_goal(self._wbc_goal)

        '''
        **************************************************
        Upper body PINK controller
        **************************************************
        '''
        # Extract upper body left/right arm pos/quat from actions
        left_arm_pos = actions_clone[:, :3].squeeze(0)
        left_arm_quat = actions_clone[:, 3:7].squeeze(0)
        right_arm_pos = actions_clone[:, 7:10].squeeze(0)
        right_arm_quat = actions_clone[:, 10:14].squeeze(0)

        # Convert from pos/quat to 4x4 transform matrix
        # Scipy requires quat xyzw, IsaacLab uses wxyz so a conversion is needed
        left_arm_quat = np.roll(left_arm_quat, -1)
        right_arm_quat = np.roll(right_arm_quat, -1)
        left_rotmat = R.from_quat(left_arm_quat).as_matrix()
        right_rotmat = R.from_quat(right_arm_quat).as_matrix()

        left_arm_pose = np.eye(4)
        left_arm_pose[:3, :3] = left_rotmat
        left_arm_pose[:3, 3] = left_arm_pos

        right_arm_pose = np.eye(4)
        right_arm_pose[:3, :3] = right_rotmat
        right_arm_pose[:3, 3] = right_arm_pos

        # Run PINK IK to get target upper body joint positions
        body_data = {"left_wrist_yaw_link": left_arm_pose, "right_wrist_yaw_link": right_arm_pose}
        target_robot_joints_mujoco = self.compute_upperbody_joint_positions(body_data)

        # arms only, no hands
        target_upper_body_joints = target_robot_joints_mujoco[self.robot_model.get_joint_group_indices("upper_body")]
        target_upper_body_joints = target_robot_joints_mujoco[self.robot_model.get_joint_group_indices("upper_body_no_hands")]
        target_upper_body_joints = torch.tensor(target_upper_body_joints, device=self.device, dtype=torch.float32).unsqueeze(0)

        '''
        **************************************************
        Prepare WBC policy input
        **************************************************
        '''

        '''
        left_arm_action_joint_ids = [11, 15, 19, 21, 23, 25, 27]
        right_arm_action_joint_ids = [12, 16, 20, 22, 24, 26, 28]
        left_hand_action_joint_ids = [29, 30, 35, 36, 37, 41]
        right_hand_action_joint_ids = [32, 33, 38, 39, 40, 42]
        '''

        # Get upper body action terms
        left_hand_action_term = self._env.action_manager.get_term("left_hand_action")
        right_hand_action_term = self._env.action_manager.get_term("right_hand_action")

        # Assemble joint positions from upper body action terms and Pink IK into observation for WBC policy
        sim_target_full_body_joints = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        sim_target_full_body_joints[:, left_hand_action_term._joint_ids] = left_hand_action_term.processed_actions
        sim_target_full_body_joints[:, right_hand_action_term._joint_ids] = right_hand_action_term.processed_actions

        wbc_obs = prepare_observations(self.num_envs, self._asset.data, self.wbc_g1_joints_order)
        wbc_target_full_body_joints = convert_sim_joint_to_wbc_joint(sim_target_full_body_joints, self._asset.data.joint_names, self.wbc_g1_joints_order)
        wbc_target_full_body_joints[:, self.robot_model.get_joint_group_indices("upper_body_no_hands")] = target_upper_body_joints
        wbc_target_upper_body_joints = wbc_target_full_body_joints[:, self.robot_model.get_joint_group_indices("upper_body")]

        self.wbc_policy.set_observation(wbc_obs)

        # TODO: add batched dimension
        wbc_action = self.wbc_policy.get_action(wbc_target_upper_body_joints)
        self._processed_actions = postprocess_actions(wbc_action, self._asset.data, self.wbc_g1_joints_order, self.device)

    def apply_actions(self):
        """Apply the computed joint positions based on the WBC solution."""
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)

        # Reset upper body controller warmup state
        self.upperbody_controller.in_warmup = True

        # Reset body IK solver internal state (PINK IK configuration)
        self.upperbody_controller.body_ik_solver.reset()

        # Reset WBC policy internal state
        # Lower body policy (G1HomiePolicy) has gait_indices, obs_history, etc.
        if env_ids is not None:
            if not isinstance(env_ids, torch.Tensor):
                env_ids_tensor = torch.tensor(env_ids, dtype=torch.int64, device=self.device)
            else:
                env_ids_tensor = env_ids.to(self.device)
            # Reset lower body policy if it has a reset method
            if hasattr(self.wbc_policy.lower_body_policy, 'reset'):
                self.wbc_policy.lower_body_policy.reset(env_ids_tensor)

        # Reset upper body policy (InterpolationPolicy) to clear trajectory state
        if hasattr(self.wbc_policy.upper_body_policy, 'reset'):
            self.wbc_policy.upper_body_policy.reset()


@configclass
class G1DecoupledWBCActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = G1DecoupledWBCAction
    """Specifies the action term class type for G1 WBC action."""

    preserve_order: bool = False
    joint_names: list[str] = MISSING
