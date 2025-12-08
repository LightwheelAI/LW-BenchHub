# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import MISSING
from typing import Dict, List, Optional, Set, Union

import numpy as np
import pinocchio as pin
import pink
import torch
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation as R

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from lw_benchhub.utils.math_utils.transform_utils import convert_quat, make_pose

from .wbc_policy.utils.robot_supplemental_info import RobotSupplementalInfo

USE_P_CONTROL = False


class DefaultIKSolverSettings:
    """
    Body IK solver settings.
    """

    def __init__(self):
        self.dt = 0.01
        self.num_step_per_frame = 1
        self.amplify_factor = 1.0
        self.posture_cost = 0.001  # Lowered from 0.01 to reduce posture stiffness
        self.posture_lm_damping = 1.0
        self.link_costs = {"hand": {"orientation_cost": 0.5, "position_cost": 10.0}}  # Increased from 1.0 to prioritize reaching
        self.posture_weight = {
            "waist_pitch": 10.0,
            "shoulder_pitch": 4.0,
            "shoulder_roll": 3.0,
            "shoulder_yaw": 2.0,
        }


class RobotModel:
    def __init__(
        self,
        robot_model_config,
        set_floating_base=False,
        supplemental_info: Optional[RobotSupplementalInfo] = None,
    ):
        # Determine loading method based on whether asset_path is provided
        if "asset_path" in robot_model_config and robot_model_config["asset_path"]:
            # With asset_path, load complete geometric model
            self.pinocchio_wrapper = pin.RobotWrapper.BuildFromURDF(
                filename=robot_model_config["urdf_path"],
                package_dirs=[robot_model_config["asset_path"]],
                root_joint=pin.JointModelFreeFlyer() if set_floating_base else None,
            )
        else:
            # Without asset_path, only load kinematic model, skip geometric model to avoid mesh file dependencies
            if set_floating_base:
                model = pin.buildModelFromUrdf(
                    robot_model_config["urdf_path"],
                    pin.JointModelFreeFlyer()
                )
            else:
                model = pin.buildModelFromUrdf(robot_model_config["urdf_path"])
            # Do not load any geometric model to avoid mesh file dependencies
            self.pinocchio_wrapper = pin.RobotWrapper(model, None, None)
        self.is_floating_base_model = set_floating_base

        # Dynamically generate joint to DOF index mapping to avoid hardcoded path dependencies
        self.joint_to_dof_index = {}

        # Get joint information from Pinocchio model
        model = self.pinocchio_wrapper.model
        joint_names = model.names

        # Skip universe joint (index 0)
        start_idx = 1 if not self.is_floating_base_model else 2

        # Create index mapping for each joint
        for i, joint_name in enumerate(joint_names[start_idx:], start=0):
            try:
                j_id = model.getJointId(joint_name)
                jmodel = model.joints[j_id]
                q_idx = jmodel.idx_q

                # Only process joints with configuration space entries (skip fixed joints)
                if q_idx != -1:
                    self.joint_to_dof_index[joint_name] = q_idx
            except Exception as e:
                print(f"Warning: Could not process joint '{joint_name}': {e}")
                continue

        # Store joint limits only for actual joints (excluding floating base)
        # if set floating base is true and the robot can move in the world
        # then we don't want to impose joint limits for the 7 dofs corresponding
        # to the floating base dofs.
        root_nq = 7 if set_floating_base else 0

        # Set up supplemental info if provided
        self.supplemental_info = supplemental_info
        self.num_dofs_body = len(self.supplemental_info.body_actuated_joints)
        self.num_dofs_hands = (
            len(self.supplemental_info.left_hand_actuated_joints) +
            len(self.supplemental_info.right_hand_actuated_joints)
        )
        self.lower_joint_limits = np.zeros(self.num_dofs)
        self.upper_joint_limits = np.zeros(self.num_dofs)
        if self.supplemental_info is not None:
            print(f"self.supplemental_info.body_actuated_joints: {self.supplemental_info.body_actuated_joints}")
            # Cache indices for body and hand actuated joints separately
            self._body_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.body_actuated_joints
            ]
            self._left_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.left_hand_actuated_joints
            ]
            self._right_hand_actuated_joint_indices = [
                self.dof_index(name) for name in self.supplemental_info.right_hand_actuated_joints
            ]
            self._hand_actuated_joint_indices = (
                self._left_hand_actuated_joint_indices + self._right_hand_actuated_joint_indices
            )

            # Cache indices for joint groups, handling nested groups
            self._joint_group_indices = {}
            for group_name, group_info in self.supplemental_info.joint_groups.items():
                indices = []
                # Add indices for direct joints
                indices.extend([self.dof_index(name) for name in group_info["joints"]])
                # Add indices from subgroups
                for subgroup_name in group_info["groups"]:
                    indices.extend(self.get_joint_group_indices(subgroup_name))
                self._joint_group_indices[group_name] = sorted(set(indices))

            # Update joint limits from supplemental info if available
            if (
                hasattr(self.supplemental_info, "joint_limits")
                and self.supplemental_info.joint_limits
            ):
                for joint_name, limits in self.supplemental_info.joint_limits.items():
                    if joint_name in self.joint_to_dof_index:
                        idx = self.joint_to_dof_index[joint_name] - root_nq
                        self.lower_joint_limits[idx] = limits[0]
                        self.upper_joint_limits[idx] = limits[1]

        self.initial_body_pose = None
        self.init_body_pose()

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom of the robot (floating base pose + joints)."""
        # Use actual Pinocchio model DOF count to ensure consistency with model
        return self.pinocchio_wrapper.model.nq
        # return self.num_dofs_body + self.num_dofs_hands

    @property
    def q_default(self) -> np.ndarray:
        """Get the zero pose of the robot."""
        # return self.pinocchio_wrapper.q0
        return self.initial_body_pose

    @property
    def joint_names(self) -> List[str]:
        """Get the names of the joints of the robot."""
        return list(self.joint_to_dof_index.keys())

    @property
    def num_joints(self) -> int:
        """Get the number of joints of the robot."""
        return len(self.joint_to_dof_index)

    def dof_index(self, joint_name: str) -> int:
        """
        Get the index in the degrees of freedom vector corresponding
        to the single-DoF joint with name `joint_name`.
        """
        if joint_name not in self.joint_to_dof_index:
            raise ValueError(
                f"Unknown joint name: '{joint_name}'. "
                f"Available joints: {list(self.joint_to_dof_index.keys())}"
            )
        return self.joint_to_dof_index[joint_name]

    def get_body_actuated_joint_indices(self) -> List[int]:
        """
        Get the indices of body actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")
        return self._body_actuated_joint_indices

    def get_hand_actuated_joint_indices(self, side: str = "both") -> List[int]:
        """
        Get the indices of hand actuated joints in the full configuration.
        Ordering is that of the actuated joints as defined in the supplemental info.
        Requires supplemental_info to be provided.

        Args:
            side: String specifying which hand to get indices for ('left', 'right', or 'both')
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        if side.lower() == "both":
            return self._hand_actuated_joint_indices
        elif side.lower() == "left":
            return self._left_hand_actuated_joint_indices
        elif side.lower() == "right":
            return self._right_hand_actuated_joint_indices
        else:
            raise ValueError("side must be 'left', 'right', or 'both'")

    def get_joint_group_indices(self, group_names: Union[str, Set[str]]) -> List[int]:
        """
        Get the indices of joints in one or more groups in the full configuration.
        Requires supplemental_info to be provided.
        The returned indices are sorted in ascending order, so that the joint ordering
        of the full model is preserved.

        Args:
            group_names: Either a single group name (str) or a set of group names (Set[str])

        Returns:
            List of joint indices in sorted order with no duplicates
        """
        if self.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Convert single string to set for uniform handling
        if isinstance(group_names, str):
            group_names = {group_names}

        # Collect indices from all groups
        all_indices = set()
        for group_name in group_names:
            if group_name not in self._joint_group_indices:
                raise ValueError(f"Unknown joint group: {group_name}")
            all_indices.update(self._joint_group_indices[group_name])

        return sorted(all_indices)

    def get_body_actuated_joints(self, q: np.ndarray) -> np.ndarray:
        """
        Get the configuration of body actuated joints from a full configuration.

        :param q: Configuration in full space
        :return: Configuration of body actuated joints
        """
        indices = self.get_body_actuated_joint_indices()

        return q[indices]

    def get_hand_actuated_joints(self, q: np.ndarray, side: str = "both") -> np.ndarray:
        """
        Get the configuration of hand actuated joints from a full configuration.

        Args:
            q: Configuration in full space
            side: String specifying which hand to get joints for ('left', 'right', or 'both')
        """
        indices = self.get_hand_actuated_joint_indices(side)
        return q[indices]

    def init_body_pose(self) -> np.ndarray:
        """
        Get the default upper body pose of the robot.
        """
        if self.initial_body_pose is not None:
            return self.initial_body_pose[self.get_joint_group_indices("upper_body")]

        q = np.zeros(self.num_dofs)
        default_joint_q = self.supplemental_info.default_joint_q
        for joint, sides in default_joint_q.items():
            for side, value in sides.items():
                joint_name = self.supplemental_info.joint_name_mapping[joint][side]
                if joint_name in self.joint_to_dof_index:
                    idx = self.dof_index(joint_name)
                    if 0 <= idx < len(q):
                        q[idx] = value
                    else:
                        print(f"Warning: Joint {joint_name} index {idx} out of bounds for q with size {len(q)}")
        self.initial_body_pose = q

    # def set_initial_body_pose(self, q: np.ndarray, q_idx=None) -> None:
    #     """
    #     Set the initial body pose of the robot.
    #     """
    #     if q_idx is None:
    #         self.initial_body_pose = q
    #     else:
    #         self.initial_body_pose[q_idx] = q

    def cache_forward_kinematics(self, q: np.ndarray, auto_clip=True) -> None:
        """
        Perform forward kinematics to update the pose of every joint and frame
        in the Pinocchio data structures for the given configuration `q`.

        :param q: A numpy array of shape (num_dofs,) representing the robot configuration.
        """
        if q.shape[0] != self.num_dofs:
            raise ValueError(f"Expected q of length {self.num_dofs}, got {q.shape[0]} instead.")

        # Apply auto-clip if enabled
        if auto_clip:
            q = self.clip_configuration(q)

        pin.framesForwardKinematics(self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q)

    def clip_configuration(self, q: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        Clip the configuration to stay within joint limits with a small tolerance.

        :param q: Configuration to clip
        :param margin: Tolerance to keep away from joint limits
        :return: Clipped configuration
        """
        q_clipped = q.copy()

        # Only clip joint positions, not floating base
        root_nq = 7 if self.is_floating_base_model else 0
        q_clipped[root_nq:] = np.clip(
            q[root_nq:], self.lower_joint_limits + margin, self.upper_joint_limits - margin
        )

        return q_clipped


class ReducedRobotModel:
    """
    A class that creates a reduced order robot model by fixing certain joints.
    This class maintains a mapping between the reduced state space and the full state space.
    """

    def __init__(
        self,
        full_robot_model: RobotModel,
        fixed_joints: List[str],
        fixed_values: Optional[List[float]] = None,
    ):
        """
        Create a reduced order robot model by fixing specified joints.

        :param full_robot_model: The original robot model
        :param fixed_joints: List of joint names to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        """
        self.full_robot = full_robot_model
        self.supplemental_info = full_robot_model.supplemental_info

        # # If fixed_values is None, use q0 from the full robot model
        # if fixed_values is None:
        #     fixed_values = []
        #     for joint_name in fixed_joints:
        #         full_idx = full_robot_model.dof_index(joint_name)
        #         fixed_values.append(full_robot_model.pinocchio_wrapper.q0[full_idx])
        # elif len(fixed_joints) != len(fixed_values):
        #     raise ValueError("fixed_joints and fixed_values must have the same length")

        # fixed_joints_indices = [i for i, name in enumerate(
        #     full_robot_model.pinocchio_wrapper.model.names.tolist()) if name in fixed_joints]
        # fixed_values = full_robot_model.pinocchio_wrapper.q0[fixed_joints_indices]

        # Correct method: use actual joint index in configuration space
        fixed_values = []
        for joint_name in fixed_joints:
            try:
                # Get joint ID
                j_id = full_robot_model.pinocchio_wrapper.model.getJointId(joint_name)
                # Get joint model
                jmodel = full_robot_model.pinocchio_wrapper.model.joints[j_id]
                # Get joint index in configuration space
                q_idx = jmodel.idx_q

                if q_idx == -1:
                    # Joint has no configuration space entry (usually fixed joint)
                    print(f"Info: Joint '{joint_name}' has no configuration space entry "
                          f"(q_idx=-1), using default value 0.0")
                    fixed_values.append(0.0)
                else:
                    # Get default value for joint
                    fixed_values.append(full_robot_model.pinocchio_wrapper.q0[q_idx])
            except Exception as e:
                print(f"Warning: Could not get configuration for joint '{joint_name}': {e}")
                # If unable to get configuration, use default value 0
                fixed_values.append(0.0)

        # Store fixed joints and their values
        self.fixed_joints = fixed_joints
        self.fixed_values = fixed_values

        # Create mapping between reduced and full state spaces
        self.reduced_to_full = []
        self.full_to_reduced = {}

        # Initialize with floating base indices if present
        if full_robot_model.is_floating_base_model:
            self.reduced_to_full.extend(range(7))  # Floating base indices
            for i in range(7):
                self.full_to_reduced[i] = i

        # Add active joint indices
        # Get joint information directly from Pinocchio model to avoid using full_robot_model.joint_names and dof_index
        for joint_name in full_robot_model.pinocchio_wrapper.model.names:
            if joint_name not in fixed_joints:
                try:
                    # Get joint ID
                    j_id = full_robot_model.pinocchio_wrapper.model.getJointId(joint_name)
                    # Get joint model
                    jmodel = full_robot_model.pinocchio_wrapper.model.joints[j_id]
                    # Get joint index in configuration space
                    full_idx = jmodel.idx_q

                    # Only process joints with configuration space entries (skip fixed joints)
                    if full_idx != -1:
                        reduced_idx = len(self.reduced_to_full)
                        self.reduced_to_full.append(full_idx)
                        self.full_to_reduced[full_idx] = reduced_idx
                except Exception as e:
                    print(f"Warning: Could not process joint '{joint_name}': {e}")
                    continue

        # Create a reduced Pinocchio model using buildReducedModel
        # First, get the list of joint IDs to lock
        locked_joint_ids = []
        for joint_name in fixed_joints:
            joint_id = full_robot_model.pinocchio_wrapper.model.getJointId(joint_name)
            if (full_robot_model.is_floating_base_model and joint_id > 1) or (
                not full_robot_model.is_floating_base_model and joint_id > 0
            ):
                locked_joint_ids.append(joint_id)

        # First build the reduced kinematic model
        reduced_model = pin.buildReducedModel(
            full_robot_model.pinocchio_wrapper.model,
            locked_joint_ids,
            full_robot_model.pinocchio_wrapper.q0,
        )

        # Then build the reduced geometry models using the reduced kinematic model
        self.pinocchio_wrapper = pin.RobotWrapper(
            model=reduced_model,
        )

        # Create joint to dof index mapping
        self.joint_to_dof_index = {}
        # Assume we only have single-dof joints
        # First two names correspond to universe and floating base joints
        names = (
            self.pinocchio_wrapper.model.names[2:]
            if self.full_robot.is_floating_base_model
            else self.pinocchio_wrapper.model.names[1:]
        )
        for name in names:
            j_id = self.pinocchio_wrapper.model.getJointId(name)
            jmodel = self.pinocchio_wrapper.model.joints[j_id]
            self.joint_to_dof_index[name] = jmodel.idx_q

        # Initialize joint limits
        root_nq = 7 if self.full_robot.is_floating_base_model else 0
        self.lower_joint_limits = deepcopy(
            self.pinocchio_wrapper.model.lowerPositionLimit[root_nq:]
        )
        self.upper_joint_limits = deepcopy(
            self.pinocchio_wrapper.model.upperPositionLimit[root_nq:]
        )

        # Update joint limits from supplemental info if available
        if self.supplemental_info is not None:
            for joint_name, limits in self.supplemental_info.joint_limits.items():
                if joint_name in self.joint_to_dof_index:
                    idx = self.joint_to_dof_index[joint_name] - root_nq
                    self.lower_joint_limits[idx] = limits[0]
                    self.upper_joint_limits[idx] = limits[1]

        # Cache indices for body and hand actuated joints in reduced space
        if self.supplemental_info is not None:
            # Get full indices for body and hand actuated joints
            full_body_indices = full_robot_model.get_body_actuated_joint_indices()
            full_hand_indices = full_robot_model.get_hand_actuated_joint_indices("both")

            # Map to reduced indices
            self._body_actuated_joint_indices = []
            for idx in full_body_indices:
                if idx in self.full_to_reduced:
                    self._body_actuated_joint_indices.append(self.full_to_reduced[idx])

            self._hand_actuated_joint_indices = []
            for idx in full_hand_indices:
                if idx in self.full_to_reduced:
                    self._hand_actuated_joint_indices.append(self.full_to_reduced[idx])

            # Cache indices for joint groups in reduced space
            self._joint_group_indices = {}
            for group_name in self.supplemental_info.joint_groups:
                full_indices = full_robot_model.get_joint_group_indices(group_name)
                reduced_indices = []
                for idx in full_indices:
                    if idx in self.full_to_reduced:
                        reduced_indices.append(self.full_to_reduced[idx])
                self._joint_group_indices[group_name] = sorted(set(reduced_indices))

    @classmethod
    def from_fixed_groups(
        cls,
        full_robot_model: RobotModel,
        fixed_group_names: List[str],
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints in specified groups.

        :param full_robot_model: The original robot model
        :param fixed_group_names: List of joint group names to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        if full_robot_model.supplemental_info is None:
            raise ValueError("supplemental_info must be provided to use this method")

        # Get all joints in the groups, including those from subgroups
        fixed_joints = set()  # Use a set to avoid duplicates

        for group_name in fixed_group_names:
            if group_name not in full_robot_model.supplemental_info.joint_groups:
                raise ValueError(f"Unknown joint group: {group_name}")

            group_info = full_robot_model.supplemental_info.joint_groups[group_name]

            # Add direct joints
            fixed_joints.update(group_info["joints"])

            # Add joints from subgroups
            for subgroup_name in group_info["groups"]:
                subgroup_joints = full_robot_model.get_joint_group_indices(subgroup_name)
                # Get joint names directly from Pinocchio model to avoid using full_robot_model.joint_names
                subgroup_joint_names = []
                for idx in subgroup_joints:
                    if idx < len(full_robot_model.pinocchio_wrapper.model.names):
                        subgroup_joint_names.append(full_robot_model.pinocchio_wrapper.model.names[idx])
                fixed_joints.update(subgroup_joint_names)

        # Convert set back to list for compatibility with the original constructor
        return cls(full_robot_model, list(fixed_joints), fixed_values)

    @classmethod
    def from_fixed_group(
        cls,
        full_robot_model: RobotModel,
        fixed_group_name: str,
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints in a specified group.
        This is a convenience method that calls from_fixed_groups with a single group.

        :param full_robot_model: The original robot model
        :param fixed_group_name: Name of the joint group to fix
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        return cls.from_fixed_groups(full_robot_model, [fixed_group_name], fixed_values)

    @classmethod
    def from_active_group(
        cls,
        full_robot_model: RobotModel,
        active_group_name: str,
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints EXCEPT those in the specified group.
        This is a convenience method that calls from_active_groups with a single group.

        :param full_robot_model: The original robot model
        :param active_group_name: Name of the joint group to keep active (all other joints will be fixed)
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """
        return cls.from_active_groups(full_robot_model, [active_group_name], fixed_values)

    @classmethod
    def from_active_groups(
        cls,
        full_robot_model: RobotModel,
        active_joint_names: List[str],
        fixed_values: Optional[List[float]] = None,
    ) -> "ReducedRobotModel":
        """
        Create a reduced order robot model by fixing all joints EXCEPT those in the specified groups.
        This is useful when you want to keep multiple groups active and fix everything else.

        :param full_robot_model: The original robot model
        :param active_joint_names: List of joint names to keep active (all other joints will be fixed)
        :param fixed_values: Optional list of values to fix the joints to. If None, uses the initial
                            joint positions (q0) from the full robot model.
        :return: A ReducedRobotModel instance
        """

        # # Get all joints in the active groups, including those from subgroups
        # active_joints = set()

        # for joint_name in active_joint_names:
        #     if joint_name in full_robot_model.pinocchio_wrapper.model.names.tolist():
        #         active_joints.update(joint_name)
        active_joint_names = set(active_joint_names)
        # Get all joints from the model
        all_joints = set(full_robot_model.pinocchio_wrapper.model.names.tolist())

        # The fixed joints are all joints minus the active joints
        fixed_joints = list(all_joints - active_joint_names)

        return cls(full_robot_model, fixed_joints, fixed_values)

    @property
    def num_dofs(self) -> int:
        """Get the number of degrees of freedom in the reduced model."""
        return self.pinocchio_wrapper.model.nq

    @property
    def q_default(self) -> np.ndarray:
        """Get the default pose of the robot (URDF zero pose + custom joint adjustments)."""
        # Get base zero pose from URDF
        q = self.pinocchio_wrapper.q0.copy()

        # Apply custom joint angle adjustments
        if self.supplemental_info is not None and self.supplemental_info.default_joint_q:
            default_joint_q = self.supplemental_info.default_joint_q
            for joint, sides in default_joint_q.items():
                for side, value in sides.items():
                    joint_name = self.supplemental_info.joint_name_mapping[joint][side]
                    if joint_name in self.joint_to_dof_index:
                        q[self.joint_to_dof_index[joint_name]] = value

        return q

    @property
    def joint_names(self) -> List[str]:
        """Get the names of the active joints in the reduced model."""
        # Get joint names directly from Pinocchio model to avoid using self.full_robot.joint_names
        all_joint_names = self.full_robot.pinocchio_wrapper.model.names
        return [name for name in all_joint_names if name not in self.fixed_joints]

    def reduced_to_full_configuration(self, q_reduced: np.ndarray) -> np.ndarray:
        """
        Convert a reduced configuration to the full configuration space.

        :param q_reduced: Configuration in reduced space
        :return: Configuration in full space with fixed joints set to their fixed values
        """
        if q_reduced.shape[0] != self.num_dofs:
            raise ValueError(
                f"Expected q_reduced of length {self.num_dofs}, got {q_reduced.shape[0]} instead"
            )

        q_full = np.zeros(self.full_robot.num_dofs)

        # Set active joints - with bounds checking
        for reduced_idx, full_idx in enumerate(self.reduced_to_full):
            if full_idx < len(q_full) and reduced_idx < len(q_reduced):
                q_full[full_idx] = q_reduced[reduced_idx]
            else:
                print(f"Warning: Skipping mapping - reduced_idx={reduced_idx}, full_idx={full_idx}, "
                      f"q_reduced.shape={q_reduced.shape}, q_full.shape={q_full.shape}")

        # Set fixed joints
        for joint_name, value in zip(self.fixed_joints, self.fixed_values):
            # Get joint index directly from Pinocchio model to avoid using self.full_robot.dof_index
            try:
                j_id = self.full_robot.pinocchio_wrapper.model.getJointId(joint_name)
                jmodel = self.full_robot.pinocchio_wrapper.model.joints[j_id]
                full_idx = jmodel.idx_q
                if full_idx != -1:  # Only process joints with configuration space entries
                    q_full[full_idx] = value
            except Exception as e:
                print(f"Warning: Could not set fixed joint '{joint_name}': {e}")
                continue

        return q_full

    def full_to_reduced_configuration(self, q_full: np.ndarray) -> np.ndarray:
        """
        Convert a full configuration to the reduced configuration space.

        :param q_full: Configuration in full space
        :return: Configuration in reduced space
        """
        if q_full.shape[0] != self.full_robot.num_dofs:
            raise ValueError(
                f"Expected q_full of length {self.full_robot.num_dofs}, got {q_full.shape[0]} instead"
            )

        q_reduced = np.zeros(self.num_dofs)

        # Copy active joints
        for reduced_idx, full_idx in enumerate(self.reduced_to_full):
            q_reduced[reduced_idx] = q_full[full_idx]

        return q_reduced

    def cache_forward_kinematics(self, q_reduced: np.ndarray, auto_clip=True) -> None:
        """
        Perform forward kinematics using the reduced configuration.

        :param q_reduced: Configuration in reduced space
        """
        # First update the full robot's forward kinematics
        q_full = self.reduced_to_full_configuration(q_reduced)
        self.full_robot.cache_forward_kinematics(q_full, auto_clip)

        # Then update the reduced model's forward kinematics
        pin.framesForwardKinematics(
            self.pinocchio_wrapper.model, self.pinocchio_wrapper.data, q_reduced
        )

    def clip_configuration(self, q_reduced: np.ndarray, margin: float = 1e-6) -> np.ndarray:
        """
        Clip the reduced configuration to stay within joint limits with a small tolerance.

        :param q_reduced: Configuration to clip
        :param margin: Tolerance to keep away from joint limits
        :return: Clipped configuration
        """
        q_full = self.reduced_to_full_configuration(q_reduced)
        q_full_clipped = self.full_robot.clip_configuration(q_full, margin)
        return self.full_to_reduced_configuration(q_full_clipped)


class WeightedPostureTask(PostureTask):
    """
    Weighted posture task for body IK PINK solver.
    """

    def __init__(
        self, cost: float, weights: np.ndarray, lm_damping: float = 0.0, gain: float = 1.0
    ) -> None:
        """Create weighted posture task.

        Args:
            cost: value used to cast joint angle differences to a homogeneous
                cost, in :math:`[\mathrm{cost}] / [\mathrm{rad}]`.
            weights: vector of weights for each joint.
            lm_damping: Unitless scale of the Levenberg-Marquardt (only when
                the error is large) regularization term, which helps when
                targets are unfeasible. Increase this value if the task is too
                jerky under unfeasible targets, but beware that too large a
                damping can slow down the task.
            gain: Task gain :math:`\alpha \in [0, 1]` for additional low-pass
                filtering. Defaults to 1.0 (no filtering) for dead-beat
                control.
        """
        super().__init__(cost=cost, lm_damping=lm_damping, gain=gain)
        self.weights = weights

    def compute_error(self, configuration):
        error = super().compute_error(configuration)
        return self.weights * error

    def compute_jacobian(self, configuration):
        J = super().compute_jacobian(configuration)
        return self.weights[:, np.newaxis] * J

    def __repr__(self):
        """Human-readable representation of the weighted posture task."""
        return (
            "WeightedPostureTask("
            f"cost={self.cost}, "
            f"weights={self.weights}, "
            f"gain={self.gain}, "
            f"lm_damping={self.lm_damping})"
        )


class PinkIKController:
    """
    PINK upper body controller with integrated inverse kinematics solving functionality.
    """

    def __init__(
        self,
        robot_model: RobotModel,
        body_active_joints: Optional[List[str]] = None,
        control_hands: bool = False,
        body_ik_solver_settings: DefaultIKSolverSettings = None,
    ):
        # Initialize robot model
        if body_active_joints is not None:
            self.body = ReducedRobotModel.from_active_groups(robot_model, body_active_joints)
            self.full_robot = self.body.full_robot
            self.using_reduced_robot_model = True
        else:
            self.body = robot_model
            self.full_robot = self.body
            self.using_reduced_robot_model = False

        # Initialize IK solver parameters
        if body_ik_solver_settings is None:
            body_ik_solver_settings = DefaultIKSolverSettings()
        self.dt = body_ik_solver_settings.dt
        self.num_step_per_frame = body_ik_solver_settings.num_step_per_frame
        self.amplify_factor = body_ik_solver_settings.amplify_factor
        self.link_costs = body_ik_solver_settings.link_costs
        self.posture_weight = body_ik_solver_settings.posture_weight
        self.posture_cost = body_ik_solver_settings.posture_cost
        self.posture_lm_damping = body_ik_solver_settings.posture_lm_damping

        # Initialize PINK configuration and tasks
        self._initialize_ik_solver()

        self.in_warmup = True

    def _initialize_ik_solver(self):
        """Initialize PINK IK solver"""
        self.configuration = pink.Configuration(
            self.body.pinocchio_wrapper.model,
            self.body.pinocchio_wrapper.data,
            self.body.q_default,
        )
        self.configuration.model.lowerPositionLimit = self.body.lower_joint_limits
        self.configuration.model.upperPositionLimit = self.body.upper_joint_limits

        # Initialize tasks
        self.tasks = {}
        for link_name, weight in self.link_costs.items():
            assert link_name != "posture", "posture is a reserved task name"

            # Map robot-agnostic link names to specific robot link names
            if link_name == "hand":
                # Get hand frame names from supplemental info
                for side in ["left", "right"]:
                    frame_name = self.body.supplemental_info.hand_frame_names[side]
                    task = FrameTask(
                        frame_name,
                        **weight,
                    )
                    self.tasks[frame_name] = task
            else:
                # For other links, use name directly
                task = FrameTask(
                    link_name,
                    **weight,
                )
                self.tasks[link_name] = task

        # Add posture task
        if self.posture_weight is not None:
            weight = np.ones(self.body.num_dofs)

            # Map robot-agnostic joint types to specific robot joint names
            for joint_type, posture_weight in self.posture_weight.items():
                if joint_type not in self.body.supplemental_info.joint_name_mapping:
                    print(f"Warning: Unknown joint type {joint_type}")
                    continue

                # Get joint name mapping
                joint_mapping = self.body.supplemental_info.joint_name_mapping[joint_type]

                # Handle single joint names and left-right mappings
                if isinstance(joint_mapping, str):
                    # Single joint (e.g., waist joint)
                    if joint_mapping in self.body.joint_to_dof_index:
                        joint_idx = self.body.joint_to_dof_index[joint_mapping]
                        weight[joint_idx] = posture_weight
                else:
                    # Left-right mapping (e.g., arm joints)
                    for side in ["left", "right"]:
                        joint_name = joint_mapping[side]
                        if joint_name in self.body.joint_to_dof_index:
                            joint_idx = self.body.joint_to_dof_index[joint_name]
                            weight[joint_idx] = posture_weight

            self.tasks["posture"] = WeightedPostureTask(
                cost=self.posture_cost,
                weights=weight,
                lm_damping=self.posture_lm_damping,
            )
        else:
            self.tasks["posture"] = PostureTask(
                cost=self.posture_cost, lm_damping=self.posture_lm_damping
            )

        # Set targets for all tasks
        for task in self.tasks.values():
            task.set_target_from_configuration(self.configuration)

    def inverse_kinematics(self, body_target_pose: Dict):
        """
        Solve inverse kinematics for given target poses.
        Args:
            body_target_pose: Dictionary of link names and corresponding target poses.
        Returns:
            Joint position vector that achieves the target poses.
        """
        # Set task targets
        for link_name, pose in body_target_pose.items():
            if link_name not in self.tasks:
                continue
            pose = pin.SE3(pose[:3, :3], pose[:3, 3])
            self.tasks[link_name].set_target(pose)

        # Iteratively solve IK
        for _ in range(self.num_step_per_frame):
            velocity = pink.solve_ik(
                self.configuration,
                self.tasks.values(),
                dt=self.dt,
                solver="osqp",
            )
            self.configuration.q = self.body.clip_configuration(
                self.configuration.q + velocity * self.dt * self.amplify_factor
            )
            self.configuration.update()
            self.body.cache_forward_kinematics(self.configuration.q)

        return self.configuration.q.copy()

    def save_configuration_q(self):
        """Save current configuration"""
        return self.configuration.q.copy()

    def load_configuration_q(self, q):
        """Load configuration"""
        self.configuration.q = q
        self.configuration.update()
        self.body.cache_forward_kinematics(self.configuration.q)

    def reset(self):
        """Reset IK solver"""
        self._initialize_ik_solver()

    def update_weights(self, weights):
        """Update task weights"""
        for link_name, weight in weights.items():
            if "position_cost" in weight:
                self.tasks[link_name].set_position_cost(weight["position_cost"])
            if "orientation_cost" in weight:
                self.tasks[link_name].set_orientation_cost(weight["orientation_cost"])


class PinkAction(ActionTerm):

    cfg: "PinkActionCfg"

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: "PinkActionCfg", env: ManagerBasedEnv):
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

        self.robot_model = RobotModel(
            self.cfg.robot_model_config,
            supplemental_info=self.cfg.robot_model_supplemental_info,
        )

        self.upperbody_controller = PinkIKController(
            robot_model=self.robot_model,
            body_active_joints=self._joint_names,
            control_hands=False,
            body_ik_solver_settings=self.cfg.body_ik_solver_settings,
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
        return self.upperbody_controller.save_configuration_q()

    def load_check_point(self, q):
        self.upperbody_controller.load_configuration_q(q)

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

        # Extract arm poses directly from actions tensor
        left_arm_pos = actions[:, :3].squeeze(0)
        left_arm_quat = actions[:, 3:7].squeeze(0)
        right_arm_pos = actions[:, 7:10].squeeze(0)
        right_arm_quat = actions[:, 10:14].squeeze(0)

        # Convert quaternions from wxyz (IsaacLab) to xyzw (scipy) format
        left_arm_quat_xyzw = convert_quat(left_arm_quat, to="xyzw")
        right_arm_quat_xyzw = convert_quat(right_arm_quat, to="xyzw")

        # Create rotation matrices and poses (minimize CPU-GPU transfers)
        left_rotmat = R.from_quat(left_arm_quat_xyzw.cpu().numpy()).as_matrix()
        right_rotmat = R.from_quat(right_arm_quat_xyzw.cpu().numpy()).as_matrix()

        left_arm_pose = make_pose(left_arm_pos, torch.tensor(left_rotmat, device=self.device))
        right_arm_pose = make_pose(right_arm_pos, torch.tensor(right_rotmat, device=self.device))

        # Run PINK IK to get target upper body joint positions
        # Use hand link names configured in supplemental_info instead of hardcoding
        body_data = {
            self.robot_model.supplemental_info.hand_frame_names["left"]: left_arm_pose.cpu().numpy(),
            self.robot_model.supplemental_info.hand_frame_names["right"]: right_arm_pose.cpu().numpy()
        }
        target_robot_joints_mujoco = self.compute_upperbody_joint_positions(body_data)

        # Convert to tensor and store
        self._processed_actions = torch.tensor(
            target_robot_joints_mujoco,
            device=self.device,
            dtype=torch.float32
        ).unsqueeze(0)

    def apply_actions(self):
        """Apply the computed joint positions based on the PINK IK solution."""
        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.
        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
        self.upperbody_controller.reset()


@configclass
class PinkActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = PinkAction
    """Specifies the action term class type for PINK action."""

    preserve_order: bool = True
    joint_names: list[str] = MISSING
    robot_model_config: dict = MISSING
    robot_model_supplemental_info: RobotSupplementalInfo = MISSING
    body_ik_solver_settings: DefaultIKSolverSettings = MISSING
