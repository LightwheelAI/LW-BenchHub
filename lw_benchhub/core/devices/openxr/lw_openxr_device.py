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

"""OpenXR-powered device for teleoperation and interaction."""

import contextlib
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.openxr.xr_cfg import XrCfg
from isaaclab.devices.retargeter_base import RetargeterBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

with contextlib.suppress(ModuleNotFoundError):
    from isaacsim.xr.openxr import OpenXRSpec

_HAND_JOINTS_INDEX = [1, 5, 10, 15, 20, 25]
# The transformation matrices to convert hand pose to canonical view.
_OPERATOR2MANO_RIGHT = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])

_OPERATOR2MANO_LEFT = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
])

_HAND_LINK_NAMES = [
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
]


class LwOpenXRDevice(OpenXRDevice):

    def __init__(
        self,
        xr_cfg: XrCfg | None,
        retargeters: list[RetargeterBase] | None = None,
        env: None = None,
    ):
        """Initialize the OpenXR device.

        Args:
            xr_cfg: Configuration object for OpenXR settings. If None, default settings are used.
            retargeters: List of retargeters to transform tracking data into robot commands.
                        If None or empty list, raw tracking data will be returned.
        """
        self._xr_cfg = xr_cfg or XrCfg()
        self.env = env
        self.robot = env.scene.articulations['robot']
        self._xr_cfg.anchor_pos = self.robot.data.root_com_pos_w[0]
        self._xr_cfg.anchor_pos[2] = -0.2
        self._xr_cfg.anchor_rot = self.robot.data.root_com_quat_w[0]
        rotation = PoseUtils.matrix_from_quat(torch.tensor([0.707, 0.0, 0.0, -0.707], dtype=torch.float32))
        anchor_rot = torch.matmul(PoseUtils.matrix_from_quat(self._xr_cfg.anchor_rot), rotation)
        self._xr_cfg.anchor_rot = PoseUtils.quat_from_matrix(anchor_rot)
        # Initialize visualization if enabled
        self._enable_visualization = True
        self._num_open_xr_hand_joints = 2 * (int(OpenXRSpec.HandJointEXT.XR_HAND_JOINT_LITTLE_TIP_EXT) + 1)
        self._device = env.unwrapped.device
        if self._enable_visualization:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/markers",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=0.005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                },
            )
            self._markers = VisualizationMarkers(marker_cfg)
        super().__init__(self._xr_cfg)

    def _retarget_abs(self, wrist: np.ndarray, is_left=False) -> np.ndarray:
        """Handle absolute pose retargeting.

        Args:
            wrist: Wrist pose data from OpenXR

        Returns:
            Retargeted wrist pose in USD control frame
        """
        # Convert wrist data in openxr frame to usd control frame

        # Create pose object for openxr_right_wrist_in_world
        # Note: The pose utils require torch tensors
        wrist_pos = torch.tensor(wrist[:3], dtype=torch.float32)
        wrist_quat = torch.tensor(wrist[3:], dtype=torch.float32)
        openxr_right_wrist_in_world = PoseUtils.make_pose(wrist_pos, PoseUtils.matrix_from_quat(wrist_quat))

        # The usd control frame is 180 degrees rotated around z axis wrt to the openxr frame
        # This was determined through trial and error
        zero_pos = torch.zeros(3, dtype=torch.float32)

        # 180 degree rotation around z axis
        if is_left:
            z_axis_rot_quat = torch.tensor([0.707107, 0.0, 0.707107, 0.0], dtype=torch.float32)
        else:
            z_axis_rot_quat = torch.tensor([0.0, -0.707107, 0.0, 0.707107], dtype=torch.float32)

        usd_right_roll_link_in_openxr_right_wrist = PoseUtils.make_pose(
            zero_pos, PoseUtils.matrix_from_quat(z_axis_rot_quat)
        )

        # Convert wrist pose in openxr frame to usd control frame
        usd_right_roll_link_in_world = PoseUtils.pose_in_A_to_pose_in_B(
            usd_right_roll_link_in_openxr_right_wrist, openxr_right_wrist_in_world
        )

        # extract position and rotation
        usd_right_roll_link_in_world_pos, usd_right_roll_link_in_world_mat = PoseUtils.unmake_pose(
            usd_right_roll_link_in_world
        )
        usd_right_roll_link_in_world_quat = PoseUtils.quat_from_matrix(usd_right_roll_link_in_world_mat)

        arm_pos, arm_quat = math_utils.subtract_frame_transforms(self.robot.data.root_com_pos_w[0],
                                                                 self.robot.data.root_com_quat_w[0],
                                                                 usd_right_roll_link_in_world_pos,
                                                                 usd_right_roll_link_in_world_quat)
        return torch.cat([arm_pos, arm_quat])

    def _convert_hand_joints(self, hand_poses: dict[str, np.ndarray], operator2mano: np.ndarray) -> np.ndarray:
        """Prepares the hand joints data for retargeting.

        Args:
            hand_poses: Dictionary containing hand pose data with joint positions and rotations
            operator2mano: Transformation matrix to convert from operator to MANO frame

        Returns:
            Joint positions with shape (13, 3)
        """
        joint_position = np.zeros((6, 3))
        hand_joints = list(hand_poses.values())
        for i in range(len(_HAND_JOINTS_INDEX)):
            joint = hand_joints[_HAND_JOINTS_INDEX[i]]
            joint_position[i] = joint[:3]

        # Convert hand pose to the canonical frame.
        joint_position = joint_position - joint_position[0:1, :]
        xr_wrist_quat = hand_poses.get("wrist")[3:]
        # OpenXR hand uses w,x,y,z order for quaternions but scipy uses x,y,z,w order
        wrist_rot = R.from_quat([xr_wrist_quat[1], xr_wrist_quat[2], xr_wrist_quat[3], xr_wrist_quat[0]]).as_matrix()

        return torch.from_numpy(joint_position @ wrist_rot @ operator2mano)

    def advance(self) -> Any:
        raw_data = self._get_raw_data()
        left_hand_poses = raw_data[self.TrackingTarget.HAND_LEFT]
        right_hand_poses = raw_data[self.TrackingTarget.HAND_RIGHT]
        if self._enable_visualization:
            joints_position = np.zeros((self._num_open_xr_hand_joints, 3))

            joints_position[::2] = np.array([pose[:3] for pose in left_hand_poses.values()])
            joints_position[1::2] = np.array([pose[:3] for pose in right_hand_poses.values()])

            self._markers.visualize(translations=torch.tensor(joints_position, device=self._device))

        left_wrist = left_hand_poses['wrist']
        right_wrist = right_hand_poses['wrist']
        left_arm_abs = self._retarget_abs(left_wrist, is_left=True)
        right_arm_abs = self._retarget_abs(right_wrist, is_left=False)
        left_hand_abs = self._convert_hand_joints(left_hand_poses, _OPERATOR2MANO_LEFT)[1:]
        right_hand_abs = self._convert_hand_joints(right_hand_poses, _OPERATOR2MANO_RIGHT)[1:]
        abs_actions = {
            "base": torch.zeros(3, dtype=torch.float32),
            "left_arm_abs": left_arm_abs,
            "right_arm_abs": right_arm_abs,
            "left_finger_tips": left_hand_abs,
            "right_finger_tips": right_hand_abs,
        }

        return self.env.cfg.isaaclab_arena_env.embodiment.preprocess_device_action(abs_actions, self, base_move=False)
