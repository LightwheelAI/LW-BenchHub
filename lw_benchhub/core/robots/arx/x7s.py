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


from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.pose import Pose

import lw_benchhub.core.mdp as mdp
import lw_benchhub.core.mdp as mdp_isaac_lab
from lw_benchhub.core.mdp import ee_frame_pos, ee_frame_quat, gripper_pos
from lw_benchhub.core.robots.robot_arena_base import EmbodimentBasePolicyObservationCfg, LwEmbodimentBase
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.log_utils import get_default_logger
from lw_benchhub.utils.math_utils import transform_utils as T
from lw_benchhub.utils.pinocchio_ik.x7s_ik import X7SBimanualIK

from .assets_cfg import X7_CFG, OFFSET_CONFIG as X7_OFFSET_CONFIG, VIS_HELPER_CFG
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    base_action: mdp.RelativeJointPositionActionCfg = MISSING
    body_action: mdp.RelativeJointPositionActionCfg = MISSING

    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING

    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class X7SCameraCfg:
    left_hand_camera: TiledCameraCfg = None
    first_person_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None
    left_shoulder_camera: TiledCameraCfg = None
    eye_in_hand_camera: TiledCameraCfg = None
    right_shoulder_camera: TiledCameraCfg = None


@configclass
class X7SPolicyObservationsCfg(EmbodimentBasePolicyObservationCfg):
    actions = ObsTerm(func=mdp_isaac_lab.last_action)
    joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel)
    eef_pos = ObsTerm(func=ee_frame_pos)
    eef_quat = ObsTerm(func=ee_frame_quat)
    gripper_pos = ObsTerm(func=gripper_pos)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class X7SSceneCfg:
    robot: ArticulationCfg = X7_CFG
    # Listens to the required transforms
    # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
    # the other frames are the fingers
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_hand_link",
                name="tool_left_hand",
                offset=OffsetCfg(
                    pos=(0.16735, 0.0, 0.0),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_link",
                name="tool_right_hand",
                offset=OffsetCfg(
                    pos=(0.16735, 0.0, 0.0),
                ),
            ),
        ],
    )
    left_gripper_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/link12",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    left_gripper_contact2 = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/link13",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    right_gripper_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/link21",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    right_gripper_contact2 = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/link22",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )


class X7SEnvCfg(LwEmbodimentBase):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None, enable_pinocchio_ik: bool = True,
                 pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.name = "X7S"
        self.robot_scale = robot_scale
        self.robot_base_offset = {"pos": [0.0, 0.0, 0.07], "rot": [0.0, 0.0, 0.0]}
        self.camera_config = X7SCameraCfg()
        self.action_config = ActionsCfg()
        self.policy_observation_config = X7SPolicyObservationsCfg()
        self.scene_config = X7SSceneCfg()
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)

        self.robot_base_link: str = "link4"
        self.robot_to_fixture_dist = 0.40
        self.robot_vis_helper_cfg = VIS_HELPER_CFG

        self.enable_pinocchio_ik = enable_pinocchio_ik
        self.pinocchio_urdf_path = pinocchio_urdf_path

        self.viewport_cfg = {
            "offset": [0.1, 0.0, 0.05],
            "lookat": [1.0, 0.12, -0.85]
        }
        self.offset_config = X7_OFFSET_CONFIG

        self.lbase_lock_state = False
        self.lbase_lock_value = 1.0
        self.lbase_joint_initial_value = 0.0
        self.lbase_joint_index = 3

        self.left_gripper_closed = True
        self.right_gripper_closed = True
        self.left_gripper_prev = 1.0
        self.right_gripper_prev = 1.0

        self.observation_cameras: dict = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_hand_link/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0, 0.12), rot=(0.57923, 0.40558, -0.40558, -0.57923), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=36,  # FOV: 91.2°
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "first_person_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link4/first_person_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.025, 0.065),
                                                    rot=(0.64086, 0.29884, -0.29884, -0.64086),  # 0, -50, -90
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=36,
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": ["product"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION,
                                 ExecuteMode.REPLAY_JOINT_TARGETS]
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_link/right_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0, 0.12), rot=(0.57923, 0.40558, -0.40558, -0.57923), convention="opengl"),  # 0.0, -70, -90
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=36,  # FOV: 91.2°
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": ["product"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "left_shoulder_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link1/left_shoulder_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.15, 0.9, 1.2), rot=(0.05796, 0.16369, -0.32874, -0.92832), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=27.7,  # Adjusted for 60° FOV
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "eye_in_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link1/eye_in_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0, 1.5), rot=(0.70711, 0.0, 0.0, -0.70711), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=27.7,  # Adjusted for 60° FOV
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "right_shoulder_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link1/right_shoulder_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.15, -0.9, 1.2), rot=(0.93905, 0.33254, -0.08216, -0.02909), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=27.7,  # Adjusted for 60° FOV
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            }
        }

        self.pid_configs = {
            'body_z': {
                'kp': 10.0,
                'kd': 5.0,
                'ki': 0.0,
                'deadzone': 0.001,
                'prev_error': 0.0,
                'integral_error': 0.0
            },
            'body_y': {
                'kp': 10.0,
                'kd': 5.0,
                'ki': 0.0,
                'deadzone': 0.001,
                'prev_error': 0.0,
                'integral_error': 0.0
            },
            'base_x': {
                'kp': 100.0,
                'kd': 2.0,
                'ki': 0.2,
                'deadzone': 0.0001,
                'prev_error': 0.0,
                'integral_error': 0.0
            },
            'base_y': {
                'kp': 25.0,
                'kd': 2.0,
                'ki': 0.2,
                'deadzone': 0.0001,
                'prev_error': 0.0,
                'integral_error': 0.0
            },
            'base_yaw': {
                'kp': 25.0,
                'kd': 2.0,
                'ki': 0.2,
                'deadzone': 0.0001,
                'prev_error': 0.0,
                'integral_error': 0.0
            }
        }

        # Set Actions for the specific robot type
        self.action_config.base_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["base.*"],
            scale=0.01,  # 01,
            use_zero_offset=True,  # use default offset is not working for base action
        )
        self.action_config.body_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["body.*"],
            scale=0.025,  # 01,
            use_zero_offset=True,  # use default offset is not working for base action
        )
        # control all gripper joints except the joints that are Exclude From Articulation
        self.action_config.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_gripper.*"],
            open_command_expr={'left_gripper1': 0.044, 'left_gripper2': 0.044},
            close_command_expr={'left_gripper1': -0., 'left_gripper2': -0.},
        )
        self.action_config.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_gripper.*"],
            open_command_expr={'right_gripper1': -0.0, 'right_gripper2': -0.0},
            close_command_expr={'right_gripper1': -0.04, 'right_gripper2': -0.04},
        )

        self.viewport_cfg = {
            "offset": [0.1, 0.0, 0.05],
            "lookat": [1.0, 0.12, -0.85]
        }

        # Initialize IK solver if enabled
        if self.enable_pinocchio_ik:
            if self.pinocchio_urdf_path is not None:
                self._bi_ik = X7SBimanualIK(urdf_path=self.pinocchio_urdf_path)
            else:
                self._bi_ik = X7SBimanualIK()
        self.last_pos = None

    def pid_control(self, target_value, current_value, axis_name):
        """
        PID controller function

        Args:
            target_value: target value
            current_value: current value
            axis_name: axis name ('body_z' or 'body_y')

        Returns:
            float: control output
        """
        config = self.pid_configs[axis_name]
        error = target_value - current_value

        if abs(error) < config['deadzone']:
            return 0.0

        p_term = config['kp'] * error

        config['integral_error'] += error
        i_term = config['ki'] * config['integral_error']

        d_term = config['kd'] * (error - config['prev_error'])

        control_output = p_term + i_term + d_term

        if abs(config['integral_error']) > 1.0:
            config['integral_error'] = 1.0 if config['integral_error'] > 0 else -1.0

        config['prev_error'] = error

        return control_output

    def reset_pid_state(self, axis_name):
        """
        reset pid state

        Args:
            axis_name: axis name ('body_z' or 'body_y')
        """
        config = self.pid_configs[axis_name]
        config['prev_error'] = 0.0
        config['integral_error'] = 0.0

    def get_joint_index(self, device, joint_name_pattern):
        """
        find joint index by joint name pattern

        Args:
            device: device object
            joint_name_pattern: joint name pattern, like "body_z", "body_y"

        Returns:
            int: joint index, if not found, return -1
        """
        joint_names = device.robot.data.joint_names
        for i, name in enumerate(joint_names):
            if joint_name_pattern in name:
                return i
        return -1

    def init_lbase_joint_indices(self, device):
        """
        initialize lbase joint indices cache

        Args:
            device: device object
        """
        self.lbase_body_z_index = self.get_joint_index(device, "body_z")
        self.lbase_body_y_index = self.get_joint_index(device, "body_y")

        if self.lbase_body_z_index == -1 or self.lbase_body_y_index == -1:
            print(f"Warning: Could not find joint indices - "
                  f"body_z: {self.lbase_body_z_index}, body_y: {self.lbase_body_y_index}")
        else:
            print(f"Joint indices initialized - "
                  f"body_z: {self.lbase_body_z_index}, body_y: {self.lbase_body_y_index}")

    def init_rbase_joint_indices(self, device):
        """
        initialize rbase joint indices cache

        Args:
            device: device object
        """
        self.rbase_base_x_joint_index = self.get_joint_index(device, "base_x_joint")
        self.rbase_base_y_joint_index = self.get_joint_index(device, "base_y_joint")
        self.rbase_base_yaw_link_index = self.get_joint_index(device, "base_yaw_link")

        if (self.rbase_base_x_joint_index == -1 or
            self.rbase_base_y_joint_index == -1 or
                self.rbase_base_yaw_link_index == -1):
            print(f"Warning: Could not find rbase joint indices - "
                  f"base_x_joint: {self.rbase_base_x_joint_index}, "
                  f"base_y_joint: {self.rbase_base_y_joint_index}, "
                  f"base_yaw_link: {self.rbase_base_yaw_link_index}")
        else:
            print(f"Rbase joint indices initialized - "
                  f"base_x_joint: {self.rbase_base_x_joint_index}, "
                  f"base_y_joint: {self.rbase_base_y_joint_index}, "
                  f"base_yaw_link: {self.rbase_base_yaw_link_index}")

    def reset_ik(self):
        """Reset the internal state of the IK solvers, if present."""
        if hasattr(self, "_bi_ik") and self._bi_ik is not None:
            if hasattr(self._bi_ik, "left") and hasattr(self._bi_ik.left, "reset"):
                self._bi_ik.left.reset()
            if hasattr(self._bi_ik, "right") and hasattr(self._bi_ik.right, "reset"):
                self._bi_ik.right.reset()

    def reset_robot_cfg_state(self):
        # lbase joint lock state
        self.lbase_lock_state = False
        self.lbase_lock_value = 1.0
        self.lbase_lock_value_body_z = 0.0
        self.lbase_lock_value_body_y = 0.0
        self.lbase_joint_initial_value = 0.0
        self.lbase_joint_index = 3
        # lbase joint indices cache
        self.lbase_body_z_index = -1
        self.lbase_body_y_index = -1
        # lbase auto-lock thresholds
        self.lbase_auto_lock_threshold = 0.05  # if input magnitude is less than this value, auto-lock lbase joints
        self.lbase_auto_unlock_threshold = 0.3  # if input magnitude is greater than this value, auto-unlock lbase joints

        # rbase base lock state
        self.rbase_lock_state = False
        self.rbase_lock_value_base_x = 0.0
        self.rbase_lock_value_base_y = 0.0
        self.rbase_lock_value_base_yaw = 0.0
        # rbase joint indices cache
        self.rbase_base_x_joint_index = -1
        self.rbase_base_y_joint_index = -1
        self.rbase_base_yaw_link_index = -1
        # rbase auto-lock thresholds
        self.rbase_auto_lock_threshold = 0.1  # if input magnitude is less than this value, auto-lock rbase joints
        self.rbase_auto_unlock_threshold = 0.3  # if input magnitude is greater than this value, auto-unlock rbase joints
        for axis_name in ['body_z', 'body_y', 'base_x', 'base_y', 'base_yaw']:
            if axis_name in self.pid_configs:
                self.pid_configs[axis_name]['prev_error'] = 0.0
                self.pid_configs[axis_name]['integral_error'] = 0.0

        self.left_gripper_closed = True
        self.right_gripper_closed = True
        self.left_gripper_prev = 1.0
        self.right_gripper_prev = 1.0

        self._last_left_eef_pose = None
        self._last_right_eef_pose = None
        self.last_pos = None

        self.reset_ik()

    def _smooth_eef_pose(
        self,
        target_pose: torch.Tensor,
        last_pose: torch.Tensor | None,
        alpha: float = 0.4,
        max_pos_delta: float = 0.05,
        max_rot_delta: float = 0.2
    ) -> torch.Tensor:
        """smooth end-effector pose"""
        from isaaclab.utils.math import quat_mul, quat_inv, axis_angle_from_quat, quat_from_angle_axis

        squeeze_output = False
        if target_pose.dim() == 2:
            target_pose = target_pose.squeeze(0)
            squeeze_output = True

        if last_pose is None:
            result = target_pose.clone()
            return result.unsqueeze(0) if squeeze_output else result

        if last_pose.dim() == 2:
            last_pose = last_pose.squeeze(0)

        # position smoothing
        smoothed_pos = alpha * target_pose[:3] + (1 - alpha) * last_pose[:3]
        pos_delta = smoothed_pos - last_pose[:3]
        pos_delta_norm = torch.norm(pos_delta)
        if pos_delta_norm > max_pos_delta:
            pos_delta = pos_delta / pos_delta_norm * max_pos_delta
        smoothed_pos = last_pose[:3] + pos_delta

        # orientation smoothing: using axis-angle
        current_quat = last_pose[3:].unsqueeze(0)      # (1, 4) [w,x,y,z]
        target_quat = target_pose[3:].unsqueeze(0)     # (1, 4)

        # 1. calculate relative rotation (from current to target)
        relative_quat = quat_mul(quat_inv(current_quat), target_quat)  # (1, 4)

        # 2. convert to axis-angle representation [rx, ry, rz], where norm = rotation angle
        axis_angle = axis_angle_from_quat(relative_quat)  # (1, 3)

        # 3. directly smooth the axis-angle (this is the "increment" of rotation)
        smoothed_axis_angle = alpha * axis_angle  # (1, 3)

        # 4. limit the rotation angle change
        angle = torch.norm(smoothed_axis_angle)
        if angle > max_rot_delta:
            smoothed_axis_angle = smoothed_axis_angle / angle * max_rot_delta

        # 5. convert back to quaternion and apply to current pose
        angle = torch.norm(smoothed_axis_angle, dim=-1)  # (1,)
        if angle > 1e-6:
            axis = smoothed_axis_angle / angle.unsqueeze(-1)  # (1, 3)
            delta_quat = quat_from_angle_axis(angle, axis)  # (1, 4)
            smoothed_quat = quat_mul(current_quat, delta_quat)  # (1, 4)
        else:
            smoothed_quat = current_quat

        smoothed_quat = smoothed_quat.squeeze(0)  # (4,)

        result = torch.cat([smoothed_pos, smoothed_quat], dim=-1)
        return result.unsqueeze(0) if squeeze_output else result

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:

        num_envs = device.env.num_envs

        rbase_input_x = abs(action.get('rbase', [0, 0])[0])
        rbase_input_y = abs(action.get('rbase', [0, 0])[1])
        rbase_input_magnitude = max(rbase_input_x, rbase_input_y)

        x_joint_vel = device.robot.data.joint_vel[0, self.rbase_base_x_joint_index].item()
        y_joint_vel = device.robot.data.joint_vel[0, self.rbase_base_y_joint_index].item()
        yaw_joint_vel = device.robot.data.joint_vel[0, self.rbase_base_yaw_link_index].item()

        left_gripper_force = device.env.scene.sensors["left_gripper_contact"].data.net_forces_w
        right_gripper_force = device.env.scene.sensors["right_gripper_contact"].data.net_forces_w

        # Check if left_gripper_force has any values with absolute value > 0
        left_gripper_has_force = torch.any(torch.abs(left_gripper_force) > 20)
        right_gripper_has_force = torch.any(torch.abs(right_gripper_force) > 20)

        if (
            not self.rbase_lock_state
            and rbase_input_magnitude < self.rbase_auto_lock_threshold
            and abs(x_joint_vel) < 0.01
            and abs(y_joint_vel) < 0.01
            and abs(yaw_joint_vel) < 0.01
            and (left_gripper_has_force
                 or right_gripper_has_force)
        ):
            if (self.rbase_base_x_joint_index == -1 or
                self.rbase_base_y_joint_index == -1 or
                    self.rbase_base_yaw_link_index == -1):
                self.init_rbase_joint_indices(device)

            if (self.rbase_base_x_joint_index != -1 and
                self.rbase_base_y_joint_index != -1 and
                    self.rbase_base_yaw_link_index != -1):
                self.rbase_lock_state = True

                self.rbase_lock_value_base_x = device.robot.data.joint_pos[0, self.rbase_base_x_joint_index].item()
                self.rbase_lock_value_base_y = device.robot.data.joint_pos[0, self.rbase_base_y_joint_index].item()
                self.rbase_lock_value_base_yaw = device.robot.data.joint_pos[0, self.rbase_base_yaw_link_index].item()

                self.reset_pid_state('base_x')
                self.reset_pid_state('base_y')
                self.reset_pid_state('base_yaw')
                print(f"Auto-locked rbase joints base_x_joint({self.rbase_base_x_joint_index}), "
                      f"base_y_joint({self.rbase_base_y_joint_index}), "
                      f"base_yaw_link({self.rbase_base_yaw_link_index}) to position: "
                      f"x={self.rbase_lock_value_base_x:.3f}, "
                      f"y={self.rbase_lock_value_base_y:.3f}, "
                      f"yaw={self.rbase_lock_value_base_yaw:.3f}, "
                      f"left_gripper_force={left_gripper_force}, "
                      f"right_gripper_force={right_gripper_force}")
            else:
                print("Warning: Could not find rbase joint indices for base_x_joint, base_y_joint, or base_yaw_link")
        elif self.rbase_lock_state and ((rbase_input_magnitude > self.rbase_auto_unlock_threshold) or (not left_gripper_has_force and not right_gripper_has_force)):
            self.rbase_lock_state = False

            self.reset_pid_state('base_x')
            self.reset_pid_state('base_y')
            self.reset_pid_state('base_yaw')
            print("Auto-unlocked rbase due to large input or small force")
            get_default_logger().info("Auto-unlocked rbase due to large input or small force")

        base_action = torch.zeros(num_envs, 3, device=device.env.device)
        non_lock_scale = 0.5

        if self.rbase_lock_state:
            if (self.rbase_base_x_joint_index != -1 and
                self.rbase_base_y_joint_index != -1 and
                    self.rbase_base_yaw_link_index != -1):
                current_x = device.robot.data.joint_pos[0, self.rbase_base_x_joint_index].item()
                current_y = device.robot.data.joint_pos[0, self.rbase_base_y_joint_index].item()
                current_yaw = device.robot.data.joint_pos[0, self.rbase_base_yaw_link_index].item()

                control_output_x = self.pid_control(self.rbase_lock_value_base_x, current_x, 'base_x')
                control_output_y = self.pid_control(self.rbase_lock_value_base_y, current_y, 'base_y')
                control_output_yaw = self.pid_control(self.rbase_lock_value_base_yaw, current_yaw, 'base_yaw')

                base_action[:, 0] = control_output_x
                base_action[:, 1] = control_output_y
                base_action[:, 2] = control_output_yaw
            else:
                if action['rsqueeze'] > 0.5:
                    base_action[:, :] = torch.tensor([action['rbase'][0], 0, action['rbase'][1]],
                                                     device=action['rbase'].device)
                else:
                    base_action[:, :] = torch.tensor([action['rbase'][0], action['rbase'][1], 0],
                                                     device=action['rbase'].device)
        else:
            if action['rsqueeze'] > 0.5:
                base_action[:, :] = torch.tensor([action['rbase'][0] * non_lock_scale, 0, action['rbase'][1] * non_lock_scale],
                                                 device=action['rbase'].device)
            else:
                base_action[:, :] = torch.tensor([action['rbase'][0] * non_lock_scale, action['rbase'][1] * non_lock_scale, 0],
                                                 device=action['rbase'].device)

        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 4],
                                                                           device.robot.data.body_link_quat_w[0, 4])
        base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
        base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
        base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)
        cos_yaw = torch.cos(torch.tensor(base_yaw, device=device.env.device))
        sin_yaw = torch.sin(torch.tensor(base_yaw, device=device.env.device))
        rot_mat_2d = torch.tensor([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ], device=device.env.device)
        robot_x = base_action[:, 0]
        robot_y = base_action[:, 1]
        local_xy = torch.tensor([robot_x, robot_y], device=device.env.device)
        world_xy = torch.matmul(rot_mat_2d, local_xy)
        base_action[:, 0] = world_xy[0]
        base_action[:, 1] = world_xy[1]

        if getattr(self, "enable_pinocchio_ik", False):

            # IK branch (base_link frame)
            left_pose = action["left_arm_abs"].clone()   # already in base_link frame
            right_pose = action["right_arm_abs"].clone()  # already in base_link frame
            left_pose[3:] = left_pose[[6, 3, 4, 5]]
            right_pose[3:] = right_pose[[6, 3, 4, 5]]
            # ensure wxyz order if needed: left_pose[3:] = left_pose[[6,3,4,5]]
            left_pose = left_pose.repeat(num_envs, 1)
            right_pose = right_pose.repeat(num_envs, 1)

            left_pose = self._smooth_eef_pose(left_pose, self._last_left_eef_pose)
            right_pose = self._smooth_eef_pose(right_pose, self._last_right_eef_pose)
            self._last_left_eef_pose = left_pose.clone()  # update
            self._last_right_eef_pose = right_pose.clone()

            # find left_hand_link and right_hand_link index
            robot = device.env.scene.articulations['robot']
            left_hand_ids, _ = robot.find_bodies("left_hand_link")
            right_hand_ids, _ = robot.find_bodies("right_hand_link")
            base_link_ids, _ = robot.find_bodies("world")

            left_hand_idx = left_hand_ids[0]
            right_hand_idx = right_hand_ids[0]
            base_link_idx = base_link_ids[0]

            # Obtain the pose in the world coordinate system
            # body_link_pose_w shape: (num_envs, num_bodies, 7) [x, y, z, w, x, y, z]
            left_hand_pose_w = robot.data.body_link_pose_w[0, left_hand_idx, :]  # shape: (7,)
            right_hand_pose_w = robot.data.body_link_pose_w[0, right_hand_idx, :]  # shape: (7,)
            base_link_pose_w = robot.data.body_link_pose_w[0, base_link_idx, :]  # shape: (7,)

            # convert to robot coordinate system (base_link frame)
            from isaaclab.utils.math import subtract_frame_transforms

            # left hand relative to base_link
            left_pos_base, left_quat_base = subtract_frame_transforms(
                base_link_pose_w[:3].unsqueeze(0),   # base position in world
                base_link_pose_w[3:].unsqueeze(0),   # base quaternion [w,x,y,z] in world
                left_hand_pose_w[:3].unsqueeze(0),   # left hand position in world
                left_hand_pose_w[3:].unsqueeze(0)    # left hand quaternion in world
            )

            # right hand relative to base_link
            right_pos_base, right_quat_base = subtract_frame_transforms(
                base_link_pose_w[:3].unsqueeze(0),
                base_link_pose_w[3:].unsqueeze(0),
                right_hand_pose_w[:3].unsqueeze(0),
                right_hand_pose_w[3:].unsqueeze(0)
            )

            target_left_pos = left_pose[:, :3]      # (3,) from (num_envs, 7) take the first environment
            target_left_quat = left_pose[:, 3:]     # (4,) [w,x,y,z]

            target_right_pos = right_pose[:, :3]
            target_right_quat = right_pose[:, 3:]

            # calculate relative pose
            relative_left_pos, relative_left_quat = subtract_frame_transforms(
                left_pos_base,      # current left hand position
                left_quat_base,     # current left hand pose [w,x,y,z]
                target_left_pos,    # target left hand position
                target_left_quat    # target left hand pose [w,x,y,z]
            )

            relative_right_pos, relative_right_quat = subtract_frame_transforms(
                right_pos_base,
                right_quat_base,
                target_right_pos,
                target_right_quat
            )

            relative_left_pose = torch.cat([relative_left_pos, relative_left_quat], dim=-1)
            relative_right_pose = torch.cat([relative_right_pos, relative_right_quat], dim=-1)

            device.env.recorder_manager.add_to_episodes(f"eef/relative_left_pose", relative_left_pose)
            device.env.recorder_manager.add_to_episodes(f"eef/relative_right_pose", relative_right_pose)
            device.env.recorder_manager.add_to_episodes(f"eef/left_pose", left_pose)
            device.env.recorder_manager.add_to_episodes(f"eef/right_pose", right_pose)

            l_q_np, _ = self._bi_ik.left.solve_pose_to_joints(left_pose.detach().cpu().numpy())
            r_q_np, _ = self._bi_ik.right.solve_pose_to_joints(right_pose.detach().cpu().numpy())
            left_arm_action = torch.tensor(l_q_np, device=device.env.device, dtype=torch.float32)
            right_arm_action = torch.tensor(r_q_np, device=device.env.device, dtype=torch.float32)

        elif self.action_config.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
            left_arm_action = left_arm_action.repeat(num_envs, 1)
            right_arm_action = right_arm_action.repeat(num_envs, 1)
        else:  # Absolute mode

            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                pose_quat = abs_arm[3:7]
                combined_quat = T.quat_multiply(base_quat, pose_quat)
                arm_action = abs_arm.clone()
                rot_mat = T.quat2mat(base_quat)
                gripper_movement = torch.matmul(rot_mat, arm_action[:3])
                pose_movement = base_movement + gripper_movement
                arm_action[:3] = pose_movement
                arm_action[3] = combined_quat[3]
                arm_action[4:7] = combined_quat[:3]
                arm_action[2] = arm_action[2] + device.robot.data.joint_pos[0, 3]
                arm_action = arm_action.repeat(num_envs, 1)
                if arm_idx == 0:
                    left_arm_action = arm_action
                else:
                    right_arm_action = arm_action

        left_gripper_pressed = action["left_gripper"] > 0
        right_gripper_pressed = action["right_gripper"] > 0
        left_gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        left_gripper_action[:] = -1.0 if left_gripper_pressed else 1.0
        right_gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        right_gripper_action[:] = -1.0 if right_gripper_pressed else 1.0

        self.set_right_hand_lines_visibility(not right_gripper_pressed)

        self.set_left_hand_lines_visibility(not left_gripper_pressed)

        input_z = abs(action.get('lbase', [0, 0])[0])
        input_y = abs(action.get('lbase', [0, 0])[1])
        input_magnitude = max(input_z, input_y)

        if not self.lbase_lock_state and input_magnitude < self.lbase_auto_lock_threshold:
            if self.lbase_body_z_index == -1 or self.lbase_body_y_index == -1:
                self.init_lbase_joint_indices(device)

            if self.lbase_body_z_index != -1 and self.lbase_body_y_index != -1:
                self.lbase_lock_state = True
                self.lbase_lock_value_body_z = device.robot.data.joint_pos[0, self.lbase_body_z_index].item()
                self.lbase_lock_value_body_y = device.robot.data.joint_pos[0, self.lbase_body_y_index].item()

                self.reset_pid_state('body_z')
                self.reset_pid_state('body_y')
                print(f"Auto-locked joints body_z({self.lbase_body_z_index}) and "
                      f"body_y({self.lbase_body_y_index}) to position: "
                      f"z={self.lbase_lock_value_body_z:.3f}, y={self.lbase_lock_value_body_y:.3f}")
            else:
                print("Warning: Could not find joint indices for body_z or body_y")

        elif self.lbase_lock_state and input_magnitude > self.lbase_auto_unlock_threshold:
            self.lbase_lock_state = False

            self.reset_pid_state('body_z')
            self.reset_pid_state('body_y')
            print(f"Auto-unlocked joint {self.lbase_joint_index} due to large input, "
                  f"reset to initial position: {self.lbase_joint_initial_value}")

        if self.lbase_lock_state:
            if self.lbase_body_z_index != -1 and self.lbase_body_y_index != -1:
                current_pos_z = device.robot.data.joint_pos[0, self.lbase_body_z_index].item()
                current_pos_y = device.robot.data.joint_pos[0, self.lbase_body_y_index].item()

                control_output_body_z = self.pid_control(self.lbase_lock_value_body_z, current_pos_z, 'body_z')
                control_output_body_y = self.pid_control(self.lbase_lock_value_body_y, current_pos_y, 'body_y')

                body_action = torch.zeros(num_envs, 2, device=device.env.device)
                body_action[:, 0] = control_output_body_z
                body_action[:, 1] = control_output_body_y
            else:
                body_action = torch.zeros(num_envs, 2, device=device.env.device)
                body_action[:, 0] = action['lbase'][0] / 20
                body_action[:, 1] = action['lbase'][1] / 4

        else:
            body_action = torch.zeros(num_envs, 2, device=device.env.device)
            body_action[:, 0] = action['lbase'][0] / 20
            body_action[:, 1] = action['lbase'][1] / 4

        return torch.concat([base_action, body_action, left_arm_action, right_arm_action,
                             left_gripper_action, right_gripper_action], dim=1)

    def set_left_hand_lines_visibility(self, visible: bool):
        """ set USD prim

        Args:
            visible: line visibility
        """
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if not stage:
                print(" cannot get USD stage")
                return

            left_line_z_path = "/World/envs/env_0/Robot/left_hand_link/left_hand_line_z"
            left_line_x_path = "/World/envs/env_0/Robot/left_hand_link/left_hand_line_x"
            left_line_y_path = "/World/envs/env_0/Robot/left_hand_link/left_hand_line_y"
            left_prim_z = stage.GetPrimAtPath(left_line_z_path)
            left_prim_x = stage.GetPrimAtPath(left_line_x_path)
            left_prim_y = stage.GetPrimAtPath(left_line_y_path)
            if left_prim_z and left_prim_z.IsValid():
                imageable = UsdGeom.Imageable(left_prim_z)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

            if left_prim_x and left_prim_x.IsValid():
                imageable = UsdGeom.Imageable(left_prim_x)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

            if left_prim_y and left_prim_y.IsValid():
                imageable = UsdGeom.Imageable(left_prim_y)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

            status = "show" if visible else "hide"
            # print(f" hand lines {status}")

        except Exception as e:
            print(f" set hand lines visibility failed: {e}")

    def set_right_hand_lines_visibility(self, visible: bool):
        """ set USD prim

        Args:
            visible:
        """
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if not stage:
                print(" cannot get USD stage")
                return

            right_line_z_path = "/World/envs/env_0/Robot/right_hand_link/right_hand_line_z"
            right_line_x_path = "/World/envs/env_0/Robot/right_hand_link/right_hand_line_x"
            right_line_y_path = "/World/envs/env_0/Robot/right_hand_link/right_hand_line_y"
            right_prim_z = stage.GetPrimAtPath(right_line_z_path)
            right_prim_x = stage.GetPrimAtPath(right_line_x_path)
            right_prim_y = stage.GetPrimAtPath(right_line_y_path)

            if right_prim_z and right_prim_z.IsValid():
                imageable = UsdGeom.Imageable(right_prim_z)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

            if right_prim_x and right_prim_x.IsValid():
                imageable = UsdGeom.Imageable(right_prim_x)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

            if right_prim_y and right_prim_y.IsValid():
                imageable = UsdGeom.Imageable(right_prim_y)
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()

        except Exception as e:
            print(f" set hand lines visibility failed: {e}")


class X7SRelEnvCfg(X7SEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None, enable_pinocchio_ik: bool = True,
                 pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose, enable_pinocchio_ik=enable_pinocchio_ik,
                         pinocchio_urdf_path=pinocchio_urdf_path)
        self.name = "X7S-Rel"

        self.action_config.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_shoulder.*", "left_wrist.*", "left_elbow.*"],  # TODO
            body_name="left_hand_link",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.1, 0, 0.0)),  #
        )
        self.action_config.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"],  # TODO
            body_name="right_hand_link",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.1, 0.0, 0)),  #
        )


class X7SAbsEnvCfg(X7SEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None, enable_pinocchio_ik: bool = True,
                 pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose, enable_pinocchio_ik=enable_pinocchio_ik,
                         pinocchio_urdf_path=pinocchio_urdf_path)
        self.name = "X7S-Abs"

        if self.enable_pinocchio_ik:
            # When using external IK, switch to low-level joint position control
            self.action_config.left_arm_action = mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["left_shoulder_y", "left_shoulder_x", "left_shoulder_z",
                                                 "left_elbow_y", "left_elbow_x",
                                                 "left_wrist_y", "left_wrist_z"],
                scale=1.0, use_default_offset=True
            )
            self.action_config.right_arm_action = mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["right_shoulder_y", "right_shoulder_x", "right_shoulder_z",
                                                 "right_elbow_y", "right_elbow_x",
                                                 "right_wrist_y", "right_wrist_z"],
                scale=1.0, use_default_offset=True
            )
            self._last_left_output = None  # for EMA
            self._last_right_output = None  # for EMA
            self._ema_alpha = 0.1  # EMA smoothing factor (0.1-0.5, smaller is smoother)
            self._max_joint_delta = 0.1  # maximum joint change (rad/step)
            self._last_left_eef_pose = None
            self._last_right_eef_pose = None

        else:
            self.action_config.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["left_shoulder.*", "left_wrist_y", "left_elbow.*"],  # TODO
                body_name="link10",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.17, 0, 0.0)),  #
            )
            self.action_config.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["right_shoulder.*", "right_wrist_y", "right_elbow.*"],  # TODO
                body_name="link19",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                ),
                scale=1.0,
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.17, 0.0, 0)),  #
            )


class X7SJointCameraCfg:
    left_hand_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None
    eye_in_hand_camera: TiledCameraCfg = None


class X7SJointEnvCfg(X7SAbsEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None, enable_pinocchio_ik: bool = False,
                 pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose, enable_pinocchio_ik=enable_pinocchio_ik,
                         pinocchio_urdf_path=pinocchio_urdf_path)
        self.name = "X7S-Joint"
        self.camera_config = X7SJointCameraCfg()
        self.observation_cameras = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_hand_link/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0, 0.12), rot=(0.57923, 0.40558, -0.40558, -0.57923), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=36,  # FOV: 91.2°
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_link/right_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0, 0.12), rot=(0.57923, 0.40558, -0.40558, -0.57923), convention="opengl"),  # 0.0, -70, -90
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=36,  # FOV: 91.2°
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "eye_in_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/link1/eye_in_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.5, 0, 1.5), rot=(0.70711, 0.0, 0.0, -0.70711), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=27.7,  # Adjusted for 60° FOV
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            }
        }

        self.action_config.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["left_shoulder_y", "left_shoulder_x", "left_shoulder_z",
                                             "left_elbow_y", "left_elbow_x",
                                             "left_wrist_y", "left_wrist_z"],
            scale=1.0, use_default_offset=True
        )
        self.action_config.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["right_shoulder_y", "right_shoulder_x", "right_shoulder_z",
                                             "right_elbow_y", "right_elbow_x",
                                             "right_wrist_y", "right_wrist_z"],
            scale=1.0, use_default_offset=True
        )
