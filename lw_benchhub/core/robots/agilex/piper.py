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
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import lw_benchhub.core.mdp as mdp

##
# Pre-defined configs
##
from .assets_cfg import PIPER_CFG, PIPER_HIGH_PD_CFG, PIPER_OFFSET_CONFIG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lw_benchhub.utils.pinocchio_ik.piper_ik import PiperPinocchioIK  # isort: skip
from isaaclab_arena.utils.pose import Pose  # isort: skip
from lw_benchhub.core.robots.robot_arena_base import LwEmbodimentBase  # isort: skip
from lw_benchhub.utils.env import ExecuteMode


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class PiperCameraCfg:
    left_hand_camera: TiledCameraCfg = None
    first_person_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None
    left_shoulder_camera: TiledCameraCfg = None
    eye_in_hand_camera: TiledCameraCfg = None
    right_shoulder_camera: TiledCameraCfg = None


@configclass
class PiperSceneCfg:
    robot: ArticulationCfg = PIPER_CFG
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/dummy_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/hand_link",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_finger_link",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_finger_link",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )


class PiperEnvCfg(LwEmbodimentBase):

    robot_cfg: ArticulationCfg = PIPER_CFG
    robot_scale: float = 1.0
    enable_pinocchio_ik: bool = True
    pinocchio_urdf_path: str | None = None

    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)

        self.name = "piper"
        self.robot_scale = robot_scale
        self.offset_config = PIPER_OFFSET_CONFIG

        # configs
        self.action_config = ActionsCfg()
        self.scene_config = PiperSceneCfg()
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.camera_config = PiperCameraCfg()

        self.observation_cameras: dict = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/hand_link/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.2, 0.0, -0.02), rot=(-0.18301, -0.68301, 0.68301, 0.18301), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=40.6,
                        focus_distance=400.0,
                        horizontal_aperture=38.11,
                        clipping_range=(0.01, 3.0),
                        lock_camera=True
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": ["teleop"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL]

            },
            # add [-0.5, 0.5) random z offset to the camera position
            "eye_in_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/dummy_link/eye_in_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.7, -0.7, 0.5), rot=(0.8, 0.5, 0.16657, 0.2414), convention="opengl"),  # rot: (0.0, -45, -90) with random z offset
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=40.6,
                        focus_distance=400.0,
                        horizontal_aperture=38.11,
                        clipping_range=(0.01, 3.0),
                        lock_camera=True
                    ),
                    width=480,
                    height=480,
                    update_period=0.05,
                ),
                "tags": ["teleop"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL]

            }
        }

        self.base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/dummy_link",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )

        # Gripper action mapping
        self.action_config.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint.*"],
            open_command_expr={"finger_joint_left": 0.035, "finger_joint_right": -0.035},
            close_command_expr={"finger_joint.*": 0.0},
        )

        # IK setup and arm action selection
        self.enable_pinocchio_ik = enable_pinocchio_ik
        self.pinocchio_urdf_path = pinocchio_urdf_path
        if self.enable_pinocchio_ik:
            if self.pinocchio_urdf_path is not None:
                self._ik_solver = PiperPinocchioIK(self.pinocchio_urdf_path)
            else:
                self._ik_solver = PiperPinocchioIK()
            self.action_config.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["joint.*"], scale=1, use_default_offset=True
            )
        else:
            self.action_config.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["joint.*"],
                body_name="hand_link",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                scale=1.0,
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            )
        self.last_pos = None

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs

        # Gripper mapping stays the same
        gripper = action["arm_gripper"] > 0
        gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        gripper_action[:] = -1.0 if gripper else 1.0

        # If IK is disabled, return zeros for joints (will be ignored by JointPositionAction if not set)
        # However, we expect IK enabled when using JointPositionAction below

        # Absolute pose input expected: [x,y,z,w,x,y,z]
        pose = action["arm_abs"]  # shape (7,)
        pose[3:] = pose[[6, 3, 4, 5]]
        # Ensure wxyz ordering and normalize quaternion
        # Current code previously did: arm_action[3:] = arm_action[[6, 3, 4, 5]] from [x,y,z,?, ?, ?, ?]
        # Here we assume input already [x,y,z,w,x,y,z]. If not, adapt upstream.
        pos = pose[0:3]
        quat = pose[3:7]
        quat = quat / torch.clamp(torch.linalg.norm(quat), min=1e-8)

        # Apply TCP->hand_link inverse offset (equivalent to previous body_offset of +0.107 along z of hand)
        # We express offset in base frame by left-multiplying target by inverse of offset at hand frame.
        # Here we approximate by shifting target position by -0.107 in hand's local z.
        # Since we don't have the current hand orientation here, we approximate using target orientation.
        tcp_offset = torch.tensor([0.0, 0.0, 0.107], device=device.env.device)
        # rotate offset by target quat and subtract
        # util: quat_apply(quat, vec)
        from isaaclab.utils import math as math_utils
        offset_b = math_utils.quat_apply(quat, tcp_offset)
        pos_hand = pos - offset_b

        # Batch for solver
        targets = torch.zeros((num_envs, 7), device=device.env.device)
        targets[:, 0:3] = pos_hand.unsqueeze(0).repeat(num_envs, 1)
        targets[:, 3:7] = quat.unsqueeze(0).repeat(num_envs, 1)

        # Warm start: optionally use current joint positions (disabled here to avoid DOF mismatch)
        current_q = None

        # Solve via Pinocchio
        targets_np = targets.detach().cpu().numpy()
        q_np, succ = self._ik_solver.solve_pose_to_joints(targets_np, warm_start=current_q)
        q_t = torch.tensor(q_np, device=device.env.device, dtype=torch.float32)

        return torch.concat([q_t, gripper_action], dim=1)


class PiperAbsEnvCfg(PiperEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose,
                         enable_pinocchio_ik=enable_pinocchio_ik, pinocchio_urdf_path=pinocchio_urdf_path)
        self.scene_config.robot = PIPER_HIGH_PD_CFG
        # Switch to joint position control; IK converts pose->joints in preprocess
        self.action_config.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"], scale=1, use_default_offset=True
        )


class PiperRLEnvCfg(PiperEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose,
                         enable_pinocchio_ik=enable_pinocchio_ik, pinocchio_urdf_path=pinocchio_urdf_path)
        self.scene_config.robot = PIPER_HIGH_PD_CFG
        self.action_config.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint1", "joint2", "joint3", "joint5", "joint6"], scale=1, use_default_offset=True
        )
