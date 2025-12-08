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
from isaaclab.utils import configclass

import lw_benchhub.core.mdp as mdp
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.log_utils import get_default_logger

##
# Pre-defined configs
##
from .assets_cfg import DOOUBLE_PIPER_CFG, DOOUBLE_PIPER_HIGH_PD_CFG, DOOUBLE_PIPER_OFFSET_CONFIG, DOUBLE_PIPER_VIS_HELPER_CFG  # isort: skip
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lw_benchhub.utils.pinocchio_ik.piper_ik import PiperPinocchioIK  # isort: skip
from isaaclab_arena.utils.pose import Pose  # isort: skip
from lw_benchhub.core.robots.robot_arena_base import LwEmbodimentBase  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg | mdp.JointPositionActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class DoublePiperCameraCfg:
    left_hand_camera: TiledCameraCfg = None
    first_person_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None
    left_shoulder_camera: TiledCameraCfg = None
    eye_in_hand_camera: TiledCameraCfg = None
    right_shoulder_camera: TiledCameraCfg = None


@configclass
class DoublePiperSceneCfg:
    robot: ArticulationCfg = DOOUBLE_PIPER_CFG
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_.*/hand_link_l",
                name="ee_tcp_l",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/piper_.*/hand_link_r",
                name="ee_tcp_r",
            ),
        ],
    )


class DoublePiperEnvCfg(LwEmbodimentBase):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None,
                 headless_mode: bool = False):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)

        self.action_config = ActionsCfg()
        self.robot_vis_helper_cfg = DOUBLE_PIPER_VIS_HELPER_CFG
        self.observation_cameras: dict = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/piper_L/hand_link_l/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.2, 0.0, -0.02), rot=(-0.18301, -0.68301, 0.68301, 0.18301), convention="opengl"),
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
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            # add [-0.5, 0.5) random z offset to the camera position
            "first_person_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/piper_R/dummy_link/first_person_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.02, 0.3, 0.52 + torch.rand(1).item() * 0.1 - 0.05), rot=(0.64597, 0.28761, -0.28761, -0.64597), convention="opengl"),  # rot: (0.0, -45, -90) with random z offset
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
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION,
                                 ExecuteMode.REPLAY_JOINT_TARGETS]
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/piper_R/hand_link_r/right_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.2, 0.0, -0.02), rot=(-0.18301, -0.68301, 0.68301, 0.18301), convention="opengl"),
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
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
        }
        self.robot_scale = robot_scale
        self.offset_config = DOOUBLE_PIPER_OFFSET_CONFIG
        self.headless_mode = headless_mode
        self.scene_config = DoublePiperSceneCfg()
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.camera_config = DoublePiperCameraCfg()
        self.name = "DoublePiper"

        # Set Actions for the specific robot type
        self.action_config.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint.*_l"],
            open_command_expr={"finger_joint_left_l": 0.035, "finger_joint_right_l": -0.035},
            close_command_expr={"finger_joint.*_l": 0.0},
        )

        self.action_config.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint.*_r"],
            open_command_expr={"finger_joint_left_r": 0.035, "finger_joint_right_r": -0.035},
            close_command_expr={"finger_joint.*_r": 0.0},
        )

        self.left_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/piper_L/left_finger_link",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )

        self.right_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/piper_R/right_finger_link",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )

        self.base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/root",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )

        # Initialize IK solvers for left/right arms if enabled
        self.enable_pinocchio_ik = enable_pinocchio_ik
        self.pinocchio_urdf_path = pinocchio_urdf_path
        if self.enable_pinocchio_ik:
            if self.pinocchio_urdf_path is not None:
                self._ik_left = PiperPinocchioIK(self.pinocchio_urdf_path)
                self._ik_right = PiperPinocchioIK(self.pinocchio_urdf_path)
            else:
                self._ik_left = PiperPinocchioIK()
                self._ik_right = PiperPinocchioIK()

        # States for gripper toggles are already initialized below

        self.left_gripper_closed = True
        self.right_gripper_closed = True
        self.left_gripper_prev = 1.0
        self.right_gripper_prev = 1.0
        self.last_pos = None

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs

        # Filter once at the beginning to avoid redundant smoothing
        action = self.filter_action(action)

        # Build joint targets via Pinocchio IK from absolute poses
        # Reorder to [x,y,z,w,x,y,z] if needed (source typically [x,y,z,qx,qy,qz,qw])
        left_pose = action["left_arm_abs"].clone()
        # left_pose[1] -= 0.3
        left_pose[3:] = left_pose[[6, 3, 4, 5]]
        right_pose = action["right_arm_abs"].clone()
        right_pose[3:] = right_pose[[6, 3, 4, 5]]
        # right_pose[1] += 0.3
        left_pose = left_pose.repeat(num_envs, 1)
        right_pose = right_pose.repeat(num_envs, 1)

        # Solve IK
        l_q_np, _ = self._ik_left.solve_pose_to_joints(left_pose.detach().cpu().numpy(), warm_start=None)
        r_q_np, _ = self._ik_right.solve_pose_to_joints(right_pose.detach().cpu().numpy(), warm_start=None)
        left_arm_action = torch.tensor(l_q_np, device=device.env.device, dtype=torch.float32)
        right_arm_action = torch.tensor(r_q_np, device=device.env.device, dtype=torch.float32)

        left_gripper_pressed = action["left_gripper"] > 0
        right_gripper_pressed = action["right_gripper"] > 0
        left_gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        left_gripper_action[:] = -1.0 if left_gripper_pressed else 1.0
        right_gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        right_gripper_action[:] = -1.0 if right_gripper_pressed else 1.0

        self.set_right_hand_lines_visibility(not right_gripper_pressed)

        self.set_left_hand_lines_visibility(not left_gripper_pressed)

        return torch.concat([left_arm_action, right_arm_action, left_gripper_action, right_gripper_action], dim=1)

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

            left_line_z_path = "/World/envs/env_0/Robot/piper_L/hand_link_l/left_hand_line_z"
            left_line_x_path = "/World/envs/env_0/Robot/piper_L/hand_link_l/left_hand_line_x"
            left_line_y_path = "/World/envs/env_0/Robot/piper_L/hand_link_l/left_hand_line_y"
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

        except Exception as e:
            print(f" set hand lines visibility failed: {e}")

    def reset_robot_cfg_state(self):
        self.left_gripper_closed = True
        self.right_gripper_closed = True
        self.left_gripper_prev = 1.0
        self.right_gripper_prev = 1.0
        self.last_pos = None

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

            right_line_z_path = "/World/envs/env_0/Robot/piper_R/hand_link_r/right_hand_line_z"
            right_line_x_path = "/World/envs/env_0/Robot/piper_R/hand_link_r/right_hand_line_x"
            right_line_y_path = "/World/envs/env_0/Robot/piper_R/hand_link_r/right_hand_line_y"
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


class DoublePiperAbsEnvCfg(DoublePiperEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None,
                 headless_mode: bool = False):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose,
                         enable_pinocchio_ik=enable_pinocchio_ik, pinocchio_urdf_path=pinocchio_urdf_path,
                         headless_mode=headless_mode)
        self.name = "DoublePiper-Abs"
        self.scene_config.robot = DOOUBLE_PIPER_HIGH_PD_CFG
        # Switch both arms to joint position control; IK converts pose->joints in preprocess
        self.action_config.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint1_l", "joint2_l", "joint3_l", "joint5_l", "joint6_l"], scale=1, use_default_offset=True
        )

        self.action_config.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint1_r", "joint2_r", "joint3_r", "joint5_r", "joint6_r"], scale=1, use_default_offset=True
        )

    def reset_robot_cfg_state(self):
        super().reset_robot_cfg_state()
        self._ik_left.reset()
        self._ik_right.reset()


class DoublePiperRelEnvCfg(DoublePiperEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None,
                 headless_mode: bool = False):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose,
                         enable_pinocchio_ik=enable_pinocchio_ik, pinocchio_urdf_path=pinocchio_urdf_path,
                         headless_mode=headless_mode)
        self.name = "DoublePiper-Rel"
        self.scene_config.robot = DOOUBLE_PIPER_HIGH_PD_CFG

        self.first_action = True
        self.prev_abs_left_pose = None
        self.prev_abs_right_pose = None

        # Switch both arms to joint position control; IK converts pose->joints in preprocess
        self.action_config.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint1_l", "joint2_l", "joint3_l", "joint5_l", "joint6_l"], scale=1, use_default_offset=True
        )

        self.action_config.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint1_r", "joint2_r", "joint3_r", "joint5_r", "joint6_r"], scale=1, use_default_offset=True
        )

    def reset_robot_cfg_state(self):
        super().reset_robot_cfg_state()
        self._ik_left.reset()
        self._ik_right.reset()
        self.first_action = True
        self.prev_abs_left_pose = None
        self.prev_abs_right_pose = None

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs

        # Filter once at the beginning to avoid redundant smoothing
        action = self.filter_action(action)

        if self.first_action:
            self.prev_abs_left_pose = action["prev_left_arm_abs_val"].clone()
            self.prev_abs_right_pose = action["prev_right_arm_abs_val"].clone()
            self.first_action = False

        rel_left_pose = action["left_arm_delta"].clone()
        rel_right_pose = action["right_arm_delta"].clone()

        get_default_logger().info(f"rel_left_pose: {rel_left_pose}")
        print(f"rel_left_pose: {rel_left_pose}")
        get_default_logger().info(f"rel_right_pose: {rel_right_pose}")
        print(f"rel_right_pose: {rel_right_pose}")
        get_default_logger().info("--------------------------------")

        # Calculate new poses by adding relative deltas to current poses
        left_pose = rel_left_pose + self.prev_abs_left_pose
        self.prev_abs_left_pose = left_pose.clone()

        # Convert axis-angle to quaternion for left arm
        from scipy.spatial.transform import Rotation as R
        left_rot = R.from_rotvec(left_pose[3:6].cpu().numpy())  # Convert axis-angle to rotation
        left_quat = left_rot.as_quat()  # Convert to quaternion [x, y, z, w]

        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        left_pose = torch.cat([left_pose[:3], torch.tensor(left_quat, device=left_pose.device, dtype=left_pose.dtype)])

        left_pose[3:] = left_pose[[6, 3, 4, 5]]

        right_pose = rel_right_pose + self.prev_abs_right_pose
        self.prev_abs_right_pose = right_pose.clone()

        # Convert axis-angle to quaternion for right arm
        right_rot = R.from_rotvec(right_pose[3:6].cpu().numpy())  # Convert axis-angle to rotation
        right_quat = right_rot.as_quat()  # Convert to quaternion [x, y, z, w]

        # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
        right_pose = torch.cat([right_pose[:3], torch.tensor(right_quat, device=right_pose.device, dtype=right_pose.dtype)])

        right_pose[3:] = right_pose[[6, 3, 4, 5]]

        left_pose = left_pose.repeat(num_envs, 1)
        right_pose = right_pose.repeat(num_envs, 1)

        # Solve IK
        # if getattr(self, "enable_pinocchio_ik", False):
        l_q_np, _ = self._ik_left.solve_pose_to_joints(left_pose.detach().cpu().numpy(), warm_start=None)
        r_q_np, _ = self._ik_right.solve_pose_to_joints(right_pose.detach().cpu().numpy(), warm_start=None)
        left_arm_action = torch.tensor(l_q_np, device=device.env.device, dtype=torch.float32)
        right_arm_action = torch.tensor(r_q_np, device=device.env.device, dtype=torch.float32)

        left_gripper_pressed = action["left_gripper"] > 0
        right_gripper_pressed = action["right_gripper"] > 0
        left_gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        left_gripper_action[:] = -1.0 if left_gripper_pressed else 1.0
        right_gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        right_gripper_action[:] = -1.0 if right_gripper_pressed else 1.0

        self.set_right_hand_lines_visibility(not right_gripper_pressed)

        self.set_left_hand_lines_visibility(not left_gripper_pressed)

        return torch.concat([left_arm_action, right_arm_action, left_gripper_action, right_gripper_action], dim=1)


class DoublePiperRLEnvCfg(DoublePiperEnvCfg):
    def __init__(self, enable_cameras: bool = False, robot_scale: float = 1.0, initial_pose: Pose | None = None,
                 enable_pinocchio_ik: bool = True, pinocchio_urdf_path: str | None = None,
                 headless_mode: bool = False):
        super().__init__(enable_cameras=enable_cameras, robot_scale=robot_scale, initial_pose=initial_pose,
                         enable_pinocchio_ik=enable_pinocchio_ik, pinocchio_urdf_path=pinocchio_urdf_path,
                         headless_mode=headless_mode)
        self.name = "DoublePiper-Rel"
        self.scene_config.robot = DOOUBLE_PIPER_HIGH_PD_CFG

        # Set actions for the specific robot type (piper)
        self.action_config.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*_l"], scale=1, use_default_offset=True
        )

        self.action_config.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint.*_r"], scale=1, use_default_offset=True
        )

        self.set_reward_gripper_joint_names(["joint.*_l", "joint.*_r"])
