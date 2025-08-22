import torch
import numpy as np
import json
import time
from dataclasses import MISSING
from pathlib import Path
from lwlab.utils.math_utils import transform_utils as T

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import isaaclab.utils.math as math_utils
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.devices import DevicesCfg
from lwlab.core.devices import VRControllerDeviceCfg, VRHandDeviceCfg

import lwlab.core.mdp as mdp
from lwlab.core.robots.base import BaseRobotCfg
from isaac_arena.core.mdp.actions.decoupled_wbc_action import G1DecoupledWBCActionCfg
from isaac_arena.embodiments.unitree.assets_cfg import (
    G1_HIGH_PD_CFG,
    OFFSET_CONFIG_G1,
    G1_Loco_CFG,
    G1_GEARWBC_CFG,
)
from isaac_arena.embodiments.unitree.common import ActionsCfg, LocoActionsCfg, DecoupledWBCActionsCfg

##
# Pre-defined configs
##

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


class G1HandMixinEnvCfg(BaseRobotCfg):
    teleop_devices: DevicesCfg = DevicesCfg(
        devices={
            "vr-hand": VRHandDeviceCfg()
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.actions.left_hand_action = mdp.JointPositionMapActionCfg(
            asset_name="robot",
            joint_names=['left_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 'left_hand_index.*', 'left_hand_middle.*'],
            post_process_fn=self.post_process_hand,
        )
        self.actions.right_hand_action = mdp.JointPositionMapActionCfg(
            asset_name="robot",
            joint_names=['right_hand_thumb_1_joint', 'right_hand_thumb_2_joint', 'right_hand_index.*', 'right_hand_middle.*'],
            post_process_fn=self.post_process_hand,
        )

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device, base_move=True) -> torch.Tensor:
        base_action = action["base"]
        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 4],
                                                                           device.robot.data.body_link_quat_w[0, 4])
        # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
        base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
        base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
        base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        rot_mat_2d = torch.tensor([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ], device=device.env.device)
        robot_x = base_action[0]
        robot_y = base_action[1]
        local_xy = torch.tensor([robot_x, robot_y], device=device.env.device)
        world_xy = torch.matmul(rot_mat_2d, local_xy)
        base_action[0] = world_xy[0]
        base_action[1] = world_xy[1]

        left_arm_action = None
        right_arm_action = None
        if self.actions.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
        else:  # Absolute mode
            if base_move:
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
                    if arm_idx == 0:
                        left_arm_action = arm_action
                    else:
                        right_arm_action = arm_action
            else:
                left_arm_action = action["left_arm_abs"]
                right_arm_action = action["right_arm_abs"]
        left_finger_tips = action["left_finger_tips"][[0, 1, 2]].flatten()
        right_finger_tips = action["right_finger_tips"][[0, 1, 2]].flatten()
        return torch.concat([base_action, left_arm_action, right_arm_action,
                             left_finger_tips, right_finger_tips]).unsqueeze(0)

    def post_process_hand(self, target_pos, joint_names):
        if target_pos.shape[-1] == 1:
            target_pos = target_pos.repeat(1, len(joint_names))
        if joint_names[0].startswith('left'):
            target_pos = (target_pos + 1) / 2
        else:
            target_pos = -(target_pos + 1) / 2
        target_pos[:, 0:4] *= -1.4  # index 0 middle 0 index 1 middlrbe 1
        target_pos[:, 4] *= np.pi / 6  # thumb 1
        target_pos[:, 5] *= 1.3  # thumb 2
        return target_pos


class G1ControllerMixinEnvCfg(BaseRobotCfg):
    teleop_devices: DevicesCfg = DevicesCfg(
        devices={
            "vr-controller": VRControllerDeviceCfg()
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.actions.left_hand_action = mdp.DexRetargetingActionCfg(
            asset_name="robot",
            config_name="unitree_dex3_left",
            retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],# in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
            joint_names=["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                         "left_hand_index_0_joint", "left_hand_index_1_joint",
                         "left_hand_middle_0_joint", "left_hand_middle_1_joint"],
            # joint_names=["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint"]
            post_process_fn=self.post_process_left
        )
        self.actions.right_hand_action = mdp.DexRetargetingActionCfg(
            asset_name="robot",
            config_name="unitree_dex3_right",
            retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],#in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
            joint_names=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                         "right_hand_index_0_joint", "right_hand_index_1_joint",
                         "right_hand_middle_0_joint", "right_hand_middle_1_joint"],
            # joint_names=["right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint"]
            post_process_fn=self.post_process_right
        )

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        base_action = torch.zeros(3,)
        if action['rsqueeze'] > 0.5:
            base_action = torch.tensor([action['rbase'][0], 0, action['rbase'][1]], device=action['rbase'].device)
        else:
            base_action = torch.tensor([action['rbase'][0], action['rbase'][1], 0,], device=action['rbase'].device)

        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 4],
                                                                           device.robot.data.body_link_quat_w[0, 4])
        # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
        base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
        base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
        base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        rot_mat_2d = torch.tensor([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ], device=device.env.device)
        robot_x = base_action[0]
        robot_y = base_action[1]
        local_xy = torch.tensor([robot_x, robot_y], device=device.env.device)
        world_xy = torch.matmul(rot_mat_2d, local_xy)
        base_action[0] = world_xy[0]
        base_action[1] = world_xy[1]
        left_arm_action = None
        right_arm_action = None
        if self.actions.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
        else:  # Absolute mode
            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                pose_quat = abs_arm[3:7]
                combined_quat = T.quat_multiply(base_quat, pose_quat)
                arm_action = abs_arm.clone()
                rot_mat = T.quat2mat(base_quat)
                gripper_movement = torch.matmul(rot_mat, arm_action[:3])
                pose_movement = base_movement + gripper_movement
                arm_action[:3] = pose_movement
                arm_action[3] = combined_quat[3]  # Update quaternion from xyzw to wxyz
                arm_action[4:7] = combined_quat[:3]
                if arm_idx == 0:
                    left_arm_action = arm_action  # Robot frame
                else:
                    right_arm_action = arm_action  # Robot frame
        left_gripper = torch.tensor([-action["left_gripper"]], device=action['rbase'].device)
        right_gripper = torch.tensor([-action["right_gripper"]], device=action['rbase'].device)
        return torch.concat([base_action, left_arm_action, right_arm_action,
                             left_gripper, right_gripper]).unsqueeze(0)

    def post_process_left(self, retargeted_actions, num_joints):
        scale = 1.3
        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = (retargeted_actions[:, 0]) * np.pi / 2 * scale  # in 0
        actions[:, 1] = (retargeted_actions[:, 1]) * np.pi / 2 * scale  # mi 0
        actions[:, 2] = 0  # th 0
        actions[:, 3] = retargeted_actions[:, 0] * scale  # in 1
        actions[:, 4] = retargeted_actions[:, 1] * scale  # mi 1
        actions[:, 5] = np.pi / 3 + retargeted_actions[:, 2] * scale  # th 1
        actions[:, 6] = np.pi / 3 + retargeted_actions[:, 2] * scale  # th 2
        return actions

    def post_process_right(self, retargeted_actions, num_joints):
        scale = 1.3
        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = (retargeted_actions[:, 0]) * np.pi / 2 * scale  # in 0
        actions[:, 1] = (retargeted_actions[:, 1]) * np.pi / 2 * scale  # mi 0
        actions[:, 2] = 0  # th 0
        actions[:, 3] = retargeted_actions[:, 0] * scale  # in 1
        actions[:, 4] = retargeted_actions[:, 1] * scale  # mi 1
        actions[:, 5] = -np.pi / 3 - retargeted_actions[:, 2] * scale  # th 1
        actions[:, 6] = -np.pi / 3 - retargeted_actions[:, 2] * scale  # th 2
        return actions


class UnitreeG1EnvCfg(BaseRobotCfg):
    actions: ActionsCfg = ActionsCfg()
    robot_scale: float = 1.0
    robot_cfg: ArticulationCfg = G1_HIGH_PD_CFG
    offset_config = OFFSET_CONFIG_G1
    robot_base_link: str = "pelvis"
    left_contact_body_name: str = "left_hand_thumb_2_link"
    right_contact_body_name: str = "right_hand_thumb_2_link"
    robot_to_fixture_dist: float = 0.50
    robot_base_offset = {"pos": [0.0, 0.0, 0.8], "rot": [0.0, 0.0, 0.0]}
    observation_cameras: dict = {
        "left_hand_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_hand_palm_link/left_hand_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0.05), rot=(0.5, -0.5, -0.5, -0.5), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=83.0,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["teleop"]
        },
        "first_person_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link/first_person_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(-0.4, 0.0, 0.8), rot=(0.66655, 0.23604, -0.23604, -0.66655), convention="opengl"),
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
            "tags": ["teleop"]
        },
        "right_hand_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_hand_palm_link/right_hand_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0.05), rot=(0.5, 0.5, 0.5, -0.5), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=83.0,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["teleop"]
        },
        "left_shoulder_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link/left_shoulder_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.7, 0.7), rot=(0.28818, 0.25453, -0.33065, -0.86188), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=62,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["teleop"]
        },
        "eye_in_hand_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link/eye_in_hand_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0, 0.4), rot=(0.707107, 0, 0, -0.707107), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=83.0,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["teleop"]
        },
        "right_shoulder_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link/right_shoulder_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, -0.7, 0.7), rot=(0.71134, 0.39777, -0.12609, -0.56558), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=62,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["teleop"]
        },
        "hand_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/hand_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.04353, -0.01712, 0.16093),
                                                rot=(0.65164, 0.27138, -0.01743, -0.70811),
                                                convention="opengl"),
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
            "tags": ["rl"]
        },
        "global_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link/global_camera",
                # offset=TiledCameraCfg.OffsetCfg(pos=(0.25, 0.3, 0.38),
                #                                 rot=(-0.01853, -0.10431, 0.33021, 0.93794),
                #                                 convention="opengl"),
                offset=TiledCameraCfg.OffsetCfg(pos=(0.9, 0, 0.24521),
                                                rot=(0.56538, 0.43517, 0.42607, 0.55627),
                                                convention="opengl"),
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
            "tags": []
        },
        "d435_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/d435_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0, 0, 0),
                                                rot=(0.52217, 0.4768, -0.4768, -0.52217),
                                                convention="opengl"),
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
            "tags": ["rl"]
        }
    }

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
                    name="tool_left_arm",
                    offset=OffsetCfg(pos=(0.0415, 0.003, 0.0)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                    name="tool_right_arm",
                    offset=OffsetCfg(pos=(0.0415, -0.003, 0.0)),
                ),
            ],
        )
        self.actions.base_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["base_.*"],
            scale={
                "base_x_joint": 0.01,
                "base_y_joint": 0.01,
                "base_yaw_joint": 0.02,
            },  # 01,
            # scale=0.01,  # 01,
            use_zero_offset=True,  # use default offset is not working for base action
        )
        self.actions.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_shoulder.*", "left_wrist.*", "left_elbow.*"],  # TODO
            body_name="left_wrist_yaw_link",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0415, 0.003, 0.0)),
        )
        self.actions.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"],  # TODO
            body_name="right_wrist_yaw_link",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0415, -0.003, 0.0)),
        )
        # self.actions.left_hand_action is missing, need to be specified in the child class
        # self.actions.right_hand_action is missing, need to be specified in the child class
        self.scene.left_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{self.left_contact_body_name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        self.scene.right_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{self.right_contact_body_name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )


class UnitreeG1LocoEnvCfg(UnitreeG1EnvCfg):
    actions: LocoActionsCfg = LocoActionsCfg()
    robot_cfg: ArticulationCfg = G1_Loco_CFG
    robot_name: str = "G1-Loco"
    robot_base_offset = {"pos": [0.0, 0.0, 0.83], "rot": [0.0, 0.0, 0.0]}

    def __post_init__(self):
        super().__post_init__()
        self.actions.base_action = mdp.LegPositionActionCfg(
            asset_name="robot",
            joint_names=['left_hip_pitch_joint', 'right_hip_pitch_joint', 'left_hip_roll_joint',
                         'right_hip_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint',
                         'left_knee_joint', 'right_knee_joint', 'left_ankle_pitch_joint',
                         'right_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint'],
            body_name="base",
            scale=1.0,  # action scaling factor
            loco_config="g1_loco.yaml",  # gait control configuration file
            squat_config="g1_squat.yaml"  # squat control configuration file
        )


class UnitreeG1HandEnvCfg(UnitreeG1EnvCfg, G1HandMixinEnvCfg):
    robot_name: str = "G1-Hand"


class UnitreeG1ControllerEnvCfg(UnitreeG1EnvCfg, G1ControllerMixinEnvCfg):
    robot_name: str = "G1-Controller"


class UnitreeG1LocoHandEnvCfg(UnitreeG1LocoEnvCfg, G1HandMixinEnvCfg):
    robot_name: str = "G1-Loco-Hand"

    # def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
    #     base_action = action["base"]
    #     base_action = torch.cat([base_action, torch.tensor([action["base_loco_mode"]], dtype=action["base"].dtype, device=action["base"].device)], dim=0)
    #     left_arm_action = None
    #     right_arm_action = None
    #     if self.actions.left_arm_action.controller.use_relative_mode:  # Relative mode
    #         left_arm_action = action["left_arm_delta"]
    #         right_arm_action = action["right_arm_delta"]
    #     else:  # Absolute mode
    #         _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
    #                                                                            device.robot.data.root_com_quat_w[0],
    #                                                                            device.robot.data.body_link_pos_w[0, 4],
    #                                                                            device.robot.data.body_link_quat_w[0, 4])
    #         # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
    #         base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
    #         base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
    #         base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

    #         for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
    #             pose_quat = abs_arm[3:7]
    #             combined_quat = T.quat_multiply(base_quat, pose_quat)
    #             arm_action = abs_arm.clone()
    #             rot_mat = T.quat2mat(base_quat)
    #             gripper_movement = torch.matmul(rot_mat, arm_action[:3])
    #             pose_movement = base_movement + gripper_movement
    #             arm_action[:3] = pose_movement
    #             arm_action[3] = combined_quat[3]  # Update rotation quaternion from xyzw to wxyz
    #             arm_action[4:7] = combined_quat[:3]
    #             if arm_idx == 0:
    #                 left_arm_action = arm_action  # Robot frame
    #             else:
    #                 right_arm_action = arm_action  # Robot frame
    #     left_finger_tips = action["left_finger_tips"][[0, 1, 2]].flatten()
    #     right_finger_tips = action["right_finger_tips"][[0, 1, 2]].flatten()
    #     return torch.concat([left_arm_action, right_arm_action,
    #                          left_finger_tips, right_finger_tips, base_action]).unsqueeze(0)


class UnitreeG1LocoControllerEnvCfg(UnitreeG1LocoEnvCfg, G1ControllerMixinEnvCfg):
    robot_name: str = "G1-Loco-Controller"


class UnitreeG1HandEnvRLCfg(UnitreeG1HandEnvCfg):
    robot_name: str = "G1-RL"
    robot_to_fixture_dist: float = 0.20
    robot_base_offset = {"pos": [0.0, 0.2, 0.92], "rot": [0.0, 0.0, 0.0]}

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        del self.actions.base_action
        del self.actions.left_hand_action
        del self.actions.left_arm_action
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"], scale=1, use_default_offset=True
        )
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
                    name="tool_right_arm",
                    offset=OffsetCfg(
                        pos=(0.13, 0.04, 0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_thumb_2_link",
                    name="tool_thumb_tip",
                    offset=OffsetCfg(
                        pos=(0.114, -0.02, 0),
                        rot=(0.7071068, 0, 0.7071068, 0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
                    name="tool_index_tip",
                    offset=OffsetCfg(
                        pos=(0.05, 0.0, 0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
                    name="tool_middle_tip",
                    offset=OffsetCfg(
                        pos=(0.05, 0.0, 0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_index_1_link",
                    name="tool_index_tip",
                    offset=OffsetCfg(
                        pos=(0.05, 0.0, 0),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_middle_1_link",
                    name="tool_middle_tip",
                    offset=OffsetCfg(
                        pos=(0.05, 0.0, 0),
                    ),
                ),
            ],
        )
        self.set_reward_gripper_joint_names(["right_hand_.*"])
        self.set_reward_arm_joint_names(["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                         "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"])


class UnitreeG1ControllerDecoupledWBCEnvCfg(UnitreeG1ControllerEnvCfg):
    actions: DecoupledWBCActionsCfg = DecoupledWBCActionsCfg()
    robot_cfg: ArticulationCfg = G1_GEARWBC_CFG
    robot_name: str = "G1-Controller-DecoupledWBC"

    def __post_init__(self):
        super().__post_init__()
        self.actions.base_action = G1DecoupledWBCActionCfg(asset_name="robot", joint_names=[".*"])

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        base_action = torch.zeros(3,)
        if action['rsqueeze'] > 0.5:
            base_action = torch.tensor([action['rbase'][0], 0, action['rbase'][1]], device=action['rbase'].device)
        else:
            base_action = torch.tensor([action['rbase'][0], action['rbase'][1], 0,], device=action['rbase'].device)

        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 4],
                                                                           device.robot.data.body_link_quat_w[0, 4])
        # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
        base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
        base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
        base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        rot_mat_2d = torch.tensor([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ], device=device.env.device)
        robot_x = base_action[0]
        robot_y = base_action[1]
        local_xy = torch.tensor([robot_x, robot_y], device=device.env.device)
        world_xy = torch.matmul(rot_mat_2d, local_xy)
        base_action[0] = world_xy[0]
        base_action[1] = world_xy[1]
        left_arm_action = None
        right_arm_action = None
        if self.actions.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
        else:  # Absolute mode
            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                pose_quat = abs_arm[3:7]
                combined_quat = T.quat_multiply(base_quat, pose_quat)
                arm_action = abs_arm.clone()
                rot_mat = T.quat2mat(base_quat)
                gripper_movement = torch.matmul(rot_mat, arm_action[:3])
                pose_movement = base_movement + gripper_movement
                arm_action[:3] = pose_movement
                arm_action[3] = combined_quat[3]  # Update quaternion from xyzw to wxyz
                arm_action[4:7] = combined_quat[:3]
                if arm_idx == 0:
                    left_arm_action = arm_action  # Robot frame
                else:
                    right_arm_action = arm_action  # Robot frame
        left_gripper = torch.tensor([-action["left_gripper"]], device=action['rbase'].device)
        right_gripper = torch.tensor([-action["right_gripper"]], device=action['rbase'].device)
        return torch.concat([left_arm_action, right_arm_action,
                             left_gripper, right_gripper, base_action]).unsqueeze(0)
