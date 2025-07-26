import torch
import numpy as np
from dataclasses import MISSING
from pathlib import Path
from . import transform_utils as T

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
import isaaclab.utils.math as math_utils
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

import lwlab.core.mdp as mdp
from .base import BaseRobotCfg
from lwlab.utils.isaaclab_assets.robots.unitree import G1_HIGH_PD_CFG, OFFSET_CONFIG_G1

##
# Pre-defined configs
##

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    base_action: mdp.RelativeJointPositionActionCfg = MISSING
    # body_action: mdp.RelativeJointPositionActionCfg = MISSING
    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING


class UnitreeG1HandEnvCfg(BaseRobotCfg):
    actions: ActionsCfg = ActionsCfg()
    robot_scale: float = 1.0
    robot_cfg: ArticulationCfg = G1_HIGH_PD_CFG
    offset_config = OFFSET_CONFIG_G1
    robot_name: str = "G1-Hand"
    render_cfgs = {
        "eye_in_hand_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/camera_link",
            offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0, 0.35), rot=(0.707107, 0, 0, -0.707107), convention="opengl"),
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
        )
    }
    robot_base_offset = {"pos": [0.0, 0.0, 0.8], "rot": [0.0, 0.0, 0.0]}
    robot_to_fixture_dist = 0.50

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set Actions for the specific robot type (franka)
        self.scene.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                # FrameTransformerCfg.FrameCfg(
                #     prim_path="{ENV_REGEX_NS}/Robot/torso_link",
                #     name="tool_torso",
                # ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_hand_palm_link",
                    name="tool_left_arm",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_hand_palm_link",
                    name="tool_right_arm",
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
        # self.actions.body_action = mdp.RelativeJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["waist_.*"],
        #     scale=0.5,  # 01,
        #     use_zero_offset=True,  # use default offset is not working for base action
        # )
        self.actions.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["left_shoulder.*", "left_wrist.*", "left_elbow.*"],  # TODO
            body_name="left_hand_palm_link",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            # body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        self.actions.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"],  # TODO
            body_name="right_hand_palm_link",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
            # body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        self.actions.left_hand_action = mdp.DexRetargetingActionCfg(
            asset_name="robot",
            config_path=Path(__file__).parent / "dex_retargeting" / "unitree_dex3_left.yaml",
            retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],# in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
            joint_names=["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                         "left_hand_index_0_joint", "left_hand_index_1_joint",
                         "left_hand_middle_0_joint", "left_hand_middle_1_joint"],
            # joint_names=["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint"]
            post_process_fn=self.post_process_left
        )
        self.actions.right_hand_action = mdp.DexRetargetingActionCfg(
            asset_name="robot",
            config_path=Path(__file__).parent / "dex_retargeting" / "unitree_dex3_right.yaml",
            retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],#in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
            joint_names=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                         "right_hand_index_0_joint", "right_hand_index_1_joint",
                         "right_hand_middle_0_joint", "right_hand_middle_1_joint"],
            # joint_names=["right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint"]
            post_process_fn=self.post_process_right
        )

        left_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/left_hand_thumb_2_link",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        setattr(self.scene, "left_gripper_contact", left_gripper_contact)

        right_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/right_hand_thumb_2_link",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        setattr(self.scene, "right_gripper_contact", right_gripper_contact)

        base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/pelvis",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )
        setattr(self.scene, "base_contact", base_contact)

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
