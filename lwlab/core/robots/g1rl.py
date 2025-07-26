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

import numpy as np
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg

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

    # left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.JointPositionActionCfg = MISSING
    # left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.BinaryJointPositionActionCfg = MISSING


class UnitreeG1HandEnvRLCfg(BaseRobotCfg):
    actions: ActionsCfg = ActionsCfg()
    robot_scale: float = 1.0
    robot_cfg: ArticulationCfg = G1_HIGH_PD_CFG
    offset_config = OFFSET_CONFIG_G1
    robot_name: str = "G1-Hand"
    robot_base_offset = {"pos": [0.0, 0.0, 0.8], "rot": [0.0, 0.0, 0.0]}

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Set Actions for the specific robot type (franka)
        self.scene.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            debug_vis=True,
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
        base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/pelvis",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )
        head_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.35, 1.05),
                                            rot=(0.556238, 0.299353, -0.376787, -0.677509),
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
        )
        global_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Scene/global_camera",
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 0.35, 1.05),
                                            rot=(0.556238, 0.299353, -0.376787, -0.677509),
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
        )
        setattr(self.scene, "head_camera", head_camera)
        setattr(self.scene, "global_camera", global_camera)
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

        self.actions.right_hand_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["right_hand_.*"],
            open_command_expr={'right_hand_index.*': 0.0, 'right_hand_middle.*': 0.0, 'right_hand_thumb.*': 0.0},
            close_command_expr={'right_hand_thumb_0_joint': 0.0, 'right_hand_thumb_1_joint': -np.pi / 6,
                                'right_hand_thumb_2_joint': -np.pi / 6, 'right_hand_index_1_joint': np.pi / 3, 'right_hand_middle_1_joint': np.pi / 3,
                                'right_hand_index_0_joint': np.pi / 6, 'right_hand_middle_0_joint': np.pi / 6},
        )

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"], scale=1, use_default_offset=True
        )

        self.set_reward_gripper_joint_names(["right_hand_.*"])
        self.set_reward_arm_joint_names(["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                         "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"])
