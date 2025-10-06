import torch
from dataclasses import MISSING
from typing import List
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

import lwlab.core.mdp as mdp

from lwlab.core.robots.base import BaseRobotCfg
##
# Pre-defined configs
##
from .assets_cfg import SO101_FOLLOWER_CFG, SO100_FOLLOWER_CFG, SO101_FOLLOWER_YELLOW_CFG  # isort: skip
from lwlab.utils.lerobot_utils import convert_action_from_so101_leader
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lwlab.utils.math_utils import transform_utils as T
import isaaclab.utils.math as math_utils

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = MISSING


class BaseLERobotEnvCfg(BaseRobotCfg):
    robot_cfg: ArticulationCfg = SO101_FOLLOWER_CFG
    robot_name: str = "LeRobot-RL"
    robot_scale: float = 1.0
    actions: ActionsCfg = ActionsCfg()
    robot_base_offset = {"pos": [-0.8, -0.75, 0.9], "rot": [0.0, 0.0, torch.pi / 2]}
    observation_cameras: dict = {
        "hand_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=36.5,
                    focus_distance=400.0,
                    horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["teleop"]
        },
        "global_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/camera_base/global_camera",
                # offset=TiledCameraCfg.OffsetCfg(pos=(0.25, -0.55, 0.27), rot=(0.74698, 0.54011, 0.23182, 0.31074), convention="opengl"), # RL
                offset=TiledCameraCfg.OffsetCfg(pos=(0.2, -0.65, 0.3), rot=(0.8, 0.5, 0.16657, 0.2414), convention="opengl"),  # IL
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=40.6,
                    focus_distance=400.0,
                    horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
                    clipping_range=(0.01, 3.0),
                    lock_camera=True
                ),
                width=224,
                height=224,
                update_period=0.05,
            ),
            "tags": ["rl", "teleop"]
        }
    }

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)


# @configclass
class LERobotEnvRLCfg(BaseLERobotEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    robot_base_offset = {"pos": [1.2, -0.8, 0.892], "rot": [0.0, 0.0, 0]}

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type
        self.actions.arm_action = mdp.RelJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow_flex", "wrist.*", "gripper"],
            scale={"shoulder.*": 0.05, "elbow_flex": 0.05, "wrist.*": 0.05, "gripper": 0.2},
            use_zero_offset=True,
            clip={"shoulder.*": (-1.0, 1.0), "elbow_flex": (-1.0, 1.0), "wrist.*": (-1.0, 1.0), "gripper": (-1.0, 1.0)}
        )

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper",  # 夹爪
                    name="tool_gripper",
                    offset=OffsetCfg(
                        pos=(-0.011, -0.0001, -0.0953),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/jaw",  # 夹爪
                    name="tool_jaw",
                    offset=OffsetCfg(
                        pos=(-0.01, -0.073, 0.019),
                    ),
                ),
            ],
        )

        base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/base",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )
        left_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        right_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        gripper_table_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/counter_1_front_group/top_geometry"],
        )
        jaw_table_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/counter_1_front_group/top_geometry"],
        )
        gripper_object_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/gripper",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/object/BuildingBlock003"],
        )
        jaw_object_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/object/BuildingBlock003"],
        )
        setattr(self.scene, "base_contact", base_contact)
        setattr(self.scene, "left_gripper_contact", left_gripper_contact)
        setattr(self.scene, "right_gripper_contact", right_gripper_contact)
        setattr(self.scene, "gripper_table_contact", gripper_table_contact)
        setattr(self.scene, "jaw_table_contact", jaw_table_contact)
        setattr(self.scene, "gripper_object_contact", gripper_object_contact)
        setattr(self.scene, "jaw_object_contact", jaw_object_contact)

        self.viewport_cfg = {
            "offset": [-1.0, 0.0, 2.0],
            "lookat": [1.0, 0.0, -0.7]
        }

        self.set_reward_gripper_joint_names(["gripper"])


class LERobot100EnvRLCfg(LERobotEnvRLCfg):
    robot_cfg: ArticulationCfg = SO100_FOLLOWER_CFG
    robot_name: str = "LeRobot100-RL"
    actions: ActionsCfg = ActionsCfg()
    robot_base_offset = {"pos": [1.2, -0.8, 0.92], "rot": [0.0, 0.0, 0.0]}
    observation_cameras: dict = {
        "global_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Base/global_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.25, -0.55, 0.27), rot=(0.74698, 0.54011, 0.23182, 0.31074), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=40.6,
                    focus_distance=400.0,
                    horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
                    clipping_range=(0.01, 3.0),
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
        # post init of parent
        super().__post_init__()

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Base",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Fixed_Jaw_tip",  # 夹爪
                    name="tool_gripper",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Moving_Jaw_tip",  # 夹爪
                    name="tool_jaw",
                ),
            ],
        )
        base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/Base",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )
        gripper_table_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/Fixed_Jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/counter_1_front_group/top_geometry"],
        )
        jaw_table_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/Moving_Jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/counter_1_front_group/top_geometry"],
        )
        gripper_object_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/Fixed_Jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/object/BuildingBlock003"],
        )
        jaw_object_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/Moving_Jaw",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/object/BuildingBlock003"],
        )
        setattr(self.scene, "base_contact", base_contact)
        setattr(self.scene, "gripper_table_contact", gripper_table_contact)
        setattr(self.scene, "jaw_table_contact", jaw_table_contact)
        setattr(self.scene, "gripper_object_contact", gripper_object_contact)
        setattr(self.scene, "jaw_object_contact", jaw_object_contact)

        self.viewport_cfg = {
            "offset": [-1.0, 0.0, 2.0],
            "lookat": [1.0, 0.0, -0.7]
        }

        self.set_reward_gripper_joint_names(["gripper"])


@configclass
class AbsJointActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


class LERobotAbsJointGripperEnvRLCfg(LERobotEnvRLCfg):
    robot_cfg: ArticulationCfg = SO101_FOLLOWER_CFG
    robot_name: str = "LeRobot-AbsJointGripper-RL"
    actions: AbsJointActionsCfg = AbsJointActionsCfg()
    robot_base_offset = {"pos": [1.2, -0.8, 0.892], "rot": [0.0, 0.0, 0]}
    observation_cameras: dict = {
        "hand_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(-0.001, 0.1, -0.04), rot=(-0.404379, -0.912179, -0.0451242, 0.0486914), convention="ros"),  # wxyz
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=36.5,
                    focus_distance=400.0,
                    horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),
                    lock_camera=True
                ),
                width=480,
                height=480,
                update_period=0.05,
            ),
            "tags": ["rl", "teleop"]
        },
        "global_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/camera_base/global_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.25, -0.55, 0.27), rot=(0.74698, 0.54011, 0.23182, 0.31074), convention="opengl"),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=40.6,
                    focus_distance=400.0,
                    horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
                    clipping_range=(0.01, 3.0),
                    lock_camera=True
                ),
                width=480,
                height=480,
                update_period=0.05,
            ),
            "tags": ["rl", "teleop"]
        }
    }

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow_flex", "wrist.*"],
            scale=1,
            use_default_offset=False
        )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1,
            use_default_offset=False
        )

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        if action.get("so101_leader") is not None:
            processed_action = convert_action_from_so101_leader(action["joint_state"], action["motor_limits"], device)
            return processed_action
        else:
            raise ValueError("only support so101_leader action")
