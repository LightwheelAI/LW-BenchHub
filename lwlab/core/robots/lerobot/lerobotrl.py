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
from .assets_cfg import SO101_FOLLOWER_CFG  # isort: skip
from lwlab.utils.lerobot_utils import convert_action_from_so101_leader
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lwlab.utils.math_utils import transform_utils as T
import isaaclab.utils.math as math_utils
import torch

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


class BaseLERobotEnvCfg(BaseRobotCfg):
    robot_cfg: ArticulationCfg = SO101_FOLLOWER_CFG
    robot_name: str = "LeRobot-RL"
    robot_scale: float = 1.0
    observation_cameras: dict = {
        "global_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/global_camera",
                offset=TiledCameraCfg.OffsetCfg(pos=(0.2, -0.65, 0.3), rot=(0.8, 0.5, 0.16657, 0.2414), convention="opengl"),
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
        super().__post_init__()
        self.scene.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = MISSING


# @configclass
class LERobotEnvRLCfg(BaseLERobotEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    robot_base_offset = {"pos": [1.2, -0.8, 0.897], "rot": [0.0, 0.0, 0]}

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow_flex", "wrist.*", "gripper"],
            scale=1,
            use_zero_offset=True,
            clip={"shoulder.*": (-0.05, 0.05), "elbow_flex": (-0.05, 0.05), "wrist.*": (-0.05, 0.05), "gripper": (-0.2, 0.2)}
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


class LERobotAbsJointGripperEnvRLCfg(BaseLERobotEnvCfg):
    robot_name: str = "LeRobot-AbsJointGripper-RL"
    actions: AbsJointActionsCfg = AbsJointActionsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow_flex", "wrist.*"],
            scale=1,
            use_default_offset=True
        )

        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1,
            use_default_offset=True
        )

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        if action.get("so101_leader") is not None:
            processed_action = convert_action_from_so101_leader(action["joint_state"], action["motor_limits"], device)
            return processed_action
        else:
            raise ValueError("only support so101_leader action")
