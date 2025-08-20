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


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


# @configclass
class LERobotEnvRLCfg(BaseRobotCfg):
    robot_cfg: ArticulationCfg = SO101_FOLLOWER_CFG
    robot_name: str = "LeRobot-RL"
    robot_scale: float = 1.0
    actions: ActionsCfg = ActionsCfg()
    robot_base_offset = {"pos": [-0.2, 0.5, 0.9], "rot": [0.0, 0.0, 0.0]}
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
            "tags": ["rl"]
        },
        "global_camera": {
            "camera_cfg": TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base/global_camera",  # 0 -0.5 0.5 (0.1650476, -0.9862856, 0.0, 0.0) (-161,0,0)
                offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.5, 0.5), rot=(0.1650476, -0.9862856, 0.0, 0.0), convention="ros"),  # wxyz
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=28.7,
                    focus_distance=400.0,
                    horizontal_aperture=38.11,  # For a 78° FOV (assuming square image)
                    clipping_range=(0.01, 50.0),
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

        # Set actions for the specific robot type
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow_flex", "wrist.*"],
            scale=1,
            use_default_offset=True
        )

        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.0},
            close_command_expr={"gripper": 0.7},
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
                        pos=(0.0, 0.0, -0.08),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.18),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/upper_arm",
                    name="tool_upperarm",
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/lower_arm",
                    name="tool_lowerarm",
                ),

                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/jaw",  # jaw
                    name="tool_jaw",
                )
            ],
        )
        base_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/base",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
        )
        setattr(self.scene, "base_contact", base_contact)

        self.viewport_cfg = {
            "offset": [-1.0, 0.0, 2.0],
            "lookat": [1.0, 0.0, -0.7]
        }

        self.set_reward_gripper_joint_names(["gripper"])

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs

        # if self.actions.arm_action.controller.use_relative_mode:  # Relative mode
        #     arm_action = action["arm_delta"]
        # else:  # Absolute mode
        #     arm_action = action["arm_abs"]
        #     arm_action[3:] = arm_action[[6,3,4,5]]

        # arm_action = arm_action.repeat(num_envs, 1)
        # gripper = action["arm_gripper"] > 0
        # gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        # gripper_action[:] = -1.0 if gripper else 1.0
        # return torch.concat([arm_action, gripper_action], dim=1)


@configclass
class AbsJointActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.JointPositionActionCfg = MISSING


class LERobotAbsJointGripperEnvRLCfg(LERobotEnvRLCfg):
    robot_name: str = "LeRobot-AbsJointGripper-RL"
    actions: AbsJointActionsCfg = AbsJointActionsCfg()

    def __post_init__(self):
        super().__post_init__()
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
