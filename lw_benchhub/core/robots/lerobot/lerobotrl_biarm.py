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

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

import lw_benchhub.core.mdp as mdp

##
# Pre-defined configs
##
from .assets_cfg import BI_SO101_FOLLOWER_CFG  # isort: skip
from lw_benchhub.utils.lerobot_utils import convert_action_from_so101_leader
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lw_benchhub.core.robots.robot_arena_base import LwEmbodimentBase
from isaaclab_arena.utils.pose import Pose
from lw_benchhub.utils.env import ExecuteMode

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class LeRobotBiArmSceneCfg:
    robot: ArticulationCfg = BI_SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left/gripper",
                name="left_tool_gripper",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.08),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left/wrist",
                name="left_ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.18),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left/upper_arm",
                name="left_tool_upperarm",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left/lower_arm",
                name="left_tool_lowerarm",
            ),

            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left/jaw",
                name="left_tool_jaw",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right/gripper",
                name="right_tool_gripper",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, -0.08),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right/wrist",
                name="right_ee_tcp",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.18),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right/upper_arm",
                name="right_tool_upperarm",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right/lower_arm",
                name="right_tool_lowerarm",
            ),

            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right/jaw",
                name="right_tool_jaw",
            )
        ],
    )
    base_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/left/base",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
    )
    left_gripper_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/left/gripper",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )
    right_gripper_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/right/gripper",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )


@configclass
class LeRobotBiArmCameraCfg:
    left_hand_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None
    global_camera: TiledCameraCfg = None


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder.*", "left_elbow_flex", "left_wrist.*"],
        scale=1,
        use_default_offset=True
    )
    left_gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_gripper"],
        scale=1,
        use_default_offset=True
    )
    right_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder.*", "right_elbow_flex", "right_wrist.*"],
        scale=1,
        use_default_offset=True
    )
    right_gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_gripper"],
        scale=1,
        use_default_offset=True
    )


class LeRobotBiArmRL(LwEmbodimentBase):
    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "LeRobot-BiARM-RL"
        self.camera_config = LeRobotBiArmCameraCfg()
        self.scene_config = LeRobotBiArmSceneCfg()
        self.action_cfg = ActionsCfg()
        self.robot_scale = self.context.robot_scale
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.robot_base_offset = {"pos": [-0.2, 0.5, 0.9], "rot": [0.0, 0.0, 0.0]}
        self.observation_cameras: dict = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left/gripper/wrist_camera",
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL]
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right/gripper/wrist_camera",
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL]
            },
            "global_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/root_link/global_camera",  # 0 -0.5 0.5 (0.1650476, -0.9862856, 0.0, 0.0) (-161,0,0)
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL]
            }
        }
        self.viewport_cfg = {
            "offset": [-1.0, 0.0, 2.0],
            "lookat": [1.0, 0.0, -0.7]
        }

        self.reward_gripper_joint_names(["gripper"])

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        if action.get("bi_so101_leader") is not None:
            processed_action = torch.zeros(device.env.num_envs, 12, device=device.env.device)
            processed_action[:, :6] = convert_action_from_so101_leader(action['joint_state']['left_arm'], action['motor_limits']['left_arm'], device)
            processed_action[:, 6:] = convert_action_from_so101_leader(action['joint_state']['right_arm'], action['motor_limits']['right_arm'], device)
            return processed_action
        else:
            raise ValueError("only support so101_leader action")
