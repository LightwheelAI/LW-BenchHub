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
from .assets_cfg import SO101_FOLLOWER_CFG, SO100_FOLLOWER_CFG  # isort: skip
from lw_benchhub.utils.lerobot_utils import convert_action_from_so101_leader
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from lw_benchhub.core.mdp.actions.joint_position_limit_action import JointPositionLimitActionCfg
from lw_benchhub.core.robots.robot_arena_base import LwEmbodimentBase
from isaaclab_arena.utils.pose import Pose
from lw_benchhub.utils.env import ExecuteMode

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class LeRobotSceneCfg:
    robot: ArticulationCfg = SO101_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class LeRobotRLSceneCfg(LeRobotSceneCfg):
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",  # gripper
                name="tool_gripper",
                offset=OffsetCfg(
                    pos=(-0.011, -0.0001, -0.0953),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/jaw",  # jaw
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


@configclass
class LeRobot100SceneCfg:
    robot: ArticulationCfg = SO100_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class LeRobot100RLSceneCfg(LeRobotRLSceneCfg):
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Base",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Fixed_Jaw_tip",  # gripper
                name="tool_gripper",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Moving_Jaw_tip",  # jaw
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


@configclass
class LeRobotCameraCfg:
    hand_camera: TiledCameraCfg = None
    global_camera: TiledCameraCfg = None


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = mdp.RelJointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder.*", "elbow_flex", "wrist.*", "gripper"],
        scale={"shoulder.*": 0.05, "elbow_flex": 0.05, "wrist.*": 0.05, "gripper": 0.2},
        use_zero_offset=True,
        clip={"shoulder.*": (-1.0, 1.0), "elbow_flex": (-1.0, 1.0), "wrist.*": (-1.0, 1.0), "gripper": (-1.0, 1.0)}
    )


@configclass
class AbsJointActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder.*", "elbow_flex", "wrist.*"],
        scale=1,
        use_default_offset=False
    )
    gripper_action: mdp.JointPositionActionCfg = JointPositionLimitActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        scale=1,
        use_default_offset=False
    )


class BaseLeRobot(LwEmbodimentBase):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "LeRobot"
        self.camera_config = LeRobotCameraCfg()
        self.scene_config = LeRobotSceneCfg()
        self.action_config = ActionsCfg()
        self.robot_scale = self.context.robot_scale
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.robot_base_offset = {"pos": [-0.8, -0.75, 0.9], "rot": [0.0, 0.0, torch.pi / 2]}
        self.observation_cameras: dict = {
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL]
            }
        }


class LeRobotRL(BaseLeRobot):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "LeRobot-RL"
        self.scene_config = LeRobotRLSceneCfg()
        self.robot_base_offset = {"pos": [1.2, -0.8, 0.892], "rot": [0.0, 0.0, 0]}
        self.viewport_cfg = {
            "offset": [-1.0, 0.0, 2.0],
            "lookat": [1.0, 0.0, -0.7]
        }
        self.reward_gripper_joint_names = ["gripper"]


class LeRobot100RL(LeRobotRL):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "LeRobot100-RL"
        self.scene_config = LeRobot100RLSceneCfg()
        self.action_config = ActionsCfg()
        self.robot_base_offset = {"pos": [1.2, -0.8, 0.92], "rot": [0.0, 0.0, 0.0]}
        self.observation_cameras: dict = {
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL]
            }
        }


class LeRobotAbsJointGripperRL(LeRobotRL):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "LeRobot-AbsJointGripper-RL"
        self.action_config = AbsJointActionsCfg()
        self.observation_cameras: dict = {
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL]
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.TRAIN, ExecuteMode.EVAL]
            }
        }

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        if action.get("so101_leader") is not None:
            processed_action = convert_action_from_so101_leader(action["joint_state"], action["motor_limits"], device)
            return processed_action
        else:
            raise ValueError("only support so101_leader action")
