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
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.pose import Pose

import lw_benchhub.core.mdp as mdp
import lw_benchhub.core.mdp as lw_benchhub_mdp
from lw_benchhub.core.mdp.actions.decoupled_wbc_action import G1DecoupledWBCActionCfg
from lw_benchhub.core.mdp.actions.g1_action import G1ActionCfg
from lw_benchhub.core.models.grippers.dex3 import Dex3GripperCfg
from lw_benchhub.core.robots.robot_arena_base import (
    EmbodimentBaseObservationCfg,
    EmbodimentBasePolicyObservationCfg,
    EmbodimentGeneralObsCfg,
    LwEmbodimentBase,
)
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.math_utils import transform_utils as T

from .assets_cfg import G1_GEARWBC_CFG, G1_HIGH_PD_CFG, G1_Loco_CFG, OFFSET_CONFIG_G1
from .common import DecoupledWBCActionsCfg, LocoActionsCfg, PinkActionsCfg, RLActionsCfg

##
# Pre-defined configs
##
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class G1SceneCfg:
    robot: ArticulationCfg = G1_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
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


@configclass
class G1HandRLSceneCfg(G1SceneCfg):
    ee_frame = FrameTransformerCfg(
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


@configclass
class G1PinkActionsCfg(PinkActionsCfg):
    base_action: mdp.RelativeJointPositionActionCfg = mdp.RelativeJointPositionActionCfg(
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

    arms_action: G1ActionCfg = G1ActionCfg(asset_name="robot", joint_names=[
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint"
    ])


@configclass
class G1LocoActionsCfg(LocoActionsCfg):
    base_action = mdp.LegPositionActionCfg(
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

    left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["left_shoulder.*", "left_wrist.*", "left_elbow.*"],  # TODO
        body_name="left_wrist_yaw_link",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0415, 0.003, 0.0)),
    )

    right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"],  # TODO
        body_name="right_wrist_yaw_link",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0415, -0.003, 0.0)),
    )


@configclass
class G1RLActionsCfg(RLActionsCfg):
    right_arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["right_shoulder.*", "right_wrist.*", "right_elbow.*"], scale=1, use_default_offset=True
    )


@configclass
class G1DecoupledWBCActionsCfg(DecoupledWBCActionsCfg):
    base_action: G1DecoupledWBCActionCfg = G1DecoupledWBCActionCfg(asset_name="robot", joint_names=[".*"])
    left_hand_action: mdp.ActionTermCfg = None
    right_hand_action: mdp.ActionTermCfg = None
    left_arm_action: mdp.ActionTermCfg = None
    right_arm_action: mdp.ActionTermCfg = None


@configclass
class G1CameraCfg:
    left_hand_camera: TiledCameraCfg = None
    first_person_camera: TiledCameraCfg = None
    right_hand_camera: TiledCameraCfg = None
    left_shoulder_camera: TiledCameraCfg = None
    eye_in_hand_camera: TiledCameraCfg = None
    right_shoulder_camera: TiledCameraCfg = None
    hand_camera: TiledCameraCfg = None
    global_camera: TiledCameraCfg = None
    d435_camera: TiledCameraCfg = None


@configclass
class G1ObservationsCfg(EmbodimentBaseObservationCfg):
    """Observation specifications for the MDP."""

    @configclass
    class G1GeneralObsCfg(EmbodimentGeneralObsCfg):
        gripper_pos: ObsTerm = ObsTerm(func=lw_benchhub_mdp.gripper_pos)

    embodiment_general_obs: G1GeneralObsCfg = G1GeneralObsCfg()


@configclass
class G1PolicyObservationsCfg(EmbodimentBasePolicyObservationCfg):
    """Observations for policy group with state values."""
    pass


class UnitreeG1EnvCfg(LwEmbodimentBase):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1"
        self.gripper_cfg = Dex3GripperCfg(
            "unitree_dex3_left",
            "unitree_dex3_right"
        )
        self.observation_config = G1ObservationsCfg()
        self.policy_observation_config = G1PolicyObservationsCfg()
        self.action_config = G1PinkActionsCfg()
        self.scene_config = G1SceneCfg()
        self.robot_scale = self.context.robot_scale
        self.scene_config.robot.spawn.scale = (self.robot_scale, self.robot_scale, self.robot_scale)
        self.scene_config.left_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{self.gripper_cfg.left_contact_body_name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        self.scene_config.right_gripper_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{self.gripper_cfg.right_contact_body_name}",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )
        self.camera_config = G1CameraCfg()
        self.robot_base_link: str = "pelvis"
        self.hand_action_mode: str | None = None
        self.robot_to_fixture_dist: float = 0.50
        self.robot_base_offset = {"pos": [0.0, 0.0, 0.8], "rot": [0.0, 0.0, 0.0]}
        self.observation_cameras: dict = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, 0.02857542, 0.42446595), rot=(0.64367383, 0.26523914, -0.27106013, -0.66472446), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=62,
                        vertical_aperture=39.8,  # Approx for fovy ~79.5 deg with f=24.0
                        clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": ["product"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE],
            },
            "first_person_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/first_person_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, -0.00937542, 0.42446595), rot=(0.64367383, 0.26523914, -0.27106013, -0.66472446), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=27.6,
                        vertical_aperture=31.9,  # Approx for fovy ~79.5 deg with f=19.3
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": ["product"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS]
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/right_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, -0.04657542, 0.42446595), rot=(0.64367383, 0.26523914, -0.27106013, -0.66472446), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=62,
                        vertical_aperture=39.8,  # Approx for fovy ~79.5 deg with f=24.0
                        clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": ["product"],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE],
            },
            "left_shoulder_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/left_shoulder_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.1, 0.4, 0.5), rot=(0.28818, 0.25453, -0.33065, -0.86188), convention="opengl"),  # -35, -30, -150
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "eye_in_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/eye_in_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.3, 0, 0.4), rot=(0.70442, 0.06163, -0.06163, -0.70442), convention="opengl"),  # 0, -10, -90
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "right_shoulder_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/right_shoulder_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.1, -0.4, 0.5), rot=(0.71134, 0.39777, -0.12609, -0.56558), convention="opengl"),  # 40, -25, -32
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
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL]
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL],
            },
            "d435_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/d435_link/d435_camera",
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
                "tags": [],
                "execute_mode": [ExecuteMode.TRAIN, ExecuteMode.EVAL],
            },
        }
        self.offset_config = OFFSET_CONFIG_G1


class UnitreeG1LocoEnvCfg(UnitreeG1EnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-Loco"
        self.action_config = G1LocoActionsCfg()
        self.scene_config.robot = G1_Loco_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.robot_base_offset = {"pos": [0.0, 0.0, 0.83], "rot": [0.0, 0.0, 0.0]}


class UnitreeG1HandEnvCfg(UnitreeG1EnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-Hand"
        self.hand_action_mode = "tracking"
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]

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
        if self.action_config.left_arm_action.controller.use_relative_mode:  # Relative mode
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


class UnitreeG1ControllerEnvCfg(UnitreeG1EnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-Controller"
        self.hand_action_mode = "handle"
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]
        # base auto-lock state
        self.base_lock_state = False
        self.base_lock_value_x = 0.0
        self.base_lock_value_y = 0.0
        self.base_lock_value_yaw = 0.0
        # base joint indices cache
        self.base_x_joint_index = -1
        self.base_y_joint_index = -1
        self.base_yaw_joint_index = -1
        # base auto-lock thresholds
        self.base_auto_lock_threshold = 0.1  # if input magnitude is less than this value, auto-lock base joints
        self.base_auto_unlock_threshold = 0.3  # if input magnitude is greater than this value, auto-unlock base joints

        # PID configurations for base control
        self.pid_configs = {
            'base_x': {
                'kp': 100.0,
                'kd': 2.0,
                'ki': 0.2,
                'deadzone': 0.0001,
                'prev_error': 0.0,
                'integral_error': 0.0
            },
            'base_y': {
                'kp': 25.0,
                'kd': 2.0,
                'ki': 0.2,
                'deadzone': 0.0001,
                'prev_error': 0.0,
                'integral_error': 0.0
            },
            'base_yaw': {
                'kp': 25.0,
                'kd': 2.0,
                'ki': 0.2,
                'deadzone': 0.0001,
                'prev_error': 0.0,
                'integral_error': 0.0
            }
        }

    def get_joint_index(self, device, joint_name_pattern):
        """
        find joint index by joint name pattern

        Args:
            device: device object
            joint_name_pattern: joint name pattern, like "base_x", "base_y"

        Returns:
            int: joint index, if not found, return -1
        """
        joint_names = device.robot.data.joint_names
        for i, name in enumerate(joint_names):
            if joint_name_pattern in name:
                return i
        return -1

    def init_base_joint_indices(self, device):
        """
        initialize base joint indices cache

        Args:
            device: device object
        """
        self.base_x_joint_index = self.get_joint_index(device, "base_x")
        self.base_y_joint_index = self.get_joint_index(device, "base_y")
        self.base_yaw_joint_index = self.get_joint_index(device, "base_yaw")

        if (self.base_x_joint_index == -1 or
            self.base_y_joint_index == -1 or
                self.base_yaw_joint_index == -1):
            print(f"Warning: Could not find base joint indices - "
                  f"base_x: {self.base_x_joint_index}, "
                  f"base_y: {self.base_y_joint_index}, "
                  f"base_yaw: {self.base_yaw_joint_index}")
        else:
            print(f"Base joint indices initialized - "
                  f"base_x: {self.base_x_joint_index}, "
                  f"base_y: {self.base_y_joint_index}, "
                  f"base_yaw: {self.base_yaw_joint_index}")

    def get_default_action(self, device, num_envs) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)

    def reset_robot_cfg_state(self):
        self.base_lock_state = False
        self.base_lock_value_x = 0.0
        self.base_lock_value_y = 0.0
        self.base_lock_value_yaw = 0.0
        self.base_x_joint_index = -1
        self.base_y_joint_index = -1
        self.base_yaw_joint_index = -1
        self.base_auto_lock_threshold = 0.05
        self.base_auto_unlock_threshold = 0.3
        for axis_name in ['base_x', 'base_y', 'base_yaw']:
            if axis_name in self.pid_configs:
                self.pid_configs[axis_name]['prev_error'] = 0.0
                self.pid_configs[axis_name]['integral_error'] = 0.0

    def pid_control(self, target_value, current_value, axis_name):
        """
        PID controller function

        Args:
            target_value: target value
            current_value: current value
            axis_name: axis name ('base_x', 'base_y', 'base_yaw')

        Returns:
            float: control output
        """
        config = self.pid_configs[axis_name]
        error = target_value - current_value

        if abs(error) < config['deadzone']:
            return 0.0

        p_term = config['kp'] * error

        config['integral_error'] += error
        i_term = config['ki'] * config['integral_error']

        d_term = config['kd'] * (error - config['prev_error'])

        control_output = p_term + i_term + d_term

        if abs(config['integral_error']) > 1.0:
            config['integral_error'] = 1.0 if config['integral_error'] > 0 else -1.0

        config['prev_error'] = error

        return control_output

    def reset_pid_state(self, axis_name):
        """
        reset pid state

        Args:
            axis_name: axis name ('base_x', 'base_y', 'base_yaw')
        """
        config = self.pid_configs[axis_name]
        config['prev_error'] = 0.0
        config['integral_error'] = 0.0

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        # Get base input values
        base_input_x = abs(action.get('rbase', [0, 0])[0])
        base_input_y = abs(action.get('rbase', [0, 0])[1])
        base_input_magnitude = max(base_input_x, base_input_y)

        x_joint_vel = device.robot.data.joint_vel[0, self.base_x_joint_index].item()
        y_joint_vel = device.robot.data.joint_vel[0, self.base_y_joint_index].item()
        yaw_joint_vel = device.robot.data.joint_vel[0, self.base_yaw_joint_index].item()

        # Auto-lock logic: lock when input magnitude is small
        if not self.base_lock_state and base_input_magnitude < self.base_auto_lock_threshold and abs(x_joint_vel) < 0.01 and abs(y_joint_vel) < 0.01 and abs(yaw_joint_vel) < 0.01:
            # Initialize joint indices if needed
            if (self.base_x_joint_index == -1 or
                self.base_y_joint_index == -1 or
                    self.base_yaw_joint_index == -1):
                self.init_base_joint_indices(device)

            if (self.base_x_joint_index != -1 and
                self.base_y_joint_index != -1 and
                    self.base_yaw_joint_index != -1):
                self.base_lock_state = True

                # Use joint positions for locking
                self.base_lock_value_x = device.robot.data.joint_pos[0, self.base_x_joint_index].item()
                self.base_lock_value_y = device.robot.data.joint_pos[0, self.base_y_joint_index].item()
                self.base_lock_value_yaw = device.robot.data.joint_pos[0, self.base_yaw_joint_index].item()

                # TODO: use joint pos target instead of joint pos for locking
                # x_target = device.robot.data.joint_pos_target[0, self.base_x_joint_index].item()
                # y_target = device.robot.data.joint_pos_target[0, self.base_y_joint_index].item()
                # yaw_target = device.robot.data.joint_pos_target[0, self.base_yaw_joint_index].item()

                self.reset_pid_state('base_x')
                self.reset_pid_state('base_y')
                self.reset_pid_state('base_yaw')
                print(f"Auto-locked base joints base_x({self.base_x_joint_index}), "
                      f"base_y({self.base_y_joint_index}), "
                      f"base_yaw({self.base_yaw_joint_index}) to position: "
                      f"x={self.base_lock_value_x:.3f}, "
                      f"y={self.base_lock_value_y:.3f}, "
                      f"yaw={self.base_lock_value_yaw:.3f}")
            else:
                print("Warning: Could not find base joint indices for base_x, base_y, or base_yaw")

        # Auto-unlock logic: unlock when input magnitude is large
        elif self.base_lock_state and base_input_magnitude > self.base_auto_unlock_threshold:
            self.base_lock_state = False

            self.reset_pid_state('base_x')
            self.reset_pid_state('base_y')
            self.reset_pid_state('base_yaw')
            print("Auto-unlocked base due to large input")

        base_action = torch.zeros(3,).to(device=action['rbase'].device)

        if self.base_lock_state:
            if (self.base_x_joint_index != -1 and
                self.base_y_joint_index != -1 and
                    self.base_yaw_joint_index != -1):
                # Use joint positions for PID control
                current_x = device.robot.data.joint_pos[0, self.base_x_joint_index].item()
                current_y = device.robot.data.joint_pos[0, self.base_y_joint_index].item()
                current_yaw = device.robot.data.joint_pos[0, self.base_yaw_joint_index].item()

                control_output_x = self.pid_control(self.base_lock_value_x, current_x, 'base_x')
                control_output_y = self.pid_control(self.base_lock_value_y, current_y, 'base_y')
                control_output_yaw = self.pid_control(self.base_lock_value_yaw, current_yaw, 'base_yaw')

                base_action[0] = control_output_x
                base_action[1] = control_output_y
                base_action[2] = control_output_yaw
            else:
                # Fallback to default behavior if joint indices not found
                if action['rsqueeze'] > 0.5:
                    base_action = torch.tensor([action['rbase'][0], 0, action['rbase'][1]],
                                               device=action['rbase'].device)
                else:
                    base_action = torch.tensor([action['rbase'][0], action['rbase'][1], 0],
                                               device=action['rbase'].device)
        else:
            # Free state: normal input control with speed reduction
            non_lock_scale = 0.5
            if action['rsqueeze'] > 0.5:
                base_action = torch.tensor([action['rbase'][0] * non_lock_scale, 0, action['rbase'][1] * non_lock_scale],
                                           device=action['rbase'].device)
            else:
                base_action = torch.tensor([action['rbase'][0] * non_lock_scale, action['rbase'][1] * non_lock_scale, 0],
                                           device=action['rbase'].device)

        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 4],
                                                                           device.robot.data.body_link_quat_w[0, 4])
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
        if hasattr(self.action_config, 'left_arm_action') and hasattr(self.action_config, 'right_arm_action'):
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
        else:
            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                pose_quat = abs_arm[3:7]
                combined_quat = pose_quat
                arm_action = abs_arm.clone()
                rot_mat = T.quat2mat(base_quat)
                gripper_movement = arm_action[:3]
                pose_movement = gripper_movement
                arm_action[:3] = pose_movement
                arm_action[3] = combined_quat[3]  # Update quaternion from xyzw to wxyz
                arm_action[4:7] = combined_quat[:3]
                if arm_idx == 0:
                    left_arm_action = arm_action  # Robot frame
                else:
                    right_arm_action = arm_action  # Robot frame
        left_gripper = torch.tensor([action["left_gripper"]], device=action['rbase'].device)
        right_gripper = torch.tensor([action["right_gripper"]], device=action['rbase'].device)
        return torch.concat([base_action, left_arm_action, right_arm_action,
                             left_gripper, right_gripper]).unsqueeze(0)


class UnitreeG1LocoHandEnvCfg(UnitreeG1LocoEnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-Loco-Hand"
        self.hand_action_mode = "tracking"
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        base_action = action["base"]
        base_action = torch.cat([base_action, torch.tensor([action["base_loco_mode"]], dtype=action["base"].dtype, device=action["base"].device)], dim=0)
        left_arm_action = None
        right_arm_action = None
        if self.action_config.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
        else:  # Absolute mode
            _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                               device.robot.data.root_com_quat_w[0],
                                                                               device.robot.data.body_link_pos_w[0, 4],
                                                                               device.robot.data.body_link_quat_w[0, 4])
            # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
            base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
            base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
            base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                pose_quat = abs_arm[3:7]
                combined_quat = T.quat_multiply(base_quat, pose_quat)
                arm_action = abs_arm.clone()
                rot_mat = T.quat2mat(base_quat)
                gripper_movement = torch.matmul(rot_mat, arm_action[:3])
                pose_movement = base_movement + gripper_movement
                arm_action[:3] = pose_movement
                arm_action[3] = combined_quat[3]  # Update rotation quaternion from xyzw to wxyz
                arm_action[4:7] = combined_quat[:3]
                if arm_idx == 0:
                    left_arm_action = arm_action  # Robot frame
                else:
                    right_arm_action = arm_action  # Robot frame
        left_finger_tips = action["left_finger_tips"][[0, 1, 2]].flatten()
        right_finger_tips = action["right_finger_tips"][[0, 1, 2]].flatten()
        return torch.concat([left_arm_action, right_arm_action,
                             left_finger_tips, right_finger_tips, base_action]).unsqueeze(0)


class UnitreeG1LocoControllerEnvCfg(UnitreeG1LocoEnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-Loco-Controller"
        self.hand_action_mode = "handle"
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        base_action = torch.zeros(3,)

        if action['rsqueeze'] > 0.5:
            base_action = torch.tensor([action['rbase'][0], 0, action['rbase'][1]], device=action['rbase'].device)
        else:
            base_action = torch.tensor([action['rbase'][0], action['rbase'][1], 0,], device=action['rbase'].device)
        base_action = torch.cat([base_action, torch.tensor([action["base_mode"]], dtype=action["rbase"].dtype, device=action["rbase"].device)], dim=0)

        left_arm_action = None
        right_arm_action = None
        if self.action_config.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
        else:  # Absolute mode
            _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                               device.robot.data.root_com_quat_w[0],
                                                                               device.robot.data.body_link_pos_w[0, 4],
                                                                               device.robot.data.body_link_quat_w[0, 4])
            # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
            base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
            base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
            base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

            cos_yaw = torch.cos(torch.tensor(base_yaw, device=device.env.device))
            sin_yaw = torch.sin(torch.tensor(base_yaw, device=device.env.device))
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

            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                pose_quat = abs_arm[3:7]
                combined_quat = T.quat_multiply(base_quat, pose_quat)
                arm_action = abs_arm.clone()
                rot_mat = T.quat2mat(base_quat)
                gripper_movement = torch.matmul(rot_mat, arm_action[:3])
                pose_movement = base_movement + gripper_movement
                arm_action[:3] = pose_movement
                arm_action[3] = combined_quat[3]  # Update rotation quaternion from xyzw to wxyz
                arm_action[4:7] = combined_quat[:3]
                if arm_idx == 0:
                    left_arm_action = arm_action  # Robot coordinate system
                else:
                    right_arm_action = arm_action  # Robot coordinate system
        left_gripper = torch.tensor([-action["left_gripper"]], device=action['rbase'].device)
        right_gripper = torch.tensor([-action["right_gripper"]], device=action['rbase'].device)
        return torch.concat([left_arm_action, right_arm_action,
                             left_gripper, right_gripper, base_action]).unsqueeze(0)


class UnitreeG1HandEnvRLCfg(UnitreeG1HandEnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-RL"
        self.hand_action_mode = "rl"
        self.eef_link_name = "right_wrist_yaw_link"
        self.scene_config = G1HandRLSceneCfg()
        self.action_config = G1RLActionsCfg()
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]
        self.robot_to_fixture_dist = 0.20
        self.robot_base_offset = {"pos": [-0.2, 0.1, 0.92], "rot": [0.0, 0.0, 0.0]}
        self.reward_gripper_joint_names = ["right_hand_.*"]
        self.reward_arm_joint_names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                                       "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]


class UnitreeG1FullHandRLEnvRLCfg(UnitreeG1HandEnvRLCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-FullHand"
        self.action_config.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["left_shoulder.*", "left_wrist.*", "left_elbow.*", "left_hand.*"], scale=1, use_default_offset=True
        )
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.robot_to_fixture_dist = 0.20
        self.robot_base_offset = {"pos": [-0.25, 0.3, 0.95], "rot": [0.0, 0.0, 0.0]}


class UnitreeG1ControllerDecoupledWBCEnvCfg(UnitreeG1ControllerEnvCfg):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "G1-Controller-DecoupledWBC"
        self.scene_config.robot = G1_GEARWBC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = G1DecoupledWBCActionsCfg()
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]
        self.init_robot_base_height = 0.75
        self.init_waist_yaw = 0.0
        self.init_waist_pitch = 0.0
        self.init_waist_roll = 0.0

        self.pitch_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate
        self.yaw_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate
        self.roll_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate
        self.height_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.sim.dt = 1 / 200  # physics frequency: 100Hz
        env_cfg.decimation = 4  # action frequency: 50Hz
        return env_cfg

    def reset_robot_cfg_state(self):
        super().reset_robot_cfg_state()
        self.init_robot_base_height = 0.75
        self.init_waist_yaw = 0.0
        self.init_waist_pitch = 0.0
        self.init_waist_roll = 0.0
        self.pitch_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate
        self.yaw_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate
        self.roll_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate
        self.height_dead_zone_state = 0  # 0: normal, 1: dead zone waiting, 2: dead zone ready to activate

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        """
        Base action [lin_x_local, lin_y_local, ang_z_local, base_height_cmd, torso_orientation_roll_cmd, torso_orientation_pitch_cmd, torso_orientation_yaw_cmd]
        range of -1, 1 each
        """
        base_action = torch.zeros(7,)
        # default as standing
        base_action[3] = self.init_robot_base_height
        base_action[4:] = torch.tensor([self.init_waist_roll, self.init_waist_pitch, self.init_waist_yaw], device=action['lbase'].device)
        # Left squeeze released: Moving-base related cmds
        # left joystick up/down to control pitch movement of the base

        if action['lbase'][0] > 0.5 or action['lbase'][0] < -0.5:
            action['lbase'][1] = 0
        if action['lbase'][1] > 0.5 or action['lbase'][1] < -0.5:
            action['lbase'][0] = 0

        joystick_input = action['lbase'][0].item()
        target_pitch, self.pitch_dead_zone_state = self._apply_dead_zone_control(
            joystick_input, self.init_waist_pitch, self.pitch_dead_zone_state, dead_zone_threshold=0.1, activation_threshold=0.5, scale_factor=0.01
        )
        base_action[5] = torch.clamp(torch.tensor(target_pitch), min=-1.5, max=1.5)
        self.init_waist_pitch = base_action[5].item()

        if action['lsqueeze'] <= 0.5:
            if action['rsqueeze'] > 0.5:
                # right joystick up/down to control linear x, left/right to control turning yaw
                base_action[:3] = torch.tensor([action['rbase'][0], 0, action['rbase'][1]], device=action['rbase'].device)
            else:
                # right joystick up/down to control linear x, left/right to control linear y
                base_action[:3] = torch.tensor([action['rbase'][0], action['rbase'][1], 0,], device=action['rbase'].device)
            # left joystick left/right to control yaw, up/down to control pitch movement of the base
            # base_action[4:] = torch.tensor([0, action['lbase'][0], action['lbase'][1]], device=action['lbase'].device)
            # left joystick left/right to control yaw movement of the base
            yaw_joystick_input = action['lbase'][1].item()
            target_yaw, self.yaw_dead_zone_state = self._apply_dead_zone_control(
                yaw_joystick_input, self.init_waist_yaw, self.yaw_dead_zone_state,
                dead_zone_threshold=0.1, activation_threshold=0.5, scale_factor=0.005
            )
            base_action[6] = torch.clamp(torch.tensor(target_yaw), min=-2, max=2)
            self.init_waist_yaw = base_action[6].item()
        # Left squeeze pressed
        else:
            # left joystick left/right to control roll, up/down to control pitch movement of the base
            # base_action[4:] = torch.tensor([-1 * action['lbase'][1], action['lbase'][0], 0], device=action['lbase'].device)
            # left joystick left/right to control roll movement of the base
            roll_joystick_input = -action['lbase'][1].item()
            target_roll, self.roll_dead_zone_state = self._apply_dead_zone_control(
                roll_joystick_input, self.init_waist_roll, self.roll_dead_zone_state,
                dead_zone_threshold=0.1, activation_threshold=0.5, scale_factor=0.02
            )
            base_action[4] = torch.clamp(torch.tensor(target_roll), min=-0.5, max=0.5)
            self.init_waist_roll = base_action[4].item()

            # Left squeeze pressed and Right squeeze pressed: right joystick up/down to control base_height_cmd (relative to current height)
            if action['rsqueeze'] > 0.5:
                # joystaick returns -1 to 1, remap to a smaller range
                height_joystick_input = action['rbase'][0].item()
                target_height, self.height_dead_zone_state = self._apply_dead_zone_control(
                    height_joystick_input, self.init_robot_base_height, self.height_dead_zone_state,
                    dead_zone_threshold=0.1, activation_threshold=0.5, scale_factor=0.01
                )
                base_action[3] = torch.clamp(torch.tensor(target_height), min=0.2, max=0.75)
                self.init_robot_base_height = base_action[3].item()

        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 3],
                                                                           device.robot.data.body_link_quat_w[0, 3])
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

        left_arm_action = None
        right_arm_action = None

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
        left_gripper = torch.tensor([action["left_gripper"]], device=action['rbase'].device)
        right_gripper = torch.tensor([action["right_gripper"]], device=action['rbase'].device)
        result = torch.concat([left_gripper, right_gripper, left_arm_action, right_arm_action, base_action]).unsqueeze(0)
        return result

    def get_default_action(self, device, num_envs) -> torch.Tensor:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device).repeat(num_envs, 1)

    def _apply_dead_zone_control(self, joystick_input, current_angle, dead_zone_state, dead_zone_threshold=0.1, activation_threshold=0.5, scale_factor=0.02):
        """
        Apply dead zone control logic for pitch angle control.

        Logic:
        1. When angle crosses zero (from negative to 0 or positive to 0), enter dead zone state
        2. In dead zone state, no control signals are accepted
        3. To exit dead zone: joystick must first enter (-dead_zone_threshold, dead_zone_threshold), then go beyond ±activation_threshold

        Args:
            joystick_input (float): Joystick input value (-1 to 1)
            current_angle (float): Current pitch angle
            dead_zone_state (int): Dead zone state (0: normal, 1: dead zone waiting, 2: dead zone ready to activate)
            dead_zone_threshold (float): Dead zone threshold for joystick input (default: 0.1)
            activation_threshold (float): Activation threshold for exiting dead zone (default: 0.5)
            scale_factor (float): Scale factor for joystick input (default: 0.02)

        Returns:
            tuple: (target_pitch, new_dead_zone_state)
        """
        # Calculate target angle
        target_pitch = current_angle + scale_factor * joystick_input
        new_dead_zone_state = dead_zone_state

        # Check if we just reached 0 from either direction (crossed zero)
        crossed_zero_from_negative = current_angle < 0 and target_pitch >= 0
        crossed_zero_from_positive = current_angle > 0 and target_pitch <= 0
        if crossed_zero_from_negative or crossed_zero_from_positive:
            # Just reached 0, enter dead zone state
            target_pitch = 0.0
            new_dead_zone_state = 1  # Enter dead zone waiting state
            return target_pitch, new_dead_zone_state

        # Handle different dead zone states
        if dead_zone_state == 0:
            # Normal state - allow normal control
            if current_angle < 0:
                # When current angle < 0, must return to 0 first
                if joystick_input > 0:
                    target_pitch = min(target_pitch, 0.0)
                else:
                    target_pitch = current_angle + scale_factor * joystick_input
            elif current_angle > 0:
                # When current angle > 0, must return to 0 first
                if joystick_input < 0:
                    target_pitch = max(target_pitch, 0.0)
                else:
                    target_pitch = current_angle + scale_factor * joystick_input
            else:
                # At 0, normal accumulation
                target_pitch = current_angle + scale_factor * joystick_input

        elif dead_zone_state == 1:
            # Dead zone waiting state - check if joystick enters dead zone
            if abs(joystick_input) <= dead_zone_threshold:
                # Entered dead zone, move to ready state
                new_dead_zone_state = 2
                target_pitch = current_angle  # No movement
            else:
                # Still outside dead zone, maintain current angle
                target_pitch = current_angle

        elif dead_zone_state == 2:
            # Dead zone ready state - check if joystick exceeds activation threshold
            if abs(joystick_input) >= activation_threshold:
                # Exceeded activation threshold, exit dead zone and resume normal control
                new_dead_zone_state = 0
                target_pitch = current_angle + scale_factor * joystick_input
            else:
                # Still in dead zone, maintain current angle
                target_pitch = current_angle

        return target_pitch, new_dead_zone_state
