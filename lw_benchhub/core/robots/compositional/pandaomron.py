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

from typing import Any, Optional

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.pose import Pose

import lw_benchhub.core.mdp as mdp
import lw_benchhub.core.mdp as lw_benchhub_mdp
from lw_benchhub.core.robots.compositional.assets_cfg import FRANKA_OMRON_HIGH_PD_CFG, OFFSET_CONFIG, VIS_HELPER_CFG  # isort: skip
from lw_benchhub.core.robots.robot_arena_base import (
    EmbodimentBaseObservationCfg,
    EmbodimentBasePolicyObservationCfg,
    EmbodimentGeneralObsCfg,
    LwEmbodimentBase,
)
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.math_utils import transform_utils as T

##
# Pre-defined configs
##
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


class PandaOmronEmbodiment(LwEmbodimentBase):

    name = "pandaomron"
    robot_base_link: str = "mobilebase0_wheeled_base"
    robot_vis_helper_cfg = VIS_HELPER_CFG

    def __init__(self, enable_cameras: bool = False, initial_pose: Optional[Pose] = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = PandaOmronSceneCfg()
        self.action_config = None
        self.observation_config = PandaOmronObservationsCfg()
        self.policy_observation_config = PandaOmronPolicyObservationsCfg()
        self.event_config = None
        self.mimic_env = None
        self.camera_config = PandaOmronCameraCfg()
        self.offset_config = OFFSET_CONFIG
        self.observation_cameras = {
            "agentview_left_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/omron_v2/mobilebase0_support/agentview_left",
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
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "agentview_right_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/omron_v2/mobilebase0_support/agentview_right",
                    offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, -0.35, 1.05), rot=(0.677509, 0.376787, -0.299353, -0.556239), convention="opengl"),
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
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE]
            },
            "eye_in_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_hand/eye_in_hand",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.05, 0, 0), rot=(0, 0.707107, 0.707107, 0), convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=36.83,  # For a 75° FOV (assuming square image)
                        clipping_range=(0.01, 50.0),  # Closer clipping for hand camera
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
                "tags": [],
                "execute_mode": [ExecuteMode.TELEOP, ExecuteMode.REPLAY_STATE, ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS]
            }
        }

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        # We override the default initial pose setting function in order to also set
        # the initial pose of the stand.
        scene_config = super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose)
        if scene_config is None or not hasattr(scene_config, "robot"):
            raise RuntimeError("scene_config must be populated with a `robot` before calling `set_robot_initial_pose`.")
        scene_config.stand.init_state.pos = pose.position_xyz
        scene_config.stand.init_state.rot = pose.rotation_wxyz
        return scene_config

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs
        base = torch.zeros(num_envs, 4, device=device.env.device)
        if action.get('spacemouse') is not None:
            base[:, :3] = torch.tensor([action["rbase"][0], action["rbase"][1], action["rbase"][2]], device=device.env.device)
        else:
            if action['rsqueeze'] > 0.5:
                base[:, :3] = torch.tensor([action["rbase"][0], action["rbase"][1], 0], device=device.env.device)
            else:
                base[:, :3] = torch.tensor([action["rbase"][0], 0, action["rbase"][1]], device=device.env.device)

        if self.action_config.arm_action.controller.use_relative_mode:  # Relative mode
            arm_action = action["arm_delta"]
        else:  # Absolute mode
            abs_pose = action["arm_abs"]
            # _cumulative_base = (device.robot.data.body_link_pos_w[0,2]-device.robot.data.root_com_pos_w[0])
            # base_quat = device.robot.data.body_link_quat_w[0,3]
            _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                               device.robot.data.root_com_quat_w[0],
                                                                               device.robot.data.body_link_pos_w[0, 2],
                                                                               device.robot.data.body_link_quat_w[0, 3])
            # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
            base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]

            cos_yaw = torch.cos(base_yaw)
            sin_yaw = torch.sin(base_yaw)
            rot_mat_2d = torch.tensor([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ], device=device.env.device)
            robot_x = base[0][0]
            robot_y = base[0][1]
            local_xy = torch.tensor([robot_x, robot_y], device=device.env.device)
            world_xy = torch.matmul(rot_mat_2d, local_xy)
            base[:, 0] = world_xy[0]
            base[:, 1] = world_xy[1]

            pose_quat = abs_pose[3:7]
            base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
            combined_quat = T.quat_multiply(base_quat, pose_quat)

            base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)
            arm_action = abs_pose.clone()
            rot_mat = T.quat2mat(base_quat)
            gripper_movement = torch.matmul(rot_mat, arm_action[:3])

            pose_movement = base_movement + gripper_movement

            arm_action[:3] = pose_movement
            arm_action[3] = combined_quat[3]  # 更新旋转四元数 xyzw 2 wxyz
            arm_action[4:7] = combined_quat[:3]

        arm_action = arm_action.repeat(num_envs, 1)
        gripper = action["arm_gripper"] > 0
        gripper_action = torch.zeros(num_envs, 1, device=device.env.device)
        gripper_action[:] = -1.0 if gripper else 1.0
        return torch.concat([arm_action, gripper_action, base], dim=1)


@configclass
class PandaOmronSceneCfg:
    robot: ArticulationCfg = FRANKA_OMRON_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                  prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_hand",
                  name="ee_tcp",
                  offset=OffsetCfg(
                      pos=(0.0, 0.0, 0.1034),
                  ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_leftfinger",
                name="tool_leftfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/omron_v2/Franka/panda_rightfinger",
                name="tool_rightfinger",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.046),
                ),
            ),
        ],
    )

    base_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/omron_v2/mobilebase0_wheeled_base",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor.*"],
    )

    left_gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/omron_v2/Franka/panda_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )

    right_gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/omron_v2/Franka/panda_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[],
    )


@configclass
class PandaOmronCameraCfg:
    agentview_left_camera: TiledCameraCfg = None
    agentview_right_camera: TiledCameraCfg = None
    eye_in_hand_camera: TiledCameraCfg = None


@configclass
class PandaOmronObservationsCfg(EmbodimentBaseObservationCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PandaOmronGeneralObsCfg(EmbodimentGeneralObsCfg):
        gripper_pos: ObsTerm = ObsTerm(func=lw_benchhub_mdp.gripper_pos)

    embodiment_general_obs: PandaOmronGeneralObsCfg = PandaOmronGeneralObsCfg()


@configclass
class PandaOmronPolicyObservationsCfg(EmbodimentBasePolicyObservationCfg):
    """Observations for policy group with state values."""
    pass


@configclass
class PandaOmronEventCfg:
    pass


@configclass
class PandaOmronMimicEnv:
    pass


class PandaOmronRelEmbodiment(PandaOmronEmbodiment):
    name: str = "PandaOmron-Rel"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = PandaOmronRelActionsCfg()


@configclass
class PandaOmronRelActionsCfg:
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=4,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0434),
                                                                         rot=(0.9537170, 0, 0, 0.3007058)),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    base_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["mobilebase_.*"],
        scale={
            "mobilebase_forward": 0.01,
            "mobilebase_side": 0.01,
            "mobilebase_yaw": 0.02,
            "mobilebase_torso_height": 0.01,
        },  # 01,
        use_zero_offset=True,  # use default offset is not working for base action
    )


class PandaOmronAbsEmbodiment(PandaOmronEmbodiment):
    name: str = "PandaOmron-Abs"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.action_config = PandaOmronAbsActionsCfg()


@configclass
class PandaOmronAbsActionsCfg:
    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    base_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["mobilebase_.*"],
        scale={
            "mobilebase_forward": 0.01,
            "mobilebase_side": 0.01,
            "mobilebase_yaw": 0.02,
            "mobilebase_torso_height": 0.01,
        },  # 01,
        use_zero_offset=True,  # use default offset is not working for base action
    )


class PandaOmronRLEmbodiment(PandaOmronEmbodiment):
    name: str = "PandaOmron-RL"

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(initial_pose)
        self.action_config = PandaOmronRLActionsCfg()


@configclass
class PandaOmronRLActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )
    base_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["mobilebase_.*"],
        scale={
            "mobilebase_forward": 0.01,
            "mobilebase_side": 0.01,
            "mobilebase_yaw": 0.02,
            "mobilebase_torso_height": 0.01,
        },  # 01,
        use_zero_offset=True,  # use default offset is not working for base action
    )
