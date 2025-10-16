# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from lwlab.core.models.fixtures import FixtureType
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import RigidObjectCfg

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch

from lwlab.core.tasks.base import BaseTaskEnvCfg

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from lwlab.core.rl.base import BaseRLEnvCfg

from lwlab_tasks.single_stage.lift_obj import LiftObj
from lwlab.core.robots.unitree.g1 import UnitreeG1HandEnvRLCfg

from . import mdp

from lwlab.core import mdp as lwlab_mdp
import torch.nn.functional as F
import cv2
from typing import Dict, Optional, Literal, List
##
# Scene definition
##
# Increase PhysX GPU aggregate pairs capacity to avoid simulation errors
sim_utils.simulation_context.gpu_total_aggregate_pairs_capacity = 160000


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.05, 0.05), pos_y=(-0.20, -0.05), pos_z=(0.3, 0.35), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

# State-based Observations


@configclass
class StateBasedObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        obj_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.1, 0.1), "y": (0, 0.25), "z": (0.0, 0.0)},
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.0), "yaw": (0.0, 90.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="BuildingBlock003"),
        },
    )

    reset_dome_lighting = EventTerm(
        func=mdp.randomize_scene_lighting,
        mode="reset",
        params={
            "intensity_range": (50.0, 800.0),
            "color_variation": 0.35,
            "default_intensity": 800.0,
            "default_color": (0.75, 0.75, 0.75),
            "asset_cfg": SceneEntityCfg("light"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.98}, weight=15.0)

    lifting_object_grasped = RewTerm(
        func=mdp.object_is_lifted_grasped,
        params={
            "grasp_threshold": 0.08,
            "velocity_threshold": 0.15,
        },
        weight=8.0
    )
    gripper_close_action_reward = RewTerm(func=mdp.gripper_close_action_reward, weight=1.0, params={'asset_cfg': SceneEntityCfg("robot", joint_names=['right_hand_.*'])})

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.97, "command_name": "object_pose"},
        weight=16.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 36000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -5e-3, "num_steps": 36000}
    # )


@configclass
class BaseLiftObjRLEnvCfg(BaseRLEnvCfg, LiftObj):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """
    observations: StateBasedObservationsCfg = StateBasedObservationsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands: CommandsCfg = CommandsCfg()
    resample_objects_placement_on_reset: bool = False
    resample_robot_placement_on_reset: bool = False
    # fix_object_pose_cfg: dict = {"object": {"pos": (3.93, -0.6, 0.95)}}  # y- near to robot

    # def set_reward_arm_joint_names(self, arm_joint_names):
    #     self.rewards.target_qpos_reward.params["asset_cfg"].joint_names = arm_joint_names

    def __post_init__(self):
        """Post initialization."""
        # general settings
        super().__post_init__()

        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.decimation = 5
        self.episode_length_s = 3.2
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.terminations.object_dropping = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.9, "asset_cfg": SceneEntityCfg("object")}, time_out=True  # TODO
        )


class G1StateLiftObjRLEnvCfg(UnitreeG1HandEnvRLCfg, BaseLiftObjRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # for LeRobot-RL
        # self.commands.object_pose.body_name = "gripper"
        # for G1-RL
        self.commands.object_pose.body_name = "right_wrist_yaw_link"
        # for franka
        # self.commands.object_pose.body_name = "panda_hand"


# Visual Observations based


@configclass
class G1VisualObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        actions = ObsTerm(func=mdp.last_action)

        image_hand = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("hand_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "model_device": "cuda:0",
            },
        )

        rgb_d435 = ObsTerm(
            func=mdp.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("d435_camera"),
                "data_type": "rgb",
                "model_name": "resnet18",
                "model_device": "cuda:0",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class G1VisualLiftObjRLEnvCfg(G1StateLiftObjRLEnvCfg):
    observations: G1VisualObservationsCfg = G1VisualObservationsCfg()


from lwlab.core.robots.lerobot.lerobotrl import LERobotEnvRLCfg, LERobot100EnvRLCfg


@configclass
class LeRobotLiftobjRewardsCfg:
    """Reward terms for the MDP."""
    reaching_reward = RewTerm(func=mdp.object_ee_distance_maniskill, weight=1.0)
    grasp_reward = RewTerm(func=mdp.object_is_grasped_maniskill, weight=1.0)
    place_reward = RewTerm(func=mdp.object_is_grasped_and_placed_maniskill, weight=1.0)
    touching_table = RewTerm(func=mdp.gripper_is_touching_table_maniskill, weight=-2.0)


@configclass
class LeRobotStateObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        target_qpos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
        delta_reset_qpos = ObsTerm(func=mdp.get_delta_reset_qpos, params={"action_name": 'arm_action'})
        obj_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class LeRobotStateLiftObjRLEnvCfg(LERobotEnvRLCfg, BaseLiftObjRLEnvCfg):
    observations: LeRobotStateObservationsCfg = LeRobotStateObservationsCfg()
    rewards: LeRobotLiftobjRewardsCfg = LeRobotLiftobjRewardsCfg()
    fix_object_pose_cfg: dict = {"object": {"pos": (2.94, -4.08, 0.95)}}  # y- near to robot

    def __post_init__(self):
        super().__post_init__()
        # for LeRobot-RL
        self.commands.object_pose.body_name = "gripper"
        # for G1-RL
        # self.commands.object_pose.body_name = "right_wrist_yaw_link"
        # for franka
        # self.commands.object_pose.body_name = "panda_hand"


class LeRobot100StateLiftObjRLEnvCfg(LERobot100EnvRLCfg, LeRobotStateLiftObjRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # for LeRobot-RL
        self.commands.object_pose.body_name = "Fixed_Jaw_tip"


@configclass
class LeRobotVisualObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        target_qpos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
        delta_reset_qpos = ObsTerm(func=mdp.get_delta_reset_qpos, params={"action_name": 'arm_action'})

        # image_global = ObsTerm(
        #     func=mdp.image_features,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("global_camera"),
        #         "data_type": "rgb",
        #         "model_name": "resnet18",
        #         "model_device": "cuda:0",
        #     },
        # )
        image_global = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("global_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class LeRobotVisualLiftObjRLEnvCfg(LeRobotStateLiftObjRLEnvCfg):
    observations: LeRobotVisualObservationsCfg = LeRobotVisualObservationsCfg()


class LeRobot100VisualLiftObjRLEnvCfg(LERobot100EnvRLCfg, LeRobotVisualLiftObjRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # for LeRobot-RL
        self.commands.object_pose.body_name = "Fixed_Jaw_tip"


@configclass
class DigitalTwinObservationCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        target_qpos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
        delta_reset_qpos = ObsTerm(func=mdp.get_delta_reset_qpos, params={"action_name": 'arm_action'})
        image_global = ObsTerm(
            func=mdp.overlay_image,
            params={
                "sensor_cfg": SceneEntityCfg("global_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class LeRobotLiftObjDigitalTwinCfg(LeRobotStateLiftObjRLEnvCfg):
    """A dictionary of rgb overlay paths.

    The key is the name of the rgb sensor, and the value is the path to the background image.
    example:{"camera_name": "path/to/greenscreen/background.png"}
    """
    rgb_overlay_mode: Optional[Literal["none", "debug", "background"]] = "none"

    render_objects = None
    task_name: str = "LiftObjDigitalTwin"
    observations: DigitalTwinObservationCfg = DigitalTwinObservationCfg()
    rgb_overlay_images: Dict[str, torch.Tensor] = {}

    foreground_semantic_id_mapping: Dict[str, int] = {}

    def __post_init__(self):
        super().__post_init__()
        self.rgb_overlay_mode = "background"
        self.rgb_overlay_paths = {
            "global_camera": "224101.jpg"
        }
        self.render_objects = [
            SceneEntityCfg("object"),
            SceneEntityCfg("robot"),
        ]
        self.setup_camera_and_foreground()
        # self.__record_semantic_id_mapping()

    def setup_camera_and_foreground(self):
        """Setup the camera for the ManagerBasedRLDigitalTwinEnv.
        1. add semantic tags to the render objects
        2. add semantic segmentation to the camera data types.
        3. modify the observation cfg to add overlay_image.
        """
        for obj in self.render_objects:
            obj_cfg = getattr(self.scene, obj.name)
            obj_cfg.spawn.semantic_tags = [("class", "foreground")]

        if self.rgb_overlay_paths is not None:
            for camera_name, path in self.rgb_overlay_paths.items():
                # preprocess camera cfg
                camera_cfg = getattr(self.scene, camera_name)
                if 'semantic_segmentation' not in camera_cfg.data_types:
                    camera_cfg.data_types.append('semantic_segmentation')
                camera_cfg.colorize_semantic_segmentation = False
                overlayed_image = self.read_overlay_image(path, target_size=(camera_cfg.width, camera_cfg.height)).to(self.sim.device)
                self.rgb_overlay_images[camera_name] = overlayed_image.repeat(self.num_envs, 1, 1, 1)
                # preprocess observation cfg
                # observation_cfg = getattr(cfg.observations.policy, camera_name)
                # observation_cfg.func = overlay_image

    def read_overlay_image(self, path: str, target_size: tuple[int, int]) -> torch.Tensor:
        """
        Read the overlay image and resize it to the target size.
        Args:
            path: the path to the overlay image.
            target_size: the target size of the overlay image.(width, height)
        Returns:
            the resized overlay image.(C, H, W)
        """
        image = torch.from_numpy(cv2.imread(path))

        if image.dim() == 3 and image.shape[2] in [3, 4]:  # [H, W, C]
            image = image.permute(2, 0, 1)  # [C, H, W]

        resize_image = F.interpolate(image.unsqueeze(0), size=(target_size[1], target_size[0]), mode="bilinear").squeeze(0)  # size is (height, width)
        resize_image = resize_image.squeeze(0)
        # reorder the image to [C, H, W]
        if resize_image.shape[0] in [3, 4]:  # [C, H, W]
            resize_image = resize_image.permute(1, 2, 0)  # [H, W, C]

        return resize_image

    def record_semantic_id_mapping(self, scene):
        for camera_name in self.rgb_overlay_paths.keys():
            for semantic_id, label in scene.sensors[camera_name].data.info['semantic_segmentation']['idToLabels'].items():
                if label['class'] == 'foreground':
                    self.foreground_semantic_id_mapping[camera_name] = int(semantic_id)
                    break


@configclass
class LeRobot100LiftObjDigitalTwinCfg(LERobot100EnvRLCfg, LeRobotLiftObjDigitalTwinCfg):

    def __post_init__(self):
        super().__post_init__()
        # for LeRobot-RL
        self.commands.object_pose.body_name = "Fixed_Jaw_tip"
