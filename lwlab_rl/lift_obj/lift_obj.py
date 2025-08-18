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

from dataclasses import MISSING
from tasks.base import BaseTaskEnvCfg

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

from tasks.single_stage.lift_obj import LiftObj
from lwlab.core.robots.unitree.g1 import UnitreeG1HandEnvRLCfg

from . import mdp

from lwlab.core import mdp as lwlab_mdp

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
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0), "yaw": (0.0, 90.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="apple_11"),
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
    # object_hegiht = RewTerm(func=mdp.object_height, weight=1.)
    # target_qpos_reward = RewTerm(func=mdp.target_qpos_reward,
    #                              params={'asset_cfg': SceneEntityCfg("robot", joint_names=MISSING)}, weight=0.5)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.97, "command_name": "object_pose"},
        weight=16.0,
    )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.97, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # def set_missing_params(self, joint_names):
    #     self.target_qpos_reward.params["asset_cfg"].joint_names = joint_names


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 36000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -5e-3, "num_steps": 36000}
    )


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
    reset_objects_enabled: bool = False
    reset_robot_enabled: bool = False
    fix_object_pose_cfg: dict = {"object": {"pos": (3.93, -0.6, 0.95)}}  # y- near to robot

    # def set_reward_arm_joint_names(self, arm_joint_names):
    #     self.rewards.target_qpos_reward.params["asset_cfg"].joint_names = arm_joint_names

    def __post_init__(self):
        """Post initialization."""
        # general settings
        super().__post_init__()

        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.decimation = 2
        self.episode_length_s = 2.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.terminations.object_dropping = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.9, "asset_cfg": SceneEntityCfg("object")}  # TODO
        )


class G1StateLiftObjRLEnvCfg(UnitreeG1HandEnvRLCfg, BaseLiftObjRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # for Lerobot-RL
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
