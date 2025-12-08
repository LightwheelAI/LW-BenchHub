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

from dataclasses import MISSING
from typing import Dict, Literal, Optional

import cv2
import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg

from lw_benchhub.core.rl.base import LwRL, RlBasePolicyObservationCfg
from lw_benchhub.core.robots.lerobot.lerobotrl import LeRobot100RL, LeRobotRL
from lw_benchhub.core.robots.unitree.g1 import UnitreeG1HandEnvRLCfg
from lw_benchhub.utils.decorators import rl_on
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.lift_obj import LiftObj

from . import mdp

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


@configclass
class G1LiftObjPolicyObsCfg(RlBasePolicyObservationCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


# g1 state-based observations
@configclass
class G1LiftObjStatePolicyObsCfg(G1LiftObjPolicyObsCfg):
    """Observations for policy group."""
    obj_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})


# g1 visual-based observations
@configclass
class G1LiftObjVisualPolicyObsCfg(G1LiftObjStatePolicyObsCfg):
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


@configclass
class LeRobotLiftObjPolicyObsCfg(RlBasePolicyObservationCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos)
    target_qpos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
    delta_reset_qpos = ObsTerm(func=mdp.get_delta_reset_qpos, params={"action_name": 'arm_action'})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


# lerobot state-based observations
@configclass
class LeRobotLiftObjStatePolicyObsCfg(LeRobotLiftObjPolicyObsCfg):
    obj_pos = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("object")})


# lerobot visual-based observations
@configclass
class LeRobotLiftObjVisualPolicyObsCfg(LeRobotLiftObjPolicyObsCfg):
    image_global = ObsTerm(
        func=mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("global_camera"),
            "data_type": "rgb",
            "normalize": False,
        },
    )


@configclass
class EventCfg:
    """Configuration for events."""
    reset_object_position: EventTerm = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (-0.1, 0.1), "y": (0, 0.25), "z": (0.0, 0.0)},
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08), "z": (0.0, 0.0), "yaw": (0.0, 90.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="BuildingBlock003"),
        },
    )

    randomize_scene_lighting: EventTerm = EventTerm(
        func=mdp.randomize_scene_lighting,
        mode="reset",
        params={
            "intensity_range": (50000.0, 60000.0),
            "asset_name": "room_light",
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
class LeRobotLiftobjRewardsCfg:
    """Reward terms for the MDP."""
    reaching_reward = RewTerm(func=mdp.object_ee_distance_maniskill, weight=1.0)
    grasp_reward = RewTerm(func=mdp.object_is_grasped_maniskill, weight=1.0)
    place_reward = RewTerm(func=mdp.object_is_grasped_and_placed_maniskill, weight=1.0)
    touching_table = RewTerm(func=mdp.gripper_is_touching_table_maniskill, weight=-2.0)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 36000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -5e-3, "num_steps": 36000}
    # )


@rl_on(task=LiftObj)
@rl_on(embodiment=UnitreeG1HandEnvRLCfg)
class G1LiftObjStateRL(LwRL):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """
    # fix_object_pose_cfg: dict = {"object": {"pos": (3.93, -0.6, 0.95)}}  # y- near to robot

    # def set_reward_arm_joint_names(self, arm_joint_names):
    #     self.rewards.target_qpos_reward.params["asset_cfg"].joint_names = arm_joint_names

    def __init__(self):
        super().__init__()
        self.rewards_cfg = RewardsCfg()
        self.events_cfg = EventCfg()
        self.curriculum_cfg = CurriculumCfg()
        self.commands_cfg = CommandsCfg()
        self.policy_observation_cfg = G1LiftObjStatePolicyObsCfg()
        self.resample_objects_placement_on_reset = False
        self.resample_robot_placement_on_reset = False

    def setup_env_config(self, orchestrator):
        super().setup_env_config(orchestrator)
        # LeRobot-RL: gripper, G1-RL: right_wrist_yaw_link, franka: panda_hand
        orchestrator.task.commands_cfg.object_pose.body_name = "right_wrist_yaw_link"
        orchestrator.task.termination_cfg.object_dropping = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.9, "asset_cfg": SceneEntityCfg("object")}, time_out=True  # TODO
        )


class G1LiftObjVisualRL(G1LiftObjStateRL):

    def __init__(self):
        super().__init__()
        self.policy_observation_cfg = G1LiftObjVisualPolicyObsCfg()


@rl_on(task=LiftObj)
@rl_on(embodiment=LeRobotRL)
class LeRobotLiftObjStateRL(LwRL):

    def __init__(self):
        super().__init__()
        self.rewards_cfg = LeRobotLiftobjRewardsCfg()
        self.policy_observation_cfg = LeRobotLiftObjStatePolicyObsCfg()
        self.events_cfg = EventCfg()
        self.curriculum_cfg = CurriculumCfg()
        self.commands_cfg = CommandsCfg()
        self.fix_object_pose_cfg: dict = {"object": {"pos": (2.94, -4.08, 0.95)}}  # y- near to robot

    def setup_env_config(self, orchestrator):
        super().setup_env_config(orchestrator)
        orchestrator.task.commands_cfg.object_pose.body_name = "gripper"


@rl_on(embodiment=LeRobot100RL)
class LeRobot100LiftObjStateRL(LeRobotLiftObjStateRL):

    def setup_env_config(self, orchestrator):
        super().setup_env_config(orchestrator)
        orchestrator.task.commands_cfg.object_pose.body_name = "Fixed_Jaw_tip"


class LeRobotLiftObjVisualRL(LeRobotLiftObjStateRL):

    def __init__(self):
        super().__init__()
        self.policy_observation_cfg = LeRobotLiftObjVisualPolicyObsCfg()


class LeRobot100LiftObjVisualRL(LeRobot100LiftObjStateRL):

    def __init__(self):
        super().__init__()
        self.policy_observation_cfg = LeRobotLiftObjVisualPolicyObsCfg()


@configclass
class LeRobotLiftObjDigitalTwin(LeRobotLiftObjVisualRL):

    def __init__(self):
        super().__init__()
        """A dictionary of rgb overlay paths.

        The key is the name of the rgb sensor, and the value is the path to the background image.
        example:{"camera_name": "path/to/greenscreen/background.png"}
        """
        self.rgb_overlay_mode: Optional[Literal["none", "debug", "background"]] = "background"
        self.render_objects = [
            SceneEntityCfg("object"),
            SceneEntityCfg("robot"),
        ]
        self.rgb_overlay_images: Dict[str, torch.Tensor] = {}
        self.foreground_semantic_id_mapping: Dict[str, int] = {}
        self.rgb_overlay_paths = {
            "global_camera": "224101.jpg"
        }

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        super().modify_env_cfg(env_cfg)

        self.setup_camera_and_foreground(env_cfg)
        # self.__record_semantic_id_mapping()

    def setup_camera_and_foreground(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        """Setup the camera for the ManagerBasedRLDigitalTwinEnv.
        1. add semantic tags to the render objects
        2. add semantic segmentation to the camera data types.
        3. modify the observation cfg to add overlay_image.
        """
        for obj in self.render_objects:
            obj_cfg = getattr(env_cfg.scene, obj.name)
            obj_cfg.spawn.semantic_tags = [("class", "foreground")]

        if self.rgb_overlay_paths is not None:
            for camera_name, path in self.rgb_overlay_paths.items():
                # preprocess camera cfg
                camera_cfg = getattr(env_cfg.scene, camera_name)
                if 'semantic_segmentation' not in camera_cfg.data_types:
                    camera_cfg.data_types.append('semantic_segmentation')
                camera_cfg.colorize_semantic_segmentation = False
                overlayed_image = self.read_overlay_image(path, target_size=(camera_cfg.width, camera_cfg.height)).to(env_cfg.sim.device)
                self.rgb_overlay_images[camera_name] = overlayed_image.repeat(self.context.num_envs, 1, 1, 1)
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
class LeRobot100LiftObjDigitalTwin(LeRobotLiftObjDigitalTwin):

    def setup_env_config(self, orchestrator):
        super().setup_env_config(orchestrator)
        orchestrator.task.commands_cfg.object_pose.body_name = "Fixed_Jaw_tip"
