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

import torch

from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg

from lw_benchhub.core.rl.base import LwRL, RlBasePolicyObservationCfg
from lw_benchhub.core.robots.unitree.g1 import UnitreeG1HandEnvRLCfg
from lw_benchhub.utils.decorators import rl_on
from lw_benchhub.utils.usd_utils import OpenUsd as Usd
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.kitchen_drawer import OpenDrawer

from . import mdp


##
# Pre-defined configs
##

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##


@configclass
class G1OpenDrawerPolicyObsCfg(RlBasePolicyObservationCfg):
    """Observation specifications for the MDP."""
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    cabinet_joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg(MISSING, joint_names=MISSING)},
    )
    cabinet_joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg(MISSING, joint_names=MISSING)},
    )
    rel_ee_drawer_distance = ObsTerm(
        func=mdp.rel_ee_drawer_distance,
        params={'target_frame': MISSING}
    )

    actions = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True

    def set_missing_params(self, asset_name, handle_frame_name, joint_names):
        self.cabinet_joint_pos.params["asset_cfg"].name = asset_name
        self.cabinet_joint_vel.params["asset_cfg"].name = asset_name
        self.rel_ee_drawer_distance.params["target_frame"] = handle_frame_name
        self.cabinet_joint_pos.params["asset_cfg"].joint_names = joint_names
        self.cabinet_joint_vel.params["asset_cfg"].joint_names = joint_names


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(MISSING, body_names=MISSING),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_cab_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(MISSING, joint_names=MISSING),
            "position_range": (0.0, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    def set_missing_params(self, asset_name, target_link, joint_names):
        self.cabinet_physics_material.params["asset_cfg"].name = asset_name
        self.cabinet_physics_material.params["asset_cfg"].body_names = target_link

        self.reset_cab_joints.params["asset_cfg"].name = asset_name
        self.reset_cab_joints.params["asset_cfg"].joint_names = joint_names


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=5.0, params={"threshold": 0.2, "target_frame": MISSING})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5, params={'target_frame': MISSING})

    # 2. Grasp the handle
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING, "target_frame": MISSING})
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125, params={'target_frame': MISSING})
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": MISSING,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),  # the joint names of the gripper
            'target_frame': MISSING
        },
    )

    # 3. Open the drawer
    open_drawer_bonus = RewTerm(
        func=mdp.open_drawer_bonus,
        weight=7.5,
        params={"asset_cfg": SceneEntityCfg(MISSING, joint_names=MISSING), 'target_frame': MISSING},
    )
    multi_stage_open_drawer = RewTerm(
        func=mdp.multi_stage_open_drawer,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg(MISSING, joint_names=MISSING), 'target_frame': MISSING},
    )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    target_qpos_reward = RewTerm(func=mdp.target_qpos_reward, weight=0.1,
                                 params={'target_qpos': MISSING,
                                         'asset_cfg': SceneEntityCfg("robot", joint_names=MISSING),  # the joint names of the arm
                                         'target_frame': MISSING})

    def set_missing_params(self, asset_name, handle_frame_name, joint_names, target_qpos):
        self.approach_ee_handle.params["target_frame"] = handle_frame_name
        self.align_ee_handle.params["target_frame"] = handle_frame_name

        self.approach_gripper_handle.params["target_frame"] = handle_frame_name
        self.approach_gripper_handle.params["offset"] = 0.04
        self.align_grasp_around_handle.params["target_frame"] = handle_frame_name
        self.grasp_handle.params["target_frame"] = handle_frame_name
        self.grasp_handle.params["open_joint_pos"] = 0.0

        self.open_drawer_bonus.params["asset_cfg"].name = asset_name
        self.open_drawer_bonus.params["target_frame"] = handle_frame_name
        self.open_drawer_bonus.params["asset_cfg"].joint_names = joint_names

        self.multi_stage_open_drawer.params["asset_cfg"].joint_names = joint_names
        self.multi_stage_open_drawer.params["target_frame"] = handle_frame_name
        self.multi_stage_open_drawer.params["asset_cfg"].name = asset_name

        self.target_qpos_reward.params["target_frame"] = handle_frame_name
        # self.target_qpos_reward.params["asset_cfg"].joint_names = joint_names
        self.target_qpos_reward.params["target_qpos"] = target_qpos


@rl_on(task=OpenDrawer)
@rl_on(embodiment=UnitreeG1HandEnvRLCfg)
class G1OpenDrawerRl(LwRL):
    """
    Class encapsulating the open drawer task.
    The robot is a G1.
    The underlying reward and observation terms are specified for the G1.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """

    def __init__(self):
        super().__init__()
        self.policy_observation_cfg = G1OpenDrawerPolicyObsCfg()
        self.rewards_cfg = RewardsCfg()
        self.events_cfg = EventCfg()

    def _set_reward_joint_names(self, gripper_joint_names, arm_joint_names):
        super()._set_reward_joint_names(gripper_joint_names, arm_joint_names)
        self.rewards_cfg.grasp_handle.params["asset_cfg"].joint_names = gripper_joint_names
        self.rewards_cfg.target_qpos_reward.params["asset_cfg"].joint_names = arm_joint_names

    def setup_env_config(self, orchestrator):
        super().setup_env_config(orchestrator)
        orchestrator.task.resample_robot_placement_on_reset = False
        orchestrator.task.resample_objects_placement_on_reset = False
        self.drawer_name = orchestrator.task.drawer.name
        self.handle_frame_name = f"{self.drawer_name}_frame"
        self.joint_names = orchestrator.task.drawer.door_joint_names
        self.base_link_name = 'corpus'
        self.target_link_name = Usd.get_prim_by_suffix(prim=orchestrator.task.drawer.prim, suffix='handle')[0].GetName()

        self.rewards_cfg.set_missing_params(
            asset_name=self.drawer_name,
            handle_frame_name=self.handle_frame_name,
            joint_names=self.joint_names,
            target_qpos=torch.tensor([-24, -50, 42.0, 50.5, -27.5, -5.0, -30.0]) * torch.pi / 180
        )
        self.events_cfg.set_missing_params(
            asset_name=self.drawer_name,
            target_link=self.target_link_name,
            joint_names=self.joint_names
        )
        self.policy_observation_cfg.set_missing_params(
            asset_name=self.drawer_name,
            handle_frame_name=self.handle_frame_name,
            joint_names=self.joint_names
        )

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        env_cfg = super().modify_env_cfg(env_cfg)
        frame_cfg = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{self.drawer_name}/{self.base_link_name}",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Scene/{self.drawer_name}/{self.target_link_name}",
                    name=self.target_link_name,
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(0.0, 0.0, 0.7071, 0.7071),  # align with end-effector frame
                    ),
                ),
            ],
        )
        setattr(env_cfg.scene, self.handle_frame_name, frame_cfg)
        return env_cfg
