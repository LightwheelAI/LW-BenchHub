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
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg

from robocasa.models.fixtures import FixtureType
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip

from dataclasses import MISSING
from tasks.base import BaseTaskEnvCfg
from lwlab.utils.usd_utils import OpenUsd as usd
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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

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

    # observation groups
    policy: PolicyCfg = PolicyCfg()


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
            "position_range": (0.0, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )


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
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
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
                                         'asset_cfg': SceneEntityCfg("robot", joint_names=MISSING),
                                         'target_frame': MISSING})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out2 = DoneTerm(func=mdp.time_out, time_out=True)
    # joint_vel_limit = DoneTerm(func=mdp.joint_vel_out_of_limit, time_out=True)


@configclass
class OpenDrawer(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """
    observations: ObservationsCfg = ObservationsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    drawer_id: FixtureType = FixtureType.DRAWER
    task_name: str = "OpenDrawerrl"

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()

        self.drawer = self.register_fixture_ref("drawer", dict(id=self.drawer_id, fix_id=4))

        self.useful_fixture_names = [self.drawer.name]

        self.init_robot_base_pos = self.drawer

    def _get_obj_cfgs(self):

        cfgs = []

        return cfgs

    def _check_success(self):
        """
        Check if the counter to cabinet pick and place task is successful.
        Checks if the object is inside the cabinet and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        return False

    def set_reward_gripper_joint_names(self, gripper_joint_names):
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = gripper_joint_names

    def set_reward_arm_joint_names(self, arm_joint_names):
        self.rewards.target_qpos_reward.params["asset_cfg"].joint_names = arm_joint_names

    def __post_init__(self):
        """Post initialization."""
        # general settings
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 4.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.0
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["right_hand_.*"]  # self.gripper_joint_names

        name = self.drawer.name

        stage = usd.get_stage(self.usd_path)
        root_prim = stage.GetPseudoRoot()
        prim = usd.get_prim_by_name(root_prim, name)[0]
        joint_prims = usd.get_all_joints_without_fixed(prim)
        joint_names = [joint_prim.GetName() for joint_prim in joint_prims]
        self.rewards.target_qpos_reward.params["asset_cfg"].joint_names = joint_names

        base_link = 'corpus'
        target_link = usd.get_prim_by_suffix(prim, 'handle')[0].GetName()

        frame_cfg = FrameTransformerCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Scene/{name}/{base_link}",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Scene/{name}/{target_link}",
                    name=target_link,
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                        rot=(0.0, 0.0, 0.7071, 0.7071),  # align with end-effector frame
                    ),
                ),
            ],
        )
        setattr(self.scene, f"{name}_frame", frame_cfg)

        self.observations.policy.cabinet_joint_pos.params["asset_cfg"].name = name
        self.observations.policy.cabinet_joint_vel.params["asset_cfg"].name = name
        self.observations.policy.rel_ee_drawer_distance.params["target_frame"] = f'{name}_frame'
        self.events.cabinet_physics_material.params["asset_cfg"].name = name

        self.events.cabinet_physics_material.params["asset_cfg"].body_names = target_link

        self.events.reset_cab_joints.params["asset_cfg"].name = name

        self.events.reset_cab_joints.params["asset_cfg"].joint_names = joint_names

        self.rewards.open_drawer_bonus.params["asset_cfg"].name = name
        self.rewards.multi_stage_open_drawer.params["asset_cfg"].name = name

        self.rewards.approach_ee_handle.params["target_frame"] = f'{name}_frame'
        self.rewards.align_ee_handle.params["target_frame"] = f'{name}_frame'
        self.rewards.approach_gripper_handle.params["target_frame"] = f'{name}_frame'
        self.rewards.align_grasp_around_handle.params["target_frame"] = f'{name}_frame'
        self.rewards.grasp_handle.params["target_frame"] = f'{name}_frame'
        self.rewards.open_drawer_bonus.params["target_frame"] = f'{name}_frame'
        self.rewards.multi_stage_open_drawer.params["target_frame"] = f'{name}_frame'

        self.observations.policy.cabinet_joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.policy.cabinet_joint_vel.params["asset_cfg"].joint_names = joint_names
        self.rewards.open_drawer_bonus.params["asset_cfg"].joint_names = joint_names
        self.rewards.multi_stage_open_drawer.params["asset_cfg"].joint_names = joint_names
        self.rewards.target_qpos_reward.params["target_frame"] = f'{name}_frame'
        self.rewards.target_qpos_reward.params["target_qpos"] = torch.tensor([-24, -50, 42.0, 50.5, -27.5, -5.0, -30.0]) * torch.pi / 180
