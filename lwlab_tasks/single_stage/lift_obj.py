# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from lwlab.core.models.fixtures import FixtureType
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.utils.env import ExecuteMode
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup

##
# Scene definition
##
# Increase PhysX GPU aggregate pairs capacity to avoid simulation errors
sim_utils.simulation_context.gpu_total_aggregate_pairs_capacity = 160000

from isaaclab.managers import EventTermCfg as EventTerm
import lwlab_rl.lift_obj.mdp as mdp
from isaaclab.managers import SceneEntityCfg


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

    # reset_dome_lighting = EventTerm(
    #     func=mdp.randomize_scene_lighting,
    #     mode="reset",
    #     params={
    #         "intensity_range": (50.0, 800.0),
    #         "color_variation": 0.35,
    #         "default_intensity": 800.0,
    #         "default_color": (0.75, 0.75, 0.75),
    #         "asset_cfg": SceneEntityCfg("light"),
    #     },
    # )


@configclass
class LeRobotVisualObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        target_qpos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
        delta_reset_qpos = ObsTerm(func=mdp.get_delta_reset_qpos, params={"action_name": 'arm_action'})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class LiftObj(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """
    counter_id: FixtureType = FixtureType.COUNTER
    task_name: str = "LiftObj"
    fix_object_pose_cfg: dict = {"object": {"pos": (2.94, -4.08, 0.95)}}  # y- near to robot
    resample_robot_placement_on_reset = False
    events: EventCfg = EventCfg()
    observations: LeRobotVisualObservationsCfg = LeRobotVisualObservationsCfg()

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref("counter", dict(id=self.counter_id, fix_id=2))
        # self.useful_fixture_names = [self.counter.name]
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):

        cfgs = []

        cfgs.append(
            dict(
                name="object",
                obj_groups="cube",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def quat_to_rotation_matrix(quat):
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        rot_matrix = torch.stack([
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1)
        ], dim=-2)

        return rot_matrix

    def normalize_vector(x: torch.Tensor, eps=1e-6):
        """normalizes a given torch tensor x and if the norm is less than eps, set the norm to 0"""
        norm = torch.linalg.norm(x, axis=1)
        norm[norm < eps] = 1
        norm = 1 / norm
        return torch.multiply(x, norm[:, None])

    def compute_angle_between(self, x1: torch.Tensor, x2: torch.Tensor):
        """Compute angle (radian) between two torch tensors"""
        x1, x2 = self.normalize_vector(x1), self.normalize_vector(x2)
        dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
        return torch.arccos(dot_prod)

    def _check_success(self):
        """
        Check if the cube is lifted.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        if self.context.execute_mode == ExecuteMode.TRAIN:
            return torch.tensor([False], device=self.env.device).repeat(self.env.num_envs)
        gripper_contact_force = self.env.scene.sensors["gripper_object_contact"]._data.force_matrix_w[:, 0, 0, :]
        jaw_contact_force = self.env.scene.sensors["jaw_object_contact"]._data.force_matrix_w[:, 0, 0, :]

        gripper_object_force = torch.linalg.norm(gripper_contact_force, dim=1)
        jaw_object_force = torch.linalg.norm(jaw_contact_force, dim=1)

        gripper_pose_w = self.env.scene._articulations['robot'].data.body_pose_w[..., -2, :]
        jaw_pose_w = self.env.scene._articulations['robot'].data.body_pose_w[..., -1, :]
        gripper_quat_w = gripper_pose_w[:, 3:7]  # [num_envs, 4]
        jaw_quat_w = jaw_pose_w[:, 3:7]          # [num_envs, 4]

        gripper_rot_matrix = self.quat_to_rotation_matrix(gripper_quat_w)
        jaw_rot_matrix = self.quat_to_rotation_matrix(jaw_quat_w)

        ldirection = -gripper_rot_matrix[..., :3, 0]  # [num_envs, 3]
        rdirection = jaw_rot_matrix[..., :3, 0]     # [num_envs, 3]

        langle = self.compute_angle_between(ldirection, gripper_contact_force)
        rangle = self.compute_angle_between(rdirection, jaw_contact_force)

        min_force = 0.5
        max_angle = 110
        lflag = torch.logical_and(
            gripper_object_force >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            jaw_object_force >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        is_grasping = torch.logical_and(lflag, rflag)
        object_height = self.env.scene['object'].data.root_pos_w[:, 2]
        is_height_sufficient = (object_height >= 0.965)

        success = is_grasping & is_height_sufficient
        return success
