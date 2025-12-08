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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] < minimal_height, 0.0, 1.0)


def hand_is_lifted(
        env: ManagerBasedRLEnv, minimal_height: float,) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    return torch.where(ee_tcp_pos[:, 2] < minimal_height, -1.0, 0.0)

# def object_height(
#     env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     """Dense reward for the object being lifted"""
#     object: RigidObject = env.scene[object_cfg.name]
#     return object.data.root_pos_w[:, 2]


def object_is_lifted_grasped(
    env: ManagerBasedRLEnv,
    grasp_threshold: float = 0.08,
    velocity_threshold: float = 0.15,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    current_height = object.data.root_pos_w[:, 2]
    object_pos = object.data.root_pos_w
    object_velocity = object.data.root_lin_vel_w
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    if not hasattr(env, '_initial_object_height'):
        env._initial_object_height = current_height.clone()

    height_gain = current_height - env._initial_object_height

    ee_object_distance = torch.norm(ee_pos - object_pos, dim=-1)
    is_near_gripper = ee_object_distance < grasp_threshold
    horizontal_velocity = torch.norm(object_velocity[:, :2], dim=-1)
    is_stable = horizontal_velocity < velocity_threshold

    is_lifted = height_gain > 0.01

    is_properly_grasped = is_near_gripper & is_stable & is_lifted

    reward = torch.where(
        is_properly_grasped,
        height_gain,
        0.0
    )

    return reward


def gripper_close_action_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                                close_distance_threshold: float = 0.1) -> torch.Tensor:
    ee_frame_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w
    distance = torch.norm(object_pos - ee_frame_pos, dim=1)
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    gripper_closeness = torch.sum(torch.abs(gripper_joint_pos), dim=-1)
    is_near_target = distance < close_distance_threshold
    reward_near = is_near_target.float() * gripper_closeness
    reward_far = (~is_near_target).float() * (-0.1 * gripper_closeness)
    distance_weight = torch.exp(-distance / close_distance_threshold)
    return reward_near + reward_far * 0.1 + distance_weight * gripper_closeness * 0.5


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)
    # return -torch.log(object_ee_distance**2 / std + 1)


def grasp_handle(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    ee_frame_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    object_pos = env.scene["object"].data.root_pos_w
    distance = torch.norm(object_pos - ee_frame_pos, dim=1)
    is_gripper_close = distance < 0.04
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(gripper_joint_pos), dim=-1) * (is_gripper_close + 0.01)


def grasp_handle_fine_grained(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    ee_thumb_pos = env.scene["ee_frame"].data.target_pos_w[..., 1, :]
    ee_index_pos = env.scene["ee_frame"].data.target_pos_w[..., 2, :]
    ee_middle_pos = env.scene["ee_frame"].data.target_pos_w[..., 3, :]
    object_pos = env.scene["object"].data.root_pos_w
    distance_2 = torch.norm(object_pos - ee_thumb_pos, dim=1)
    distance_3 = torch.norm(object_pos - ee_index_pos, dim=1)
    distance_4 = torch.norm(object_pos - ee_middle_pos, dim=1)

    return - 2 * torch.log(distance_2 / 0.1 + 1) - torch.log(distance_3 / 0.1 + 1) - torch.log(distance_4 / 0.1 + 1)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_z = torch.tensor([0.0, 0.0, 1.0], device=ee_tcp_quat.device)
    handle_z = handle_z.unsqueeze(0).repeat(ee_tcp_quat.shape[0], 1)
    # get current x and z direction of the gripper
    ee_tcp_z = ee_tcp_rot_mat[..., 2]
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2)


def target_qpos_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    arm_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    lift_object = object_is_lifted(env, 0.97)
    target_qpos = torch.pi / 2 * torch.ones_like(arm_joint_pos)
    return -1 * torch.norm(torch.abs(target_qpos - torch.abs(arm_joint_pos)) / torch.abs(target_qpos), dim=-1) * (1 - lift_object)


def object_ee_distance_maniskill(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    gripper = ee_frame.data.target_pos_w[..., 0, :]
    jaw = ee_frame.data.target_pos_w[..., 1, :]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = (gripper + jaw) / 2
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.linalg.norm(cube_pos_w - ee_w, dim=1)

    # 计算reward
    reward = 1.0 - torch.tanh(5 * object_ee_distance)

    return reward


def normalize_vector(x: torch.Tensor, eps=1e-6):
    """normalizes a given torch tensor x and if the norm is less than eps, set the norm to 0"""
    norm = torch.linalg.norm(x, axis=1)
    norm[norm < eps] = 1
    norm = 1 / norm
    return torch.multiply(x, norm[:, None])


def compute_angle_between(x1: torch.Tensor, x2: torch.Tensor):
    """Compute angle (radian) between two torch tensors"""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
    return torch.arccos(dot_prod)


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


def object_is_grasped_maniskill(
    env: ManagerBasedRLEnv,
    min_force: float = 0.5,
    max_angle: float = 60
) -> torch.Tensor:
    # (num_envs, num_bodies, num_filter_shapes, 3)
    gripper_contact_force = env.scene.sensors["gripper_object_contact"]._data.force_matrix_w[:, 0, 0, :]
    jaw_contact_force = env.scene.sensors["jaw_object_contact"]._data.force_matrix_w[:, 0, 0, :]

    gripper_object_force = torch.linalg.norm(gripper_contact_force, dim=1)
    jaw_object_force = torch.linalg.norm(jaw_contact_force, dim=1)

    gripper_pose_w = env.scene._articulations['robot'].data.body_pose_w[..., -2, :]
    jaw_pose_w = env.scene._articulations['robot'].data.body_pose_w[..., -1, :]
    gripper_quat_w = gripper_pose_w[:, 3:7]  # [num_envs, 4]
    jaw_quat_w = jaw_pose_w[:, 3:7]          # [num_envs, 4]

    gripper_rot_matrix = quat_to_rotation_matrix(gripper_quat_w)
    jaw_rot_matrix = quat_to_rotation_matrix(jaw_quat_w)

    ldirection = -gripper_rot_matrix[..., :3, 0]  # [num_envs, 3]
    rdirection = jaw_rot_matrix[..., :3, 0]     # [num_envs, 3]

    langle = compute_angle_between(ldirection, gripper_contact_force)
    rangle = compute_angle_between(rdirection, jaw_contact_force)

    # print("langle:", torch.rad2deg(langle))
    # print("rangle:", torch.rad2deg(rangle))
    # print("-----------------------------------------")
    lflag = torch.logical_and(
        gripper_object_force >= min_force, torch.rad2deg(langle) <= max_angle
    )
    rflag = torch.logical_and(
        jaw_object_force >= min_force, torch.rad2deg(rangle) <= max_angle
    )
    is_grasp = torch.logical_and(lflag, rflag)
    # print(f"is_grasp:{is_grasp}")
    return is_grasp


def object_is_grasped_and_placed_maniskill(
    env: ManagerBasedRLEnv,
    min_force: float = 0.5,
    max_angel: float = 60
) -> torch.Tensor:

    is_grasp = object_is_grasped_maniskill(env, min_force, max_angel)

    target_qpos = env.scene['robot'].data.joint_pos
    device = target_qpos.device
    rest_qpos = torch.tensor(list(env.scene['robot'].cfg.init_state.joint_pos.values()), device=device)

    distance_to_rest_qpos = torch.linalg.norm(
        target_qpos[:, :-1] - rest_qpos[:-1], axis=-1
    )
    place_reward = torch.exp(-2 * distance_to_rest_qpos) * is_grasp
    # print(f"place_reward:{place_reward}")
    return place_reward


def gripper_is_touching_table_maniskill(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:

    gripper_table_force = torch.linalg.norm(env.scene.sensors["gripper_table_contact"]._data.force_matrix_w[:, 0, 0, :], dim=1)
    jaw_table_force = torch.linalg.norm(env.scene.sensors["jaw_table_contact"]._data.force_matrix_w[:, 0, 0, :], dim=1)
    touching_table = torch.logical_or(
        gripper_table_force >= 1e-2,
        jaw_table_force >= 1e-2,
    )
    # print(f"touching_table:{touching_table}")

    return touching_table.float()
