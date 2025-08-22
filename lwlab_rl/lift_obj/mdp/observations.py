# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def camera_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    quat_w_ros = object.data.quat_w_ros[:, :4]
    object_pos_b_with_quat = torch.concatenate([object_pos_b, quat_w_ros], dim=1)

    return object_pos_b_with_quat


def get_delta_reset_qpos(
    env: ManagerBasedRLEnv,
    action_name: str = 'arm_action'
) -> torch.Tensor:
    device = env.device
    target_qpos = env.action_manager.get_term(action_name).processed_actions + env.scene['robot']._data.joint_pos
    rest_qpos = torch.tensor(list(env.scene['robot'].cfg.init_state.joint_pos.values()), device=device)
    dist_to_rest_qpos = target_qpos[:, :-1] - rest_qpos[:-1]
    return dist_to_rest_qpos


def get_target_qpos(
    env: ManagerBasedRLEnv,
    action_name: str = 'arm_action'
) -> torch.Tensor:
    """The last input action to the environment.

       The name of the action term for which the action is required. If None, the
       entire action tensor is returned.
       """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).processed_actions + env.scene['robot']._data.joint_pos
