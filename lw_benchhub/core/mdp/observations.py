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
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, ArticulationData
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerData
from isaaclab.utils.math import subtract_frame_transforms


def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the fingertips relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

    return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w - env.scene.env_origins.unsqueeze(1)

    return ee_pos


def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat


def ee_pose(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The pose of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.

    Returns:
        The pose of the end-effector in the environment frame.
        The pose is a tensor of shape (num_envs, 7) where the last dimension is [x, y, z, qw, qx, qy, qz].
    """
    return torch.cat([ee_pos(env), ee_quat(env, make_quat_unique)], dim=-1)


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
        return env.scene['robot']._data.joint_pos_target


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def get_eef_base(env: ManagerBasedRLEnv, base_link_name: str = 'dummy_link', eef_link_name: str = 'hand_link') -> torch.Tensor:
    robot = env.scene.articulations['robot']
    hand_ids, _ = robot.find_bodies(eef_link_name)
    base_link_ids, _ = robot.find_bodies(base_link_name)
    hand_idx = hand_ids[0]
    base_link_idx = base_link_ids[0]
    hand_pose_w = robot.data.body_link_pose_w[0, hand_idx, :]
    base_link_pose_w = robot.data.body_link_pose_w[0, base_link_idx, :]
    hand_pos_base, hand_quat_base = subtract_frame_transforms(
        base_link_pose_w[:3].unsqueeze(0),   # base position in world
        base_link_pose_w[3:].unsqueeze(0),   # base quaternion [w,x,y,z] in world
        hand_pose_w[:3].unsqueeze(0),   # left hand position in world
        hand_pose_w[3:].unsqueeze(0)    # left hand quaternion in world
    )

    eef_base = torch.cat([hand_pos_base, hand_quat_base], dim=-1)
    return eef_base
