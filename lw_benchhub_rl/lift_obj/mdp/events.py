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
import random

from typing import Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import AssetBase

try:
    from pxr import Gf
except ImportError:
    # Fallback if pxr is not available
    Gf = None


def randomize_camera_uniform(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, float],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Reset the camera to a random position and rotation uniformly within the given ranges.

    * It samples the camera position and rotation from the given ranges and adds them to the
      default camera position and rotation, before setting them into the physics simulation.

    The function takes a dictionary of pose ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or rotation is set to zero for that axis.
    """
    asset: Camera = env.scene[asset_cfg.name]

    ori_pos_w = asset.data.pos_w
    if convention == "ros":
        ori_quat_w = asset.data.quat_w_ros
    elif convention == "opengl":
        ori_quat_w = asset.data.quat_w_opengl
    elif convention == "world":
        ori_quat_w = asset.data.quat_w_world

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = ori_pos_w[:, 0:3] + rand_samples[:, 0:3]  # camera usually spawn with robot, so no need to add env_origins
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(ori_quat_w, orientations_delta)

    asset.set_world_poses(positions, orientations, env_ids, convention)


def randomize_scene_lighting(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (400.0, 1200.0),
    color_range: tuple[float, float] = (0.0, 2.0),
    padding_ratio: float = 0.3,
    asset_name: str = "light",
):
    scene_range_x = env.cfg.isaaclab_arena_env.scene.scene_range[:, 0]
    scene_range_y = env.cfg.isaaclab_arena_env.scene.scene_range[:, 1]
    x_padding_inside = padding_ratio * (scene_range_x[1] - scene_range_x[0])
    y_padding_inside = padding_ratio * (scene_range_y[1] - scene_range_y[0])
    for env_id in env_ids:
        scene_env_prefix = env.scene.env_prim_paths[env_id]
        prim_path = scene_env_prefix + "/" + asset_name
        prim = env.scene.stage.GetPrimAtPath(prim_path)
        position_attr = prim.GetAttribute("xformOp:translate")
        intensity_attr = prim.GetAttribute("inputs:intensity")
        color_attr = prim.GetAttribute("inputs:color")
        original_z = position_attr.Get()[2]
        new_position_x = _sample_random_value(scene_range_x[0] + x_padding_inside, scene_range_x[1] - x_padding_inside, 1)[0]
        new_position_y = _sample_random_value(scene_range_y[0] + y_padding_inside, scene_range_y[1] - y_padding_inside, 1)[0]
        new_position = (new_position_x, new_position_y, original_z)
        position_attr.Set(new_position)

        new_intensity = _sample_random_value(intensity_range[0], intensity_range[1], 1)[0]
        intensity_attr.Set(new_intensity)

        new_color = _sample_random_value(color_range[0], color_range[1])
        color_attr.Set(new_color)


def _sample_random_value(min: float, max: float, n: float = 3):
    """Sample a random value from the given range.

    Args:
        min: The minimum value.
        max: The maximum value.
        n: The number of values to sample.
    """
    old_state = random.getstate()
    random.seed(None)
    result = tuple(random.random() * (max - min) + min for _ in range(n))
    random.setstate(old_state)
    return result
