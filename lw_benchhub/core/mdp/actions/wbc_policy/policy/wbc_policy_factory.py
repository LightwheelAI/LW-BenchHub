# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np

from .g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy
from .g1_homie_policy import G1HomiePolicy, G1HomiePolicyV2
from .identity_policy import IdentityPolicy
from .interpolation_policy import InterpolationPolicy


def get_wbc_policy(
    robot_type,
    robot_model,
    wbc_config,
    default_base_height=0.74,
    init_time=time.monotonic(),
):
    current_upper_body_pose = robot_model.get_initial_upper_body_pose()

    if robot_type == "g1":
        upper_body_policy_type = wbc_config.get("upper_body_policy_type", "interpolation")
        if upper_body_policy_type == "identity":
            upper_body_policy = IdentityPolicy()
        else:
            upper_body_policy = InterpolationPolicy(
                init_time=init_time,
                init_values={
                    "target_upper_body_pose": current_upper_body_pose,
                    "base_height_command": np.array([default_base_height]),
                },
                max_change_rate=wbc_config["upper_body_max_joint_speed"],
            )

        lower_body_policy_type = wbc_config.get("VERSION", "default")
        if lower_body_policy_type == "homie":
            lower_body_policy = G1HomiePolicy(
                robot_model=robot_model,
                config=wbc_config["HOMIE_CONFIG"],
                model_path=wbc_config["model_path"],
            )
        elif lower_body_policy_type == "homie_v2":
            lower_body_policy = G1HomiePolicyV2(
                robot_model=robot_model,
                config=wbc_config["HOMIE_CONFIG"],
                model_path=wbc_config["model_path"],
            )
        else:
            raise ValueError(f"Invalid lower body policy type: {lower_body_policy_type}, Supported lower body policy types: homie")

        wbc_policy = G1DecoupledWholeBodyPolicy(
            robot_model=robot_model,
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
        )
    else:
        raise ValueError(f"Invalid robot type: {robot_type}. Supported robot types: g1")
    return wbc_policy
