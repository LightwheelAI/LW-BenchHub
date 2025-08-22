# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Tuple

import numpy as np

from isaaclab.sensors import Camera

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from isaac_arena.policies.GR00T.io_utils import load_robot_joints_config
from isaac_arena.policies.GR00T.joints_conversion import remap_policy_joints_to_sim_joints, remap_sim_joints_to_policy_joints
from isaac_arena.policies.GR00T.robot_joints import JointsAbsPosition
from isaac_arena.policies.GR00T.policy_cfg import GR00TN15Config


class Gr00tN15Policy():
    def __init__(self, args: GR00TN15Config):
        self.args = args
        self.policy = self._load_policy()
        self._load_policy_joints_config()
        self._load_sim_joints_config()

    def _load_policy_joints_config(self):
        """Load the policy joint config from the data config."""
        self.gr00t_joints_config = load_robot_joints_config(self.args.gr00t_joints_config_path)

    def _load_sim_joints_config(self):
        """Load the simulation joint config from the data config."""
        self.g1_state_joints_config = load_robot_joints_config(self.args.state_joints_config_path)
        self.g1_action_joints_config = load_robot_joints_config(self.args.action_joints_config_path)

    def _load_policy(self):
        """Load the policy from the model path."""
        # assert os.path.exists(self.args.model_path), f"Model path {self.args.model_path} does not exist"

        # Use the same data preprocessor as the loaded fine-tuned ckpts
        self.data_config = DATA_CONFIG_MAP[self.args.data_config]

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        # load the policy
        return Gr00tPolicy(
            model_path=self.args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.args.embodiment_tag,
            denoising_steps=self.args.denoising_steps,
            device=self.args.policy_device,
        )

    def get_new_goal(
        self, current_state: JointsAbsPosition, ego_camera: Camera, language_instruction: str
    ) -> Tuple[JointsAbsPosition, np.ndarray, np.ndarray]:
        """
        Run policy prediction on the given observations. Produce a new action goal for the robot.

        Args:
            current_state: robot proprioceptive state observation
            ego_camera: camera sensor observation
            language_instruction: language instruction for the task

        Returns:
            A dictionary containing the inferred action for robot joints.
        """
        rgb = ego_camera.data.output["rgb"]
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        robot_state_policy = remap_sim_joints_to_policy_joints(current_state, self.gr00t_joints_config)

        # Pack inputs to dictionary and run the inference
        observations = {
            "annotation.human.task_description": [language_instruction],  # list of strings
            "video.ego_view": rgb.reshape(-1, 1, 480, 640, 3),  # numpy array of shape (N, 1, 480, 640, 3)
            "state.left_arm": robot_state_policy["left_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.right_arm": robot_state_policy["right_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.left_hand": robot_state_policy["left_hand"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.right_hand": robot_state_policy["right_hand"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
            "state.waist": robot_state_policy["waist"].reshape(-1, 1, 3),  # numpy array of shape (N, 1, 3)
        }

        robot_action_policy = self.policy.get_action(observations)

        base_height_command = robot_action_policy["action.base_height_command"]
        navigate_command = robot_action_policy["action.navigate_command"]

        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.gr00t_joints_config, self.g1_action_joints_config, self.args.simulation_device
        )

        return robot_action_sim, base_height_command, navigate_command

    def reset(self):
        """Resets the policy's internal state."""
        # As GN1 is a single-shot policy, we don't need to reset its internal state
        pass
