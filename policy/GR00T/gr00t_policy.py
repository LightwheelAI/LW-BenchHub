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

import os
import sys
from typing_extensions import Mapping

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
# sys.path.append(parent_directory)
isaac_gr00t_path = os.path.join(parent_directory, "Isaac-GR00T")
sys.path.append(isaac_gr00t_path)

import torch
from typing import Dict, Any
from policy.base import BasePolicy

try:
    from policy.GR00T.data_config.data_config import LW_DATA_CONFIG_MAP
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
except ImportError as e:
    print(f"gr00t not found, please install gr00t first: {e},if you not run gr00t policy, please ignore this error")


class GR00TPolicy(BasePolicy):
    """GR00T Policy Implementation"""

    def __init__(self, usr_args):
        super().__init__(usr_args)

    def _load_policy(self):
        """Load the policy from the model path."""
        # Use the same data preprocessor as the loaded fine-tuned ckpts
        if self.usr_args["data_config"] in DATA_CONFIG_MAP:
            self.data_config = DATA_CONFIG_MAP[self.usr_args["data_config"]]
        elif self.usr_args["data_config"] in LW_DATA_CONFIG_MAP:
            self.data_config = LW_DATA_CONFIG_MAP[self.usr_args["data_config"]]

        modality_config = self.data_config.modality_config()
        modality_transform = self.data_config.transform()
        # load the policy
        return Gr00tPolicy(
            model_path=self.usr_args["ckpt_setting"],
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=self.usr_args["embodiment_tag"],
            denoising_steps=self.usr_args["denoising_steps"],
            device=self.simulation_device,
        )

    def get_model(self, usr_args):
        """Get GR00T model instance"""
        observation_config = usr_args.get("observation_config", None)
        self.simulation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_config = observation_config or {}
        self.model = self._load_policy()

    def _build_observation_window(self, obs):
        custom_mapping = self.observation_config.get("custom_mapping", {})
        obs_window = {"annotation.human.action.task_description": [self.instruction]}
        for key, mapping in custom_mapping.items():
            if isinstance(mapping, dict):
                obs_window[key] = obs[next(iter(mapping.keys()))][..., next(iter(mapping.values()))]
                obs_window[key] = obs_window[key][None, ...]  # N, env, ...
            elif key == "joint_pos" or key == "eef_base":
                joint_order = [i for i in range(len(self.data_config.state_keys))] if not mapping else mapping
                for idx, key in enumerate(self.data_config.state_keys):
                    obs_window[key] = obs[key][0][joint_order[idx]]
                    obs_window[key] = obs_window[key][None, ...]  # N, env, ...
                    obs_window[key] = obs_window[key][None, ...]  # N, env, ...
                    obs_window[key] = obs_window[key][None, ...]  # N, env, ...
            else:
                mapping_key = next(iter([key for key in obs.keys() if key.startswith(mapping)]), None)
                obs_window[key] = obs[mapping_key][None, ...]  # N, env, H,W,C

        return obs_window

    def encode_obs(self, observation):
        observation = super().encode_obs(observation, transpose=False, keep_dim_env=True)
        observation = self._build_observation_window(observation)
        return observation

    def _mapping_action(self, actions):
        robot_actions = []
        for key in actions.keys():
            robot_actions.append(torch.tensor(actions[key], device=self.simulation_device))
        robot_actions = torch.concat(robot_actions, dim=-1)
        return robot_actions

    def get_action(self, observation):
        robot_action_policy = self.model.get_action(observation)
        robot_actions = self._mapping_action(robot_action_policy)
        return robot_actions

    def x7s_obs_mapping(self, obs):
        obs['state.gripper'] = obs['state.gripper'] / 0.044
        return obs

    def x7s_action_mapping(self, action: torch.Tensor) -> torch.Tensor:
        action[..., -1] = -action[..., -1]
        return action

    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        for _ in range(usr_args['time_out_limit']):
            observation = self.encode_obs(observation)
            # observation = self.x7s_obs_mapping(observation)
            actions = self.get_action(observation)  # env, horizon, action_dim
            actions = self.x7s_action_mapping(actions)
            for i in range(self.usr_args["num_feedback_actions"]):
                observation, terminated = self.step_environment(task_env, actions[:, i], usr_args)
                self.add_video_frame(video_writer, observation, usr_args['record_camera'])
                if terminated:
                    return terminated
        return terminated

    def reset_model(model):
        pass
