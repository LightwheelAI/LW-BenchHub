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

"""
PI Policy Implementation
"""

import torch
import sys
import os
from typing import Dict, Any

# Add current directory to path
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)
from policy.base import BasePolicy
try:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except ImportError as e:
    print(f"PI not found, please install PI first: {e} , if you not run pi policy, please ignore this error")


class PIPolicy(BasePolicy):
    """PI Policy Implementation"""

    def __init__(self, usr_args: Dict[str, Any]):
        super().__init__(usr_args)

    def get_model(self, usr_args: Dict[str, Any]):
        """Get PI model instance"""

        train_config_name = usr_args["train_config_name"]
        checkpoint = usr_args["checkpoint"]
        observation_config = usr_args.get("observation_config", None)
        config = _config.get_config(train_config_name)
        self.model = _policy_config.create_trained_policy(config, checkpoint)
        print("loading PI0.5 model success!")
        self.observation_window = None
        self.observation_config = observation_config or {}

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.model.infer(self.observation_window)["actions"]

    def encode_obs(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Encode observation"""

        observation = super().encode_obs(observation)
        observation = self._build_observation_window(observation)
        return observation

    def _build_observation_window(self, obs):
        custom_mapping = self.observation_config.get("custom_mapping", {})
        obs_window = {"prompt": self.instruction}
        for key, mapping in custom_mapping.items():
            if isinstance(mapping, dict):
                obs_window[key] = {k: obs[v] for k, v in mapping.items()}
            else:
                obs_window[key] = obs[mapping]
        return obs_window

    def piper_action_mapping(self, action: torch.Tensor) -> torch.Tensor:
        action[..., -1] = -action[..., -1]
        return action

    def eval(self, task_env: Any, observation: Dict[str, Any],
             usr_args: Dict[str, Any], video_writer: Any) -> bool:
        """Evaluate PI policy"""
        for _ in range(usr_args['time_out_limit']):
            self.observation_window = self.encode_obs(observation)
            actions = self.get_action()
            chunk = actions.shape[0]
            actions = torch.from_numpy(actions).float().cuda()
            actions = self.piper_action_mapping(actions)
            for i in range(chunk):
                action = actions[i]
                observation, terminated = self.step_environment(task_env, action.unsqueeze(0), usr_args)
                self.add_video_frame(video_writer, observation, usr_args['record_camera'])
                if terminated:
                    return terminated
        return terminated

    def reset_model(self) -> None:
        """Reset PI model state"""
        self._reset_obsrvationwindows()

    def _reset_obsrvationwindows(self):
        self.observation_window = None
        print("successfully unset observation window")
