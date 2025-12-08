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

import gymnasium as gym


def register_rl_env(robot_name, task_name, env_cfg_entry_point, skrl_cfg_entry_point, rsl_rl_cfg_entry_point, variant=None):
    if variant:
        task_name = f"{task_name}-{variant}"
    gym.register(
        id=f"Robocasa-Rl-{robot_name}-{task_name}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "skrl_cfg_entry_point": skrl_cfg_entry_point,
            "rsl_rl_cfg_entry_point": rsl_rl_cfg_entry_point,
        },
        disable_env_checker=True,
    )
