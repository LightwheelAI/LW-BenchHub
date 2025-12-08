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

import argparse

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import (
    IsaacLabArenaManagerBasedRLEnvCfg,
)
from isaaclab_arena.utils.configclass import combine_configclass_instances

from lw_benchhub.core.rl.base import LwRL


class LwEnvBuilder(ArenaEnvBuilder):

    DEFAULT_SCENE_CFG = InteractiveSceneCfg(num_envs=4096, env_spacing=30.0)

    def __init__(self, arena_env: IsaacLabArenaEnvironment, args: argparse.Namespace, rl_env: LwRL):
        super().__init__(arena_env, args)
        self.rl_env = rl_env

    def orchestrate(self) -> None:
        super().orchestrate()
        if self.rl_env:
            self.rl_env.setup_env_config(self.arena_env.orchestrator)

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg = super().modify_env_cfg(env_cfg)
        if self.rl_env:
            env_cfg = self.rl_env.modify_env_cfg(env_cfg)
        return env_cfg

    def compose_manager_cfg(self) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg = super().compose_manager_cfg()
        policy_observation_cfg = combine_configclass_instances(
            "PolicyObservationGroupCfg",
            self.arena_env.embodiment.get_policy_observation_cfg(),
            self.arena_env.task.get_policy_observation_cfg(),
            self.rl_env.get_policy_observation_cfg() if self.rl_env else None,
        )

        new_policy = type('PolicyObservationGroupCfg', (ObsGroup,), {})()
        for key, value in policy_observation_cfg.__dict__.items():
            if not key.startswith('_'):
                setattr(new_policy, key, value)
        env_cfg.observations.policy = new_policy
        return env_cfg
