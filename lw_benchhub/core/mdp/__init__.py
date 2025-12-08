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

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .actions import *  # noqa: F401, F403


_reset_scene_to_default = reset_scene_to_default  # noqa: F405

# override the default reset_scene_to_default


def reset_scene_to_default(env, env_ids):
    _reset_scene_to_default(env, env_ids)
    env.cfg.isaaclab_arena_env.orchestrator._reset_internal(env, env_ids)
    if hasattr(env.cfg.isaaclab_arena_env.embodiment, 'reset_robot_cfg_state'):
        env.cfg.isaaclab_arena_env.embodiment.reset_robot_cfg_state()
