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

from lwlab.core.rl.base import register_rl_env
from .open_drawer import agents as open_drawer_agents
from .lift_obj import agents as lift_obj_agents

register_rl_env(
    robot_name="G1-RL",
    task_name="OpenDrawer",
    env_cfg_entry_point=f"{__name__}.open_drawer.open_drawer:OpenDrawerG1RlCfg",
    skrl_cfg_entry_point=f"{open_drawer_agents.__name__}:skrl_ppo_cfg.yaml",
)

register_rl_env(
    robot_name="G1-RL",
    task_name="LiftObj",
    variant="State",
    env_cfg_entry_point=f"{__name__}.lift_obj.lift_obj:G1StateLiftObjRLEnvCfg",
    skrl_cfg_entry_point=f"{lift_obj_agents.__name__}:skrl_ppo_cfg.yaml",
)

register_rl_env(
    robot_name="G1-RL",
    task_name="LiftObj",
    variant="Visual",
    env_cfg_entry_point=f"{__name__}.lift_obj.lift_obj:G1VisualLiftObjRLEnvCfg",
    skrl_cfg_entry_point=f"{lift_obj_agents.__name__}:skrl_ppo_cfg.yaml",
)

register_rl_env(
    robot_name="Lerobot-RL",
    task_name="LiftObj",
    variant="State",
    env_cfg_entry_point=f"{__name__}.lift_obj.lift_obj:LerobotStateLiftObjRLEnvCfg",
    skrl_cfg_entry_point=f"{lift_obj_agents.__name__}:skrl_ppo_cfg.yaml",
)

register_rl_env(
    robot_name="Lerobot-RL",
    task_name="LiftObj",
    variant="Visual",
    env_cfg_entry_point=f"{__name__}.lift_obj.lift_obj:LerobotVisualLiftObjRLEnvCfg",
    skrl_cfg_entry_point=f"{lift_obj_agents.__name__}:skrl_ppo_cfg.yaml",
)
