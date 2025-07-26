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

gym.register(
    id="Robocasa-Task-OpenDrawer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_drawer:OpenDrawer",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseDrawer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_drawer:CloseDrawer",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseElectricKettleLid",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_electric_kettle:CloseElectricKettleLid",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-OpenElectricKettleLid",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_electric_kettle:OpenElectricKettleLid",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-OpenOven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:OpenOven",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseOven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:CloseOven",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-OpenToasterOvenDoor",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:OpenToasterOvenDoor",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseToasterOvenDoor",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:CloseToasterOvenDoor",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-OpenDishwasher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:OpenDishwasher",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseDishwasher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:CloseDishwasher",
    },
    disable_env_checker=True,
)
