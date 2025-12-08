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
    id="Robocasa-Task-PnPCounterToCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToCabinet",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCabinetToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCabinetToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCounterToSink",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToSink",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPSinkToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPSinkToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCounterToMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToMicrowave",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPMicrowaveToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPMicrowaveToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCounterToOven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToOven",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPOvenToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPOvenToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCounterToStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPStoveToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPStoveToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPToasterToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPToasterToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCounterToToasterOven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToToasterOven",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPToasterOvenToCounter",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPToasterOvenToCounter",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PnPCounterToStandMixer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_pnp:PnPCounterToStandMixer",
    },
    disable_env_checker=True,
)


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
    id="Robocasa-Task-SlideDishwasherRack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_drawer:SlideDishwasherRack",
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
    id="Robocasa-Task-TurnOnElectricKettle",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_electric_kettle:TurnOnElectricKettle",
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
    id="Robocasa-Task-OpenMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:OpenMicrowave",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:CloseMicrowave",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-OpenFridge",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:OpenFridge",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseFridge",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:CloseFridge",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:CloseCabinet",
    },
    disable_env_checker=True,
)


gym.register(
    id="Robocasa-Task-OpenCabinet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_doors:OpenCabinet",
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

gym.register(
    id="Robocasa-Task-StartCoffeeMachine",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_coffee:StartCoffeeMachine",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CoffeeServeMug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_coffee:CoffeeServeMug",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CoffeeSetupMug",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_coffee:CoffeeSetupMug",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CoffeeSetupMug-Mimic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_coffee:CoffeeSetupMugMimic",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOnMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_microwave:TurnOnMicrowave",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOffMicrowave",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_microwave:TurnOffMicrowave",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOnSinkFaucet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_sink:TurnOnSinkFaucet",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOffSinkFaucet",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_sink:TurnOffSinkFaucet",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnSinkSpout",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_sink:TurnSinkSpout",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-OpenStandMixerHead",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_stand_mixer:OpenStandMixerHead",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-CloseStandMixerHead",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_stand_mixer:CloseStandMixerHead",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOnStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_stove:TurnOnStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOffStove",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_stove:TurnOffStove",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-AdjustToasterOvenTemperature",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_toaster_oven:AdjustToasterOvenTemperature",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOnToasterOven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_toaster_oven:TurnOnToasterOven",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-SlideToasterOvenRack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_toaster_oven:SlideToasterOvenRack",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-TurnOnToaster",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_toaster:TurnOnToaster",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-NavigateKitchen",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_navigate:NavigateKitchen",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-PreheatOven",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_oven:PreheatOven",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-SlideOvenRack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen_oven:SlideOvenRack",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Task-LiftObj",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_obj:LiftObj",
    },
    disable_env_checker=True,
)
