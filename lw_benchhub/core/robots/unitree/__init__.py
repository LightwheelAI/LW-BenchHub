
import gymnasium as gym

gym.register(
    id="Robocasa-Robot-G1-Hand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1HandEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-Controller",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1ControllerEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-Loco-Hand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1LocoHandEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-Loco-Controller",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1LocoControllerEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1HandEnvRLCfg",
    },
    disable_env_checker=True,
)

gym.register(

    id="Robocasa-Robot-G1-FullHand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1FullHandRLEnvRLCfg",
    },
    disable_env_checker=True,
)

gym.register(

    id="Robocasa-Robot-G1-Controller-DecoupledWBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1:UnitreeG1ControllerDecoupledWBCEnvCfg",
    },
    disable_env_checker=True,
)
