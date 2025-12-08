import gymnasium as gym


gym.register(
    id="Robocasa-Scene-Robocasakitchen",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen:LwScene",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Scene-Libero",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.libero:LiberoEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Scene-Testasset",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen:TestAssetScene",
    },
    disable_env_checker=True,
)
