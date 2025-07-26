import gymnasium as gym


gym.register(
    id="Robocasa-Scene-Robocasakitchen",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen:RobocasaKitchenEnvCfg",
    },
    disable_env_checker=True,
)
