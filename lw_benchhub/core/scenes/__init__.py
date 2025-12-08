import gymnasium as gym

gym.register(
    id="Local-Scene-Usd",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base:LocalScene",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Scene-Usd",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.kitchen.kitchen:LwScene",
    },
    disable_env_checker=True,
)
