import gymnasium as gym

gym.register(
    id="Robocasa-Task-Usd",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_usd:BaseUsdTaskCfg",
    },
    disable_env_checker=True,
)
