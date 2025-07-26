import gymnasium as gym

gym.register(
    id="Robocasa-Scene-Usd",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base:BaseSceneEnvCfg",
    },
    disable_env_checker=True,
)
