import gymnasium as gym

gym.register(
    id="Robocasa-Robot-Panda",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka:FrankaAbsEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Robocasa-Robot-Panda-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka:FrankaRLEnvCfg",
    },
    disable_env_checker=True,
)
