import gymnasium as gym

gym.register(
    id="Robocasa-Robot-DoublePiper-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_piper:DoublePiperAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-DoublePiper-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_piper:DoublePiperRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-DoublePiper-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_piper:DoublePiperRLEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-Piper-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.piper:PiperAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-Piper-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.piper:PiperRLEnvCfg",
    },
    disable_env_checker=True,
)
