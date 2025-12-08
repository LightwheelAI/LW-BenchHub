import gymnasium as gym

gym.register(
    id="Robocasa-Robot-X7S-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7s:X7SRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-X7S-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7s:X7SAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-X7S-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.x7s:X7SJointEnvCfg",
    },
    disable_env_checker=True,
)
