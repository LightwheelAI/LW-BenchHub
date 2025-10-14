import gymnasium as gym

gym.register(
    id="Robocasa-Robot-PandaOmron-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pandaomron:PandaOmronRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-PandaOmron-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pandaomron:PandaOmronAbsEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-PandaOmron-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pandaomron:PandaOmronRLEnvCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="Robocasa-Robot-DoublePanda-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_panda:DoublePandaRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-DoublePanda-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.double_panda:DoublePandaAbsEnvCfg",
    },
    disable_env_checker=True,
)
