import gymnasium as gym

gym.register(
    id="Robocasa-Robot-PandaOmron-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pandaomron:PandaOmronRelEmbodiment",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-PandaOmron-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pandaomron:PandaOmronAbsEmbodiment",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-PandaOmron-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pandaomron:PandaOmronRLEmbodiment",
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
