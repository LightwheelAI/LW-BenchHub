import gymnasium as gym

gym.register(
    id="Robocasa-Robot-G1-Hand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree:UnitreeG1HandEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-Loco-Hand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unitree_locomotion:UnitreeG1LocoHandEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1rl:UnitreeG1HandEnvRLCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-PandaOmron-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.compositional:PandaOmronRelEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-PandaOmron-Abs",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.compositional:PandaOmronAbsEnvCfg",
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
