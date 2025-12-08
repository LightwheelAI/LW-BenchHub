import gymnasium as gym

gym.register(
    id="Robocasa-Robot-G1-WBC-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arena:G1ArenaJointEmbodiment",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-G1-WBC-Pink",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arena:G1ArenaPinkEmbodiment",
    },
    disable_env_checker=True,
)
