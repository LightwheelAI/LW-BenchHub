import gymnasium as gym

gym.register(
    id="Robocasa-Robot-LeRobot-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lerobotrl:LeRobotRL",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-LeRobot-AbsJointGripper-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lerobotrl:LeRobotAbsJointGripperRL",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-LeRobot-BiARM-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lerobotrl_biarm:LeRobotBiArmRL",
    },
    disable_env_checker=True,
)

gym.register(
    id="Robocasa-Robot-LeRobot100-RL",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lerobotrl:LeRobot100RL",
    },
    disable_env_checker=True,
)
