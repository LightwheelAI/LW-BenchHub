import gymnasium as gym

gym.register(
    id="Robocasa-Task-PnPOrange",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.pnp_orange:PnPOrange"},
)
