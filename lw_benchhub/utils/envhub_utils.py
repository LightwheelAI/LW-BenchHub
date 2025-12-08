import yaml
import random
import gymnasium as gym


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = {
        "scene_backend": "robocasa",
        "task_backend": "robocasa",
        "device": "cuda:0",
        "robot_scale": 1.0,
        "first_person_view": False,
        "disable_fabric": False,
        "num_envs": 1,
        "usd_simplify": False,
        "video": False,
        "for_rl": False,
        "variant": "Visual",
        "concatenate_terms": False,
        "distributed": False,
        "seed": 42,
        "sources": None,
        "object_projects": None,
        "execute_mode": "eval",
        "replay_cfgs": {"add_camera_to_observation": True},
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    return DotDict(config)


def make_env_cfg(cfg):
    from isaaclab_tasks.utils import parse_env_cfg

    if "-" in cfg.task:
        env_cfg = parse_env_cfg(
            cfg.task, device=cfg.device, num_envs=cfg.num_envs, use_fabric=not cfg.disable_fabric
        )
        task_name = cfg.task
    else:  # robocasa
        from lw_benchhub.utils.env import parse_env_cfg, str_to_execute_mode

        env_cfg = parse_env_cfg(
            scene_backend=cfg.scene_backend,
            task_backend=cfg.task_backend,
            task_name=cfg.task,
            robot_name=cfg.robot,
            scene_name=cfg.layout,
            rl_name=cfg.rl,
            robot_scale=cfg.robot_scale,
            device=cfg.device,
            num_envs=cfg.num_envs,
            use_fabric=not cfg.disable_fabric,
            first_person_view=cfg.first_person_view,
            enable_cameras=cfg.enable_cameras,
            execute_mode=str_to_execute_mode(cfg.execute_mode),
            headless_mode=cfg.headless,
            usd_simplify=cfg.usd_simplify,
            seed=cfg.seed,
            sources=cfg.sources,
            object_projects=cfg.object_projects,
            replay_cfgs=cfg.replay_cfgs
        )
        task_name = f"Robocasa-{cfg.task}-{cfg.robot}-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True
        )

    env_cfg.observations.policy.concatenate_terms = cfg.concatenate_terms
    # modify configuration
    env_cfg.terminations.time_out = None
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = cfg.num_envs if cfg.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = cfg.device if cfg.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if cfg.seed == -1:
        cfg.seed = random.randint(0, 10000)
    return task_name, env_cfg


def export_env(config_path: str):

    cfg = parse_config(config_path)

    from isaaclab.app import AppLauncher
    _ = AppLauncher(enable_cameras=cfg.enable_cameras)

    task_name, env_cfg = make_env_cfg(cfg)
    gym_env = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode="rgb_array" if cfg.video else None
    )
    return gym_env
