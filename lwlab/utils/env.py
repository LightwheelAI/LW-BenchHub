# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING
import gymnasium as gym
import importlib
import inspect
import os
import enum
import yaml
from importlib.metadata import entry_points

from isaaclab.utils.configclass import configclass
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

from lwlab.utils.log_utils import get_default_logger


def discover_lwlab_modules():
    for ep in entry_points(group="lwlab_modules"):
        yield ep.name, ep.value


def discover_and_import_lwlab_modules():
    from isaaclab_tasks.utils import import_packages

    for name, value in discover_lwlab_modules():
        print(f"Importing {value}")
        import_packages(value)


class ExecuteMode(enum.Enum):
    """The mode to execute the task."""

    TRAIN = 0
    EVAL = 1
    TELEOP = 2
    REPLAY_JOINT_TARGETS = 3
    REPLAY_ACTION = 4
    REPLAY_STATE = 5


def load_cfg_cls_from_registry(cfg_type: str, cfg_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry.

    This function loads the configuration object from the gym registry for the given task name.
    It supports both YAML and Python configuration files.

    It expects the configuration to be registered in the gym registry as:

    .. code-block:: python

        gym.register(
            id="My-Awesome-Task-v0",
            ...
            kwargs={"env_entry_point_cfg": "path.to.config:ConfigClass"},
        )

    The parsed configuration object for above example can be obtained as:

    .. code-block:: python

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = load_cfg_from_registry("My-Awesome-Task-v0", "env_entry_point_cfg")

    Args:
        cfg_name: The name of the environment.
        entry_point_key: The entry point key to resolve the configuration file.

    Returns:
        The parsed configuration object. If the entry point is a YAML file, it is parsed into a dictionary.
        If the entry point is a Python class, it is instantiated and returned.

    Raises:
        ValueError: If the entry point key is not available in the gym registry for the task.
    """
    # obtain the configuration entry point
    if not cfg_name:
        return None
    assert cfg_type in ["scene", "task", "robot", "rl"]
    cfg_name = f"Robocasa-{cfg_type.capitalize()}-{cfg_name}"
    cfg_entry_point = gym.spec(cfg_name).kwargs.get(entry_point_key)
    # check if entry point exists
    if cfg_entry_point is None:
        raise ValueError(
            f"Could not find configuration for the environment: '{cfg_name}'."
            f" Please check that the gym registry has the entry point: '{entry_point_key}'."
        )
    # parse the default config file
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # load the configuration
        get_default_logger().info(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            # resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # load the configuration
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)

        else:
            cfg_cls = cfg_entry_point
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")

        if callable(cfg_cls) and "rsl_rl" in str(cfg_cls):
            cfg = cfg_cls()
        else:
            cfg = cfg_cls
    return cfg


def parse_env_cfg(
    scene_name: str,
    robot_name: str,
    task_name: str,
    robot_scale: float,
    execute_mode: ExecuteMode,
    for_rl: bool = False,
    rl_variant: str = None,
    device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None,
    replay_cfgs: dict | None = None,
    first_person_view: bool = False,
    enable_cameras: bool = False,
    usd_simplify: bool = False,
    object_init_offset: list[float] = [0.0, 0.0],
    max_scene_retry: int = 5,
    max_object_placement_retry: int = 3,
    seed: int | None = None,
    sources: list[str] | None = None,
    object_projects: list[str] | None = None,
    headless_mode: bool = False,
    teleop_device: str = None,
    resample_objects_placement_on_reset: bool | None = None,
    resample_robot_placement_on_reset: bool | None = None,
    **kwargs,
) -> "ManagerBasedRLEnvCfg":
    """Parse configuration for an environment and override based on inputs.

    Args:
        task_name: The name of the environment.
        device: The device to run the simulation on. Defaults to "cuda:0".
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
        use_fabric: Whether to enable/disable fabric interface. If false, all read/write operations go through USD.
            This slows down the simulation but allows seeing the changes in the USD through the USD stage.
            Defaults to None, in which case it is left unchanged.

    Returns:
        The parsed configuration object.

    Raises:
        RuntimeError: If the configuration for the task is not a class. We assume users always use a class for the
            environment configuration.
    """
    # import_all_inits(os.path.join(ISAAC_ROBOCASA_ROOT, './tasks/_APIs'))
    # Import all configs in this package
    from lwlab.core.context import get_context
    context = get_context()
    context.scene_name = scene_name
    context.robot_name = robot_name
    context.task_name = task_name
    context.execute_mode = execute_mode
    context.device = device
    context.robot_scale = robot_scale
    context.first_person_view = first_person_view
    context.enable_cameras = enable_cameras
    context.usd_simplify = usd_simplify
    context.object_init_offset = object_init_offset
    context.max_scene_retry = max_scene_retry
    context.max_object_placement_retry = max_object_placement_retry
    context.seed = seed
    context.sources = sources
    context.object_projects = object_projects
    context.headless_mode = headless_mode
    context.extra_params = kwargs
    context.replay_cfgs = replay_cfgs
    context.resample_objects_placement_on_reset = resample_objects_placement_on_reset
    context.resample_robot_placement_on_reset = resample_robot_placement_on_reset
    discover_and_import_lwlab_modules()

    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment

    if scene_name.endswith(".usd"):
        scene_type = "USD"
    else:
        scene_type, *_ = scene_name.split("-", 1)
    scene = load_cfg_cls_from_registry("scene", "Robocasakitchen", "env_cfg_entry_point")

    if not for_rl:
        if scene_type == "USD":
            task = load_cfg_cls_from_registry("task", "Usd", "env_cfg_entry_point")
        else:
            task = load_cfg_cls_from_registry("task", task_name, "env_cfg_entry_point")
        robot = load_cfg_cls_from_registry("robot", robot_name, "env_cfg_entry_point")
    else:
        # TODO: how to handle rl_variant? not robot here
        if rl_variant:
            task_name = f"{task_name}-{rl_variant}"
        task = load_cfg_cls_from_registry("rl", f"{robot_name}-{task_name}", "env_cfg_entry_point")

    # TODO: how to handle teleop_device, remove it in main?
    # if teleop_device is not None:
    #     teleop_device = environment_base.device_registry.get_device_by_name(teleop_device)()
    # else:
    #     teleop_device = None
    teleop_device = None

    isaac_arena_environment = IsaacArenaEnvironment(
        name=task_name,
        embodiment=robot(),
        scene=scene(),
        task=task(),
        teleop_device=teleop_device,
    )

    # parse the configuration for the isaac arena environment ;now fake here
    # TODO: fix this
    import argparse
    args = {"device": device, "num_envs": num_envs, "disable_fabric": use_fabric, 'mimic': False}
    args = argparse.Namespace(**args)

    from isaac_arena.environments.compile_env import ArenaEnvBuilder
    arena_builder = ArenaEnvBuilder(isaac_arena_environment, args)
    env_name, cfg = arena_builder.build_registered()

    # check that it is not a dict
    # we assume users always use a class for the configuration
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")

    if execute_mode == ExecuteMode.REPLAY_JOINT_TARGETS:
        cfg.set_replay_joint_targets_action()
    # simulation device
    cfg.sim.device = device
    # disable fabric to read/write through USD
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # number of environments
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    if replay_cfgs and "ep_meta" in replay_cfgs and "sim_args" in replay_cfgs["ep_meta"]:
        cfg.sim.dt = replay_cfgs["ep_meta"]["sim_args"]["dt"]
        cfg.sim.decimation = replay_cfgs["ep_meta"]["sim_args"]["decimation"]
        # cfg.sim.render_interval = replay_cfgs["ep_meta"]["sim_args"]["render_interval"]
        # cfg.scene.num_envs = replay_cfgs["ep_meta"]["sim_args"]["num_envs"]
    return cfg
