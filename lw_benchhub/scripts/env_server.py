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

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
import random
from functools import partial

import gymnasium as gym

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--remote_protocol", type=str, default="ipc", help="Remote protocol, can be ipc or restful")
parser.add_argument("--ipc_host", type=str, default="127.0.0.1", help="IPC host")
parser.add_argument("--ipc_port", type=int, default=50000, help="IPC port")
parser.add_argument("--ipc_authkey", type=str, default="lightwheel", help="IPC authkey")
parser.add_argument("--restful_host", type=str, default="0.0.0.0", help="Restful host")
parser.add_argument("--restful_port", type=int, default=8000, help="Restful port")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

if args_cli.remote_protocol == "restful":
    from lw_benchhub.distributed.restful import RestfulEnvWrapper
    RemoteEnvWrapper = partial(RestfulEnvWrapper, address=(args_cli.restful_host, args_cli.restful_port))
elif args_cli.remote_protocol == "ipc":   # ipc
    from lw_benchhub.distributed.ipc import IpcDistributedEnvWrapper
    RemoteEnvWrapper = partial(IpcDistributedEnvWrapper, address=(args_cli.ipc_host, args_cli.ipc_port), authkey=args_cli.ipc_authkey.encode())


app_launcher_args = vars(args_cli)
app_launcher = None
simulation_app = None


def make_env_cfg(cfg):
    from isaaclab_tasks.utils import parse_env_cfg

    if "-" in cfg.task:
        env_cfg = parse_env_cfg(
            cfg.task, device=cfg.device, num_envs=cfg.num_envs, use_fabric=not cfg.disable_fabric
        )
        task_name = cfg.task
    else:
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode, str_to_execute_mode

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
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=str_to_execute_mode(cfg.execute_mode),
            headless_mode=args_cli.headless,
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
    # multi-gpu training config
    if cfg.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # randomly sample a seed if seed = -1
    if cfg.seed == -1:
        cfg.seed = random.randint(0, 10000)
    return task_name, env_cfg


def make_env(cfg, launcher_args, args_override: dict = None):
    global app_launcher
    global simulation_app
    if app_launcher is None:
        args_override = args_override or {}
        app_launcher_args_ = {**launcher_args, **args_override}
        app_launcher = AppLauncher(app_launcher_args_)
        simulation_app = app_launcher.app
    from isaaclab.envs import ManagerBasedEnv
    from lw_benchhub.utils.place_utils.env_utils import warmup_rendering
    task_name, env_cfg = make_env_cfg(cfg)
    gym_env = gym.make(
        task_name,
        cfg=env_cfg,
        render_mode="rgb_array" if cfg.video else None
    )
    env: ManagerBasedEnv = gym_env.unwrapped
    warmup_rendering(env)
    return env


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    with RemoteEnvWrapper(env_initializer=partial(make_env, launcher_args=app_launcher_args)) as env_server:
        env_server.serve()


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
