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
import json
import os
from dataclasses import dataclass, field

from tqdm import tqdm

from isaaclab.app import AppLauncher

from lw_benchhub.utils.config_loader import config_loader

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default="default", help="task config")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)

app_launcher_args = vars(args_cli)
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)

simulation_app = app_launcher.app

args_cli.device = f"cuda:0"


def main(args):
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    import gymnasium as gym
    import torch
    import numpy as np
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg
    from lw_benchhub.utils.place_utils.env_utils import set_seed
    from lw_benchhub.core.context import get_context
    from lw_benchhub.utils.render_utils import optimize_rendering

    if "-" in args_cli.task:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task

    else:  # robocasa
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        env_cfg = parse_env_cfg(
            scene_backend=args_cli.scene_backend,
            task_backend=args_cli.task_backend,
            task_name=args_cli.task,
            robot_name=args_cli.robot,
            scene_name=args_cli.layout,
            rl_name=args_cli.rl,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.TRAIN,
            usd_simplify=args_cli.usd_simplify,
            rl_variant=args_cli.variant,
            seed=args_cli.seed,
            sources=args_cli.sources,
            object_projects=args_cli.object_projects,
            headless_mode=args_cli.headless,
        )
        task_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",  # lw_benchhub.enhance.envs:ManagerBasedRLDigitalTwinEnv
            kwargs={
            },
            disable_env_checker=True,
        )

    env_cfg.observations.policy.concatenate_terms = False
    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg)  # .unwrapped
    set_seed(env_cfg.seed, env.unwrapped, args.torch_deterministic)

    if args_cli.enable_rendering_optimization:
        optimize_rendering(env.unwrapped, args_cli)

    from policy.maniskill_ppo.agent import PPOArgs, PPO, observation

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env.reset()
    next_obs, _ = env.reset()
    next_obs = observation(next_obs['policy'])
    args.num_envs = args_cli.num_envs
    agent = PPO(env, next_obs, args, env_cfg.sim.device)
    if args_cli.checkpoint:
        agent.load_model(args_cli.checkpoint)
    next_done = torch.zeros(args_cli.num_envs, device=env_cfg.sim.device)
    num_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent.num_iterations

    pbar = tqdm(range(1, num_iterations + 1), desc="Training Progress")
    for iteration in pbar:
        if args.save_model and iteration % args.save_model_freq == 1:
            agent.save_model(iteration)
        if args.anneal_lr:
            agent.anneal_lr(iteration)
        next_obs, next_done = agent.collect_data(next_obs, next_done)

        agent.update(next_obs, next_done)
    if args.save_model:
        checkpoint_path = agent.save_model(iteration)

    if hasattr(args_cli, "check_success") and args_cli.check_success:
        get_context().execute_mode = ExecuteMode.EVAL
        next_obs, _ = env.reset()
        next_obs = observation(next_obs['policy'])
        eval_iter = 640
        success_count = 0
        episode_count = 0
        with torch.inference_mode():
            for _ in tqdm(range(eval_iter), desc="Evaluation Progress"):
                action = agent.agent.get_action(next_obs, deterministic=True)
                # action = torch.zeros_like(action, device=env_cfg.sim.device)
                next_obs, _, terminations, _, extras = env.step(action)
                next_obs = observation(next_obs['policy'])
                success_count += extras["is_success"].sum().item()
                episode_count += terminations.sum().item()

            success_rate = success_count / (episode_count + 1e-8)
            parent_dir = os.path.dirname(checkpoint_path)
            result_path = os.path.join(parent_dir, "result.json")
            print(f"success_rate: {success_rate}")
            with open(result_path, "w") as f:
                json.dump({"success_rate": success_rate, "ci_success": success_rate >= 0.7}, f)

    # close the simulator
    env.close()
    simulation_app.close()


from policy.maniskill_ppo.agent import PPOArgs


@dataclass
class Args:
    # env_id: str
    # """The environment id to train on"""
    # env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)


if __name__ == "__main__":
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    args = Args()
    main(args.ppo)
