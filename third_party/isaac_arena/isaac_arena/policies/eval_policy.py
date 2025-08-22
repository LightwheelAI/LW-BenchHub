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

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
from pathlib import Path
import mediapy as media
import tqdm
from itertools import count

import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from isaaclab.app import AppLauncher

from lwlab.utils.isaaclab_utils import get_robot_joint_target_from_scene
from lwlab.utils.config_loader import config_loader
# add argparse arguments
parser = argparse.ArgumentParser(description="Eval policy in Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--task_config", type=str, default=None, help="task config")
parser.add_argument("--width", type=int, default=1920, help="Width of the rendered image.")
parser.add_argument("--height", type=int, default=1080, help="Height of the rendered image.")
parser.add_argument("--layout", type=str, default=None, help="layout name")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--overrides", nargs=argparse.REMAINDER)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)
import yaml
with open(f"configs/policy/{args_cli.task_config}.yml", 'r') as file:
    cam_config = yaml.safe_load(file)

args_cli.enable_cameras = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch
from datetime import datetime
from isaaclab.devices import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


import importlib
import sys
import yaml
sys.path.append(f"./")
sys.path.append(f"./policy")


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def parse_args_and_config():

    with open(args_cli.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args_cli.overrides:
        overrides = parse_override_pairs(args_cli.overrides)
        config.update(overrides)

    return config


def main(usr_args):
    """Replay episodes loaded from a file."""
    global is_paused
    from isaaclab.envs import ManagerBasedRLEnv

    # import_all_inits(os.path.join(ISAAC_ROBOCASA_ROOT, './tasks/_APIs'))
    from isaaclab_tasks.utils import import_packages
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages("tasks", _BLACKLIST_PKGS)

    # Load dataset

    num_envs = args_cli.num_envs

    if "-" in args_cli.task:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task
    else:  # robocasa
        from lwlab.utils.env import parse_env_cfg, ExecuteMode
        task_name = args_cli.task if args_cli.task is None else args_cli.task
        env_cfg = parse_env_cfg(
            task_name=task_name,
            robot_name=args_cli.robot,
            scene_name=args_cli.layout,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
            replay_cfgs={"render_resolution": (args_cli.width, args_cli.height)},
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.REPLAY_JOINT_TARGETS,
            usd_simplify=args_cli.usd_simplify,
        )

        env_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"
        gym.register(
            id=env_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                # "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
            },
            disable_env_checker=True,
        )

    # Disable all recorders and terminations
    env_cfg.recorders = {}
    delattr(env_cfg.terminations, "time_out")

    # create environment from loaded config
    # env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped
    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    policy_name = usr_args["policy_name"]
    from isaac_arena.policies.GR00T.deploy_policy import get_model, eval, reset_model

    # get_model = eval_function_decorator(policy_name, "get_model")
    # eval_func = eval_function_decorator(policy_name, "eval")
    # reset_func = eval_function_decorator(policy_name, "reset_model")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    # save_dir.mkdir(parents=True, exist_ok=True)
    usr_args['actions_dim'] = env.action_space.shape[1]
    usr_args['decimation'] = env_cfg.decimation
    usr_args.update(cam_config)
    # prepare video writer and json file
    has_success = False
    model = get_model(usr_args)
    test_num = 20
    suc_num = 0
    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for idx in tqdm.tqdm(range(test_num)):
            eval_video_path = Path(f"./eval_result/episode{idx}.mp4")
            eval_video_path.parent.mkdir(parents=True, exist_ok=True)
            with media.VideoWriter(path=eval_video_path, shape=(args_cli.height, args_cli.width), fps=30) as v:
                obs, _ = env.reset()
                reset_model(model)
                # Get idle action (idle actions are applied to envs without next action)
                idle_action = torch.zeros(env.action_space.shape)
                step = 0
                step_limit = 50
                observation = {}
                observation["observation"] = {
                    "head_camera": {
                        "rgb": obs['policy'][usr_args['head_camera']].cpu().numpy()[0]
                    },
                    "left_camera": {
                        "rgb": obs['policy'][usr_args['left_camera']].cpu().numpy()[0]
                    },
                    "right_camera": {
                        "rgb": obs['policy'][usr_args['right_camera']].cpu().numpy()[0]
                    }
                }
                observation["joint_action"] = {
                    "vector": obs['policy']['joint_pos'][0].cpu().numpy()
                }
                v.add_image(obs['policy'][usr_args['head_camera']].cpu().numpy()[0])
                while simulation_app.is_running() and not simulation_app.is_exiting() and step < step_limit:
                    # initialize actions with idle action so those without next action will not move
                    has_success = eval(env, model, observation, usr_args, v)
                    step += 1
                    if has_success:
                        suc_num += 1
                        break
    env.close()


if __name__ == "__main__":
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
    # close sim app
    simulation_app.close()
