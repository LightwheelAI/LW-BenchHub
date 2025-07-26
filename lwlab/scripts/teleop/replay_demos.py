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
import os
import json
import cv2
import mediapy as media
import tqdm
from itertools import count
import numpy as np
import h5py
from pathlib import Path
from isaaclab.app import AppLauncher
import pinocchio  # noqa: F401

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--width", type=int, default=1920, help="Width of the rendered image.")
parser.add_argument("--height", type=int, default=1080, help="Height of the rendered image.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="/your/path/to/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument("--robot_scale", type=float, default=1.0, help="robot scale")
parser.add_argument("--first_person_view", action="store_true", default=False, help="first person view")
parser.add_argument("--only_last_clips", action="store_true", default=True, help="only replay the last clips")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True


def main():
    """Replay episodes loaded from a file."""
    global is_paused
    # launch the simulator
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # import_all_inits(os.path.join(ISAAC_ROBOCASA_ROOT, './tasks/_APIs'))
    from isaaclab_tasks.utils import import_packages
    # The blacklist is used to prevent importing configs from sub-packages
    _BLACKLIST_PKGS = ["utils", ".mdp"]
    # Import all configs in this package
    import_packages("tasks", _BLACKLIST_PKGS)

    import contextlib
    import gymnasium as gym
    import torch
    from isaaclab.devices import Se3Keyboard
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from lwlab.utils.env import set_camera_follow_pose

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        if args_cli.only_last_clips:
            episode_indices_to_replay = [episode_count - 1]
        else:
            episode_indices_to_replay = list(range(episode_count))

    num_envs = args_cli.num_envs

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])

    if "-" in env_args["env_name"] and not env_args["env_name"].startswith("Robocasa"):
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task
    else:  # robocasa
        from lwlab.utils.env import parse_env_cfg, ExecuteMode
        task_name = env_args["task_name"] if args_cli.task is None else args_cli.task
        env_cfg = parse_env_cfg(
            task_name=task_name,
            robot_name=env_args["robot_name"],
            scene_name=f"robocasakitchen-{env_args['layout_id']}-{env_args['style_id']}",
            robot_scale=args_cli.robot_scale,
            device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
            replay_cfgs={"hdf5_path": args_cli.dataset_file, "ep_meta": env_args, "render_resolution": (args_cli.width, args_cli.height)},
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.EVAL
        )
        env_name = f"Robocasa-{task_name}-{env_args['robot_name']}-v0"
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
    delattr(env_cfg.terminations, "success")

    # create environment from loaded config
    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    # reset before starting
    env.reset()

    # prepare video writer
    replay_mp4_path = Path(args_cli.dataset_file).parent / "isaac_replay_state.mp4"
    replay_mp4_path.parent.mkdir(parents=True, exist_ok=True)
   # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    if len(args_cli.select_episodes) > 0:
        episode_names = [episode_names[i] for i in args_cli.select_episodes]
    replayed_episode_count = 0

    with (
        contextlib.suppress(KeyboardInterrupt) and torch.inference_mode(),
        media.VideoWriter(path=replay_mp4_path, shape=(args_cli.height, len(env.cfg.render_cfgs) * args_cli.width), fps=30) as v,
        h5py.File(replay_mp4_path.parent / f"replay_state_ee_poses.hdf5", "w") as ee_poses_hdf5_f,
    ):
        ee_poses_hdf5_f.create_group("data")
        env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
        env_id = 0
        for i in tqdm.tqdm(episode_indices_to_replay, desc="Replaying"):
            ep = episode_names[i]
            print(f"Replaying episode {ep}")
            env_episode_data_map[env_id] = dataset_file_handler.load_episode(ep, env.device)
            next_state = env_episode_data_map[env_id].get_next_state()
            ee_poses = []
            for _ in tqdm.tqdm(count(), desc=f"Replaying {ep}"):
                if next_state is None:
                    break
                if args_cli.first_person_view:
                    set_camera_follow_pose(env, env_cfg.viewport_cfg["offset"], env_cfg.viewport_cfg["lookat"])
                obs, _ = env.reset_to(next_state, torch.tensor([env_id], device=env.device), is_relative=True if env_args["robot_name"].endswith("Rel") else False)
                env.sim.render()
                next_state = env_episode_data_map[env_id].get_next_state()
                ee_poses.append(obs['policy']['ee_pose'].cpu().numpy())
                if app_launcher._enable_cameras:
                    full_image = np.concatenate([obs['policy'][name].cpu().numpy() for name in env.cfg.render_cfgs.keys()], axis=0).transpose(0, 2, 1, 3)
                    full_image = full_image.reshape(-1, *full_image.shape[2:]).transpose(1, 0, 2)
                    v.add_image(full_image)
                    cv2.imshow("replay", full_image[..., ::-1])
                    cv2.waitKey(1)
            if ee_poses:
                ee_poses = np.concatenate(ee_poses, axis=0)
                # save ee poses to hdf5 file
                ee_poses_hdf5_f.create_dataset(f"data/{ep}/ee_poses", data=ee_poses)
            # np.save(replay_mp4_path.parent / f"ee_poses.npy", ee_poses)
    print(f"Finished replaying {len(episode_names)} episode{'s' if replayed_episode_count > 1 else ''}.")
    env.close()
    simulation_app.close()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    # run the main function
    main()
