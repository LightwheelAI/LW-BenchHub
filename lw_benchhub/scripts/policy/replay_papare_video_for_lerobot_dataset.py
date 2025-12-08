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
import json
import os
from itertools import count
from pathlib import Path

import mediapy as media
import tqdm

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Replay demonstrations in Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay episodes.")
parser.add_argument("--task", type=str, default=None, help="Force to use the specified task.")
parser.add_argument("--width", type=int, default=1920, help="Width of the rendered image.")
parser.add_argument("--height", type=int, default=1080, help="Height of the rendered image.")
parser.add_argument("--without_image", action="store_true", default=False, help="without image")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--robot_scale", type=float, default=1.0, help="robot scale")
parser.add_argument("--first_person_view", action="store_true", help="first person view")
parser.add_argument("--replay_all_clips", action="store_true", help="replay all clips. If not specified, only replay the last clips")
parser.add_argument("--select_cameras", type=str, nargs="+", default=['left_hand_camera', 'first_person_camera'], help="select cameras to record")
parser.add_argument("--root_path", type=str, default='/home/zsy/Downloads/x7s_3', help="root path of the dataset")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True
if not args_cli.without_image:
    import cv2

global is_paused
# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def make_env_from_hdf5(dataset_file):

    import gymnasium as gym
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from isaaclab.sensors.camera.tiled_camera import TiledCamera
    # Load dataset
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"The dataset file {dataset_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        if not args_cli.replay_all_clips:
            episode_indices_to_replay = [episode_count - 1]
        else:
            episode_indices_to_replay = list(range(0, episode_count))

    num_envs = args_cli.num_envs

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    episode_names_to_replay = []
    for idx in episode_indices_to_replay:
        episode_names_to_replay.append(episode_names[idx])
    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
    usd_simplify = env_args["usd_simplify"] if 'usd_simplify' in env_args else False
    if "-" in env_args["env_name"] and not env_args["env_name"].startswith("Robocasa"):
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task
    else:  # robocasa
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
        task_name = env_args["task_name"] if args_cli.task is None else args_cli.task
        env_cfg = parse_env_cfg(
            task_name=task_name,
            robot_name=env_args["robot_name"],
            scene_name=f"robocasakitchen-{env_args['layout_id']}-{env_args['style_id']}",
            robot_scale=args_cli.robot_scale,
            device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
            replay_cfgs={"hdf5_path": dataset_file, "ep_meta": env_args, "render_resolution": (args_cli.width, args_cli.height), "ep_names": episode_names_to_replay},
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.REPLAY_STATE,
            usd_simplify=usd_simplify,
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
    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    if len(args_cli.select_episodes) > 0:
        episode_names = [episode_names[i] for i in args_cli.select_episodes]
    replayed_episode_count = 0

    # calculate the shape of the video recorder
    num_cameras = sum(env.cfg.task_type in c["tags"] for c in env.cfg.observation_cameras.values())
    if num_cameras > 4:
        # two rows layout: height is twice the original, width is the maximum width of each row
        cameras_per_row = (num_cameras + 1) // 2
        video_height = args_cli.height
        video_width = args_cli.width
    else:
        # single row layout: original calculation
        video_height = args_cli.height
        video_width = args_cli.width

    def convert_tensors_to_serializable(obj):
        """Convert PyTorch tensors to JSON serializable format."""
        if isinstance(obj, dict):
            return {key: convert_tensors_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_to_serializable(item) for item in obj]
        elif hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):  # PyTorch tensor
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj

    for i in tqdm.tqdm(episode_indices_to_replay, desc="Replaying"):

        save_dir = Path(dataset_file).parent / 'replay_results' / f'{episode_names_to_replay[i]}'
        save_dir.mkdir(parents=True, exist_ok=True)
        all_cameras = {name: sensor for name, sensor in env.scene.sensors.items() if isinstance(sensor, TiledCamera)}
        all_cameras_names = all_cameras.keys()
        if args_cli.select_cameras:
            all_cameras_names = [name for name in all_cameras_names if name in args_cli.select_cameras]

        cam_paths = [save_dir / f"{cam_name}.mp4" for cam_name in all_cameras_names]
        video_writers = {cam_path.stem: media.VideoWriter(path=cam_path, shape=(video_height, video_width), fps=30) for cam_path in cam_paths}

        # Open all video writers
        for video_writer in video_writers.values():
            video_writer.__enter__()

        env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
        env_id = 0
        ep = episode_names[i]
        print(f"Replaying episode {ep}")
        env_episode_data_map[env_id] = dataset_file_handler.load_episode(ep, env.device)
        next_state = env_episode_data_map[env_id].get_next_state()
        # ee_poses = []
        step_count = 0

        # randomize_scene_lighting(env, torch.tensor([env_id], device=env.device), intensity_range=(50.0, 800.0), color_variation=0.35, default_intensity=800.0, default_color=(0.75, 0.75, 0.75), asset_cfg=SceneEntityCfg("light"))

        for _ in tqdm.tqdm(count(), desc=f"Replaying {ep}"):
            if next_state is None:
                break
            obs, _ = env.reset_to(next_state, torch.tensor([env_id], device=env.device), is_relative=True if env_args["robot_name"].endswith("Rel") else False)
            env.sim.render()
            next_state = env_episode_data_map[env_id].get_next_state()
            # Collect next_state data instead of printing
            if isinstance(next_state, dict):
                next_state["robot"] = next_state.get("articulation", {}).get("robot", {})
                next_state["robot"]["joint_names"] = env.scene.articulations["robot"].joint_names
            step_count += 1
            if app_launcher._enable_cameras:
                camera_images = {name: obs['policy'][name].cpu().numpy() for name in [n for n, c in env.cfg.observation_cameras.items() if env.cfg.task_type in c["tags"]]}

                for cam_name, video_writer in video_writers.items():
                    video_writer.add_image(camera_images[cam_name].squeeze())

        # Close all video writers after finishing this episode
        for video_writer in video_writers.values():
            video_writer.__exit__(None, None, None)

        replayed_episode_count += 1
    dataset_file_handler.close()
    print(f"Finished replaying {len(episode_names)} episode{'s' if replayed_episode_count > 1 else ''}.")
    return env


def main():
    root_path = args_cli.root_path
    import subprocess
    import sys

    if os.path.isfile(root_path):
        env = make_env_from_hdf5(root_path)
        env.close()
        simulation_app.close()
        return

    clip_names = os.listdir(root_path)
    dataset_files = [os.path.join(root_path, clip_name, 'dataset_success.hdf5') for clip_name in clip_names]

    for i, dataset_file in enumerate(dataset_files):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(dataset_files)}: {dataset_file}")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            __file__,
            "--root_path", dataset_file,
            "--headless",
            "--num_envs", "1",
            "--width", "480",
            "--height", "480",
            "--enable_cameras",
            "--device", "cpu",
        ]
        cmd = [c for c in cmd if c]

        result = subprocess.run(cmd)
        print(f"{'✓' if result.returncode == 0 else '✗'} Done")

    print("\nAll datasets processed!")
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
