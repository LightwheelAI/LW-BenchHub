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

import h5py
import numpy as np
import pinocchio  # noqa: F401
import tqdm

from isaaclab.app import AppLauncher

from lw_benchhub.utils.video_recorder import VideoProcessor, calculate_camera_layout, get_video_duration

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
parser.add_argument("--dataset_file", type=str, default="/your/path/to/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument("--robot_scale", type=float, default=1.0, help="robot scale")
parser.add_argument("--first_person_view", action="store_true", help="first person view")
parser.add_argument("--replay_all_clips", action="store_true", help="replay all clips. If not specified, only replay the last clips")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True
if not args_cli.without_image:
    import cv2


def check_terminations(env):
    # -- check terminations
    env.reset_buf = env.termination_manager.compute()
    env.reset_time_outs = env.termination_manager.time_outs
    env.extras["is_success"] = env.termination_manager._terminated_buf


def main():
    """Replay episodes loaded from a file."""
    global is_paused  # noqa: F824
    # launch the simulator
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import contextlib
    import gymnasium as gym
    import torch
    from isaaclab.devices import Se3Keyboard
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from lw_benchhub.utils.place_utils.env_utils import set_camera_follow_pose
    from lw_benchhub.utils.place_utils.env_utils import set_seed

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
        if not args_cli.replay_all_clips:
            episode_indices_to_replay = [episode_count - 1]
        else:
            episode_indices_to_replay = list(range(episode_count))

    num_envs = args_cli.num_envs

    # prepare video writer
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    replayed_episode_count = 0

    episode_names_to_replay = []
    for idx in episode_indices_to_replay:
        episode_names_to_replay.append(episode_names[idx])

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
    if "LW_API_ENDPOINT" in env_args.keys():
        os.environ["LW_API_ENDPOINT"] = env_args["LW_API_ENDPOINT"]
    usd_simplify = env_args["usd_simplify"] if 'usd_simplify' in env_args else False
    if "-" in env_args["env_name"] and not env_args["env_name"].startswith("Robocasa"):
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task.strip()
    else:  # robocasa
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
        scene_backend = env_args["scene_backend"] if "scene_backend" in env_args else "robocasa"
        task_backend = env_args["task_backend"] if "task_backend" in env_args else "robocasa"
        task_name = env_args["task_name"] if args_cli.task is None else args_cli.task.strip()
        # TODO delete the hardcoded task names
        if task_name == "PutButterInBasket":
            if "PutButterInBasket2" in env_args["env_name"]:
                task_name = "PutButterInBasket2"
        if task_name == "Libero90PutBlackBowlOnPlate":
            if "Libero90PutBlackBowlOnCabinet" in env_args["env_name"]:
                task_name = "Libero90PutBlackBowlOnCabinet"
        if task_name == "PickBowlOPickBowlOnCabinetPlaceOnPlatenStovePlaceOnPlate":
            task_name = "PickBowlOnCabinetPlaceOnPlate"
        robot_name = env_args["robot_name"]
        if robot_name == "double_piper_abs":
            robot_name = "DoublePiper-Abs"
        if robot_name == "double_piper_rel":
            robot_name = "DoublePiper-Rel"
        if "scene_type" in env_args:
            scene_type = env_args["scene_type"]
        else:
            if "libero" in env_args["usd_path"]:
                scene_type = "libero"
            else:
                scene_type = "robocasakitchen"
        scene_name = f"{scene_type}-{env_args['layout_id']}-{env_args['style_id']}"
        env_cfg = parse_env_cfg(
            scene_backend=scene_backend,
            task_backend=task_backend,
            task_name=task_name,
            robot_name=robot_name,
            scene_name=scene_name,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
            replay_cfgs={"hdf5_path": args_cli.dataset_file, "ep_meta": env_args, "render_resolution": (args_cli.width, args_cli.height), "ep_names": episode_names_to_replay, "add_camera_to_observation": True},
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.REPLAY_STATE,
            usd_simplify=usd_simplify,
            seed=env_args["seed"] if "seed" in env_args else None,
            sources=env_args["sources"] if "sources" in env_args else None,
            object_projects=env_args["object_projects"] if "object_projects" in env_args else None,
            headless_mode=args_cli.headless,
        )
        env_name = f"Robocasa-{task_name}-{robot_name}-v0"
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
    # delattr(env_cfg.terminations, "success")

    # create environment from loaded config
    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped
    set_seed(env_cfg.seed, env)

    # reset before starting
    import carb
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)

    env.reset()

    # prepare video writer
    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    if len(args_cli.select_episodes) > 0:
        episode_names = [episode_names[i] for i in args_cli.select_episodes]
    replayed_episode_count = 0
    # Initialize video processor
    video_processor = None

    # Create a single video writer for all episodes
    if args_cli.replay_all_clips:
        save_dir = Path(args_cli.dataset_file).parent / 'replay_results'
    else:
        save_dir = Path(args_cli.dataset_file).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    replay_mp4_path = save_dir / "isaac_replay_state.mp4"

    # Initialize video processor (manages VideoWriter internally)
    if app_launcher._enable_cameras:
        # calculate the shape of the video recorder
        num_cameras = sum(env.cfg.isaaclab_arena_env.task.context.execute_mode in c["execute_mode"] for c in env.cfg.isaaclab_arena_env.embodiment.observation_cameras.values())
        max_cameras_per_row = 4  # Maximum cameras per row
        # Calculate layout using shared function
        num_rows, cameras_per_row_list, max_cameras_in_row = calculate_camera_layout(
            num_cameras, max_cameras_per_row
        )

        # Print layout design
        print(f"Video grid layout -> rows: {num_rows}, per_row: {cameras_per_row_list} total cameras: {num_cameras}")
        # Calculate video dimensions
        video_height = args_cli.height * num_rows
        video_width = max_cameras_in_row * args_cli.width

        # Calculate product video dimensions
        product_camera_names = [n for n, c in env.cfg.isaaclab_arena_env.embodiment.observation_cameras.items() if "product" in c.get("tags", [])]
        product_mp4_path = None
        if product_camera_names:
            product_num_cameras = len(product_camera_names)
            max_cameras_per_row = 4  # Maximum cameras per row
            # Calculate layout using shared function
            product_num_rows, product_cameras_per_row_list, product_max_cameras_in_row = calculate_camera_layout(
                product_num_cameras, max_cameras_per_row
            )

            # Print product layout design
            print(f"Product video grid layout -> rows: {product_num_rows}, per_row: {product_cameras_per_row_list} total cameras: {product_num_cameras}")

            # Calculate product video dimensions
            product_video_height = args_cli.height * product_num_rows
            product_video_width = product_max_cameras_in_row * args_cli.width
            product_mp4_path = save_dir / "isaac_replay_product.mp4"
        else:
            # No product cameras, use regular video dimensions
            product_video_height = 0
            product_video_width = 0

        video_processor = VideoProcessor(
            replay_mp4_path,
            video_height,
            video_width,
            args_cli,
            product_mp4_path=product_mp4_path,
            product_camera_names=product_camera_names,
            product_video_height=product_video_height,
            product_video_width=product_video_width
        )
    else:
        video_processor = None

    with (
        contextlib.suppress(KeyboardInterrupt),
        h5py.File(save_dir / f"replay_state_ee_poses.hdf5", "w") as ee_poses_hdf5_f,
    ):
        ee_poses_hdf5_f.create_group("data")
        for i in tqdm.tqdm(episode_indices_to_replay, desc="Replaying"):
            if args_cli.replay_all_clips:
                episode_save_dir = save_dir / f'{episode_names[i]}'
                episode_save_dir.mkdir(parents=True, exist_ok=True)
            else:
                episode_save_dir = save_dir

            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            env_id = 0
            ep = episode_names[i]
            print(f"Replaying episode {ep}")
            env_episode_data_map[env_id] = dataset_file_handler.load_episode(ep, env.device)
            next_state = env_episode_data_map[env_id].get_next_state()
            ee_poses = []
            step_count = 0
            for _ in tqdm.tqdm(count(), desc=f"Replaying {ep}"):
                if next_state is None:
                    break
                if args_cli.first_person_view:
                    set_camera_follow_pose(env, env_cfg.viewport_cfg["offset"], env_cfg.viewport_cfg["lookat"])
                obs, _ = env.reset_to(next_state, torch.tensor([env_id], device=env.device), is_relative=True if env_args["robot_name"].endswith("Rel") else False)
                check_terminations(env=env)
                env.sim.render()
                next_state = env_episode_data_map[env_id].get_next_state()
                step_count += 1
                ee_pos = obs['embodiment_general_obs']['eef_pos']
                ee_quat = obs['embodiment_general_obs']['eef_quat']
                eef_pose = torch.cat([ee_pos, ee_quat], dim=-1)
                ee_poses.append(eef_pose.cpu().numpy())
                if app_launcher._enable_cameras and video_processor:
                    # Add frame to video processing queue
                    camera_names = env.cfg.isaaclab_arena_env.embodiment.active_observation_camera_names
                    video_processor.add_frame(obs['policy'], camera_names)

            if ee_poses:
                ee_poses = np.concatenate(ee_poses, axis=0)
                # save ee poses to hdf5 file
                ee_poses_hdf5_f.create_dataset(f"data/{ep}/ee_poses", data=ee_poses)

    print("Closing process start")
    # Wait for all video processing tasks to complete and cleanup
    if video_processor:
        video_processor.shutdown()
        print("Shut down video processor")

        # Check video file after processing
        video_path = video_processor.get_video_path()
        if os.path.exists(video_path):
            video_duration = get_video_duration(video_path)
            print(f"Video duration: {video_duration} seconds")
        else:
            print(f"Video file not found: {video_path}")

        video_meta_json_path = replay_mp4_path.parent / "video_meta.json"
        with open(video_meta_json_path, 'w') as f:
            json.dump({"video_duration": video_duration}, f, indent=2)

    print(f"Finished replaying {len(episode_names)} episode{'s' if replayed_episode_count > 1 else ''}.")

    env.close()
    print("Close simulation env")
    simulation_app.close()
    if not args_cli.without_image:
        cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    # run the main function
    main()
