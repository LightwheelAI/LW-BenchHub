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
import copy
import json
import os
from itertools import count
from pathlib import Path

import numpy as np
import tqdm
from PIL import ImageFont

from isaaclab.app import AppLauncher

from lw_benchhub.utils.isaaclab_utils import get_robot_joint_target_from_scene, update_sensors
from lw_benchhub.utils.log_utils import get_default_logger, get_logger
from lw_benchhub.utils.video_recorder import VideoProcessor, calculate_camera_layout

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
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument("--robot_scale", type=float, default=1.0, help="robot scale")
parser.add_argument("--first_person_view", action="store_true", default=False, help="first person view")
parser.add_argument("--replay_mode", type=str, default="action", help="replay mode(action or joint_target)")
parser.add_argument("--layout", type=str, default=None, help="layout name")
parser.add_argument("--replay_all_clips", action="store_true", help="replay all clips, otherwise only replay the last clips")
parser.add_argument("--record", action="store_true", default=False, help="record the replayed actions")
parser.add_argument("--demo", type=int, default=-1, help="demo num in hdf5.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

if args_cli.replay_mode not in ("action", "joint_target"):
    raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action' or 'joint_target'")

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

if not args_cli.without_image:
    import cv2

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch

if not args_cli.headless:
    from lw_benchhub.core.devices.keyboard.se3_keyboard import Se3Keyboard
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from lw_benchhub.utils.place_utils.env_utils import set_seed

is_paused = False

obj_state_logger = get_logger("obj_state_logger")


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index].squeeze(0)
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name].squeeze(0)
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def get_active_check_points(episode_data):
    """Get the active check points from the episode data."""
    active_check_points = []
    check_points = episode_data.data.get("checkpoints", {})
    frame_index_triggered = check_points.get("frame_index_triggered")
    frame_index = check_points.get("frame_index")

    if frame_index_triggered is None or frame_index is None:
        return active_check_points

    for i in range(len(frame_index_triggered)):
        if frame_index_triggered[i].item() > 0:
            active_check_points.append(int(frame_index[i].item()))
    return active_check_points


def main():
    """Replay episodes loaded from a file."""
    from isaaclab.envs import ManagerBasedRLEnv

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

    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    replayed_episode_count = 0

    episode_names_to_replay = []
    for idx in episode_indices_to_replay:
        episode_names_to_replay.append(episode_names[idx])

    num_envs = args_cli.num_envs

    env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
    if "LW_API_ENDPOINT" in env_args.keys():
        os.environ["LW_API_ENDPOINT"] = env_args["LW_API_ENDPOINT"]
        from lightwheel_sdk.loader import lw_client
        lw_client.host = env_args["LW_API_ENDPOINT"]
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
        if args_cli.layout is None:
            scene_name = f"{scene_type}-{env_args['layout_id']}-{env_args['style_id']}"
        else:
            scene_name = args_cli.layout
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
            execute_mode=ExecuteMode.REPLAY_ACTION if args_cli.replay_mode == "action" else ExecuteMode.REPLAY_JOINT_TARGETS,
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
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.dataset_file)
        env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0] + f"_{args_cli.replay_mode}_replay_record.hdf5"

    env_cfg.terminations.time_out = None

    # create environment from loaded config
    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped
    set_seed(env_cfg.seed, env)

    # warmup rendering
    if not args_cli.headless and env.common_step_counter <= 1:
        for _ in range(env.cfg.warmup_steps):
            update_sensors(env, env.physics_dt)

    if app_launcher._enable_cameras and not args_cli.headless:
        teleop_interface = Se3Keyboard(pos_sensitivity=0.1, rot_sensitivity=0.1)
        teleop_interface.add_callback("N", play_cb)
        teleop_interface.add_callback("B", pause_cb)
        print('Press "B" to pause and "N" to resume the replayed actions.')
    print(args_cli.dataset_file)

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    elif args_cli.replay_mode == "action":
        idle_action = torch.zeros(env.action_space.shape)
    elif args_cli.replay_mode == "joint_target":
        idle_action = torch.zeros(env.action_space.shape)
    else:
        raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action' or 'joint_target'")

    import carb
    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", True)
    # reset before starting
    env.reset(seed=env.cfg.seed)
    if app_launcher._enable_cameras and not args_cli.headless:
        teleop_interface.reset()
    font = ImageFont.load_default()
    if hasattr(font, 'font_variant'):
        font = font.font_variant(size=20)

    # prepare video writer and json file
    success_info = {}
    has_success = False

    # read ee_poses from hdf5
    import h5py

    # simulate environment -- run everything in inference mode
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

    # Initialize async image processor
    video_processor = None
    joint_pos_list = None
    episode_name = None
    ee_poses = None
    joint_target_list = None
    gt_joint_target_list = None
    gt_joint_pos = None
    obj_states = None
    obj_force_states = None

    for i in episode_indices_to_replay:
        episode_indices_to_replay_tmp = [i]
        if not args_cli.replay_all_clips:
            save_dir = Path(args_cli.dataset_file).parent
        else:
            save_dir = Path(args_cli.dataset_file).parent / 'replay_results' / episode_names[i]
        save_dir.mkdir(parents=True, exist_ok=True)
        replay_mp4_path = save_dir / f"isaac_replay_action_{args_cli.replay_mode}.mp4"
        replay_json_path = replay_mp4_path.with_suffix('.json')
        if (ee_poses_path := save_dir / "replay_state_ee_poses.hdf5").exists():
            gt_ee_poses_f = h5py.File(ee_poses_path, "r")
        else:
            gt_ee_poses_f = None

        # Initialize video processor (manages VideoWriter internally)
        if app_launcher._enable_cameras:
            video_processor = VideoProcessor(replay_mp4_path, video_height, video_width, args_cli)
        else:
            video_processor = None

        with (
            contextlib.suppress(KeyboardInterrupt),
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_pose_divergence.hdf5", "w") as pose_divergence_f,
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_obj_pose_divergence.hdf5", "w") as obj_divergence_f,
            h5py.File(save_dir / f"isaac_replay_action_{args_cli.replay_mode}_obj_force.hdf5", "w") as obj_force_f
        ):
            pose_divergence_f.create_group("data")
            obj_divergence_f.create_group("data")
            obj_force_f.create_group("data")
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            for _ in tqdm.tqdm(count(), desc=f"Replaying actions {args_cli.replay_mode}"):

                # initialize actions with idle action so those without next action will not move
                actions = idle_action
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    env_next_joint_target = env_episode_data_map[env_id].get_next_joint_target()

                    if env_next_action is None:
                        next_episode_index = None
                        while episode_indices_to_replay_tmp:
                            next_episode_index = episode_indices_to_replay_tmp.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if replayed_episode_count and joint_pos_list is not None and isinstance(joint_pos_list, list):
                            # compare ee_poses with gt_ee_poses, calculate pose divergence
                            if gt_ee_poses_f is not None and episode_name is not None:
                                gt_ee_poses = gt_ee_poses_f["data"][episode_name]["ee_poses"][:]
                                if ee_poses is not None:
                                    ee_poses = np.concatenate(ee_poses, axis=0)
                                    pose_divergence = np.linalg.norm(ee_poses[:-1, :, :3] - gt_ee_poses[:len(ee_poses) - 1, :, :3], axis=-1)
                                    pose_divergence_norm = np.linalg.norm(pose_divergence, axis=-1)
                                    print(f"Pose divergence: last step: {pose_divergence[-1].tolist()}, mean: {np.mean(pose_divergence,axis=0).tolist()} max: {np.max(pose_divergence,axis=0).tolist()}")
                                    success_info[episode_name] = {
                                        "pose_divergence_last_step": pose_divergence[-1].tolist(),
                                        "pose_divergence_mean": np.mean(pose_divergence, axis=0).tolist(),
                                        "pose_divergence_max": np.max(pose_divergence, axis=0).tolist(),
                                        "pose_divergence_norm_last_step": pose_divergence_norm[-1].tolist(),
                                        "pose_divergence_norm_mean": np.mean(pose_divergence_norm, axis=0).tolist(),
                                        "pose_divergence_norm_max": np.max(pose_divergence_norm, axis=0).tolist(),
                                        "success": has_success
                                    }
                                    # save pose divergence to hdf5
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/ee_poses", data=ee_poses[:])
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/pose_divergence", data=pose_divergence)
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/pose_divergence_norm", data=pose_divergence_norm)
                            else:
                                if episode_name is not None:
                                    success_info[episode_name] = {
                                        "success": has_success
                                    }

                            # compare joint_pos_list with gt_joint_pos, calculate joint divergence
                            if gt_joint_pos is not None:
                                joint_pos_list = np.concatenate(joint_pos_list, axis=0)
                                joint_divergence = joint_pos_list[:-1] - gt_joint_pos.cpu().numpy()[:len(joint_pos_list) - 1]
                                # print(f"Joint divergence: last step: {joint_divergence[-1]}, mean: {joint_divergence.mean()} max: {joint_divergence.max()}")
                                # save joint divergence to hdf5
                                if episode_name is not None:
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/joint_pos", data=joint_pos_list[:])
                                    pose_divergence_f.create_dataset(f"data/{episode_name}/joint_pos_divergence", data=joint_divergence[:])
                                if args_cli.replay_mode == "joint_target" and joint_target_list is not None and gt_joint_target_list is not None:
                                    joint_target_list = np.concatenate(joint_target_list, axis=0)
                                    gt_joint_target_list = np.concatenate(gt_joint_target_list, axis=0)
                                    joint_target_divergence = joint_target_list - gt_joint_target_list
                                    if episode_name is not None:
                                        pose_divergence_f.create_dataset(f"data/{episode_name}/joint_target", data=joint_target_list[:])
                                        pose_divergence_f.create_dataset(f"data/{episode_name}/gt_joint_target", data=gt_joint_target_list[:])
                                        pose_divergence_f.create_dataset(f"data/{episode_name}/joint_target_divergence", data=joint_target_divergence[:])

                        if next_episode_index is not None:
                            ee_poses = []
                            joint_pos_list = []
                            joint_target_list = []
                            gt_joint_target_list = []
                            obj_states = {}
                            obj_force_states = {}
                            replayed_episode_count += 1
                            episode_name = episode_names[next_episode_index]
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode {episode_name} to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            current_frame_index = 0
                            env_episode_data_map[env_id] = episode_data
                            active_check_points = get_active_check_points(episode_data)
                            checkpoint_path = os.path.join(save_dir, f"isaac_replay_action_{args_cli.replay_mode}_checkpoint_{current_frame_index}.pt")
                            print(f"---------- Active check points: {active_check_points} -------------")
                            env_episode_data_map[env_id] = episode_data

                            # if replayed_episode_count <= 1:
                            if "states" not in episode_data._data:
                                break
                            if args_cli.replay_mode == "action":
                                gt_joint_pos = episode_data._data["states"]["articulation"]["robot"]["joint_position"]
                            elif args_cli.replay_mode == "joint_target":
                                gt_joint_pos = episode_data._data["states"]["articulation"]["robot"]["joint_position"]
                            else:
                                raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action' or 'joint_target'")
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            from lw_benchhub.utils.place_utils.env_utils import reset_physx
                            reset_physx(env)
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), seed=env.cfg.seed, is_relative=False)

                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            if args_cli.replay_mode == "joint_target":
                                env_next_joint_target = env_episode_data_map[env_id].get_next_joint_target()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    if args_cli.replay_mode == "action":
                        actions[env_id] = env_next_action
                    elif args_cli.replay_mode == "joint_target":
                        actions[env_id] = env_next_joint_target["joint_pos_target"]
                    else:
                        raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action' or 'joint_target'")
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                if not has_next_action:
                    break

                obs, _, ter, _, _ = env.step(actions)

                if current_frame_index in active_check_points:
                    from lw_benchhub.utils.teleop_utils import save_checkpoint, load_checkpoint
                    save_checkpoint(env, checkpoint_path)
                    load_checkpoint(env, checkpoint_path)
                current_frame_index += 1

                env_objs = env.scene.rigid_objects
                all_keys = list(env_objs.keys())

                current_obj_state = env.scene.get_state(is_relative=True)["rigid_object"]
                # runtime_state[asset_type][asset_name][state_name][runtime_env_index].squeeze(0)

                for obj_name in all_keys:
                    # Initialize object state tracking lists if not exists
                    if obj_name not in obj_states:
                        obj_states[obj_name] = {
                            'obj_root_pose': [],
                            'obj_root_vel': [],
                            'ref_root_pose': [],
                            'ref_root_vel': []
                        }
                    if obj_name not in obj_force_states:
                        obj_force_states[obj_name] = {
                            'obj_force': [],
                        }

                    obj_root_pose = current_obj_state[obj_name]["root_pose"].cpu().numpy()
                    obj_root_vel = current_obj_state[obj_name]["root_velocity"].cpu().numpy()
                    # Append current object states to lists (will be written to hdf5 at episode end)
                    obj_states[obj_name]['obj_root_pose'].append(obj_root_pose)
                    obj_states[obj_name]['obj_root_vel'].append(obj_root_vel)
                    if f'{obj_name}_contact' in env.scene.sensors:
                        # Get force data for specific environment and create a copy to avoid reference issues
                        temp_force = env.scene.sensors[f'{obj_name}_contact']._data.net_forces_w[:, 0, :].cpu().numpy().copy()
                        obj_force_states[obj_name]['obj_force'].append(temp_force)
                        get_default_logger().info(f"Obj {obj_name} force shape: {temp_force.shape}, force: {temp_force}")
                    else:
                        get_default_logger().warning(f"Sensor {f'{obj_name}_contact'} not found")

                    # Get corresponding reference states from dataset for current step
                    if obj_name in env_episode_data_map[env_id]._data['states']['rigid_object']:
                        current_state_idx = env_episode_data_map[env_id].next_state_index
                        total_states = len(env_episode_data_map[env_id]._data['states']['rigid_object'][obj_name]['root_pose'])

                        if current_state_idx < total_states:
                            # Get the specific state for current step, ensuring correct indexing
                            ref_root_pose = env_episode_data_map[env_id]._data['states']['rigid_object'][obj_name]['root_pose'][current_state_idx].unsqueeze(0).cpu().numpy()
                            ref_root_vel = env_episode_data_map[env_id]._data['states']['rigid_object'][obj_name]['root_velocity'][current_state_idx].unsqueeze(0).cpu().numpy()

                            # Append reference states to lists (will be written to hdf5 at episode end)
                            obj_states[obj_name]['ref_root_pose'].append(ref_root_pose)
                            obj_states[obj_name]['ref_root_vel'].append(ref_root_vel)
                        else:
                            get_default_logger().warning(f"State index {current_state_idx} out of range for {obj_name} (total: {total_states})")

                ee_poses.append(obs['embodiment_general_obs']['eef_pos'].cpu().numpy())
                joint_pos_list.append(obs['embodiment_general_obs']['joint_pos'].cpu().numpy())
                if args_cli.replay_mode == "joint_target":
                    joint_target_list.append(get_robot_joint_target_from_scene(env.scene)["joint_pos_target"].cpu().numpy())
                    gt_actions = copy.deepcopy(actions)
                    gt_joint_target_list.append(gt_actions.reshape(env.cfg.decimation, -1)[-1:, ...].cpu().numpy())
                if app_launcher._enable_cameras and video_processor:
                    camera_names = env.cfg.isaaclab_arena_env.embodiment.active_observation_camera_names
                    # Process images asynchronously
                    video_processor.add_frame(obs['policy'], camera_names)

                state_from_dataset = env_episode_data_map[0].get_next_state()
                if state_validation_enabled:
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
                if ter:
                    has_success = True
                    env_episode_data_map[env_id]._next_action_index += 9999999999
            if obj_force_states is not None and obj_force_states:
                for obj_name, obj_force_data in obj_force_states.items():
                    if len(obj_force_data['obj_force']) > 0:
                        obj_force_array = np.concatenate(obj_force_data['obj_force'], axis=0)
                        obj_force_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_force", data=obj_force_array[:])
                    else:
                        get_default_logger().warning(f"Object {obj_name} has no force data")
                        continue

            if obj_states is not None and obj_states:
                get_default_logger().info(f"Processing {len(obj_states)} objects for episode {episode_name}")
                for obj_name, obj_data in obj_states.items():
                    if obj_data['obj_root_pose'] and obj_data['ref_root_pose']:
                        # Convert accumulated lists to numpy arrays for batch processing
                        obj_root_pose_array = np.concatenate(obj_data['obj_root_pose'], axis=0)
                        obj_root_vel_array = np.concatenate(obj_data['obj_root_vel'], axis=0)
                        ref_root_pose_array = np.concatenate(obj_data['ref_root_pose'], axis=0)
                        ref_root_vel_array = np.concatenate(obj_data['ref_root_vel'], axis=0)

                        # Ensure arrays have the same length before calculating divergences
                        min_length = min(len(obj_root_pose_array), len(ref_root_pose_array))

                        if min_length > 0:
                            obj_root_pose_array = obj_root_pose_array[:min_length]
                            obj_root_vel_array = obj_root_vel_array[:min_length]
                            ref_root_pose_array = ref_root_pose_array[:min_length]
                            ref_root_vel_array = ref_root_vel_array[:min_length]

                            # Calculate divergences between simulation and reference data
                            obj_root_pose_divergence = obj_root_pose_array - ref_root_pose_array
                            obj_root_vel_divergence = obj_root_vel_array - ref_root_vel_array

                            # Batch write all object state data to hdf5 file at once
                            obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_pose", data=obj_root_pose_array[:])
                            obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/ref_root_pose", data=ref_root_pose_array[:])
                            obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_pose_divergence", data=obj_root_pose_divergence[:])
                            obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_vel", data=obj_root_vel_array[:])
                            obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/ref_root_vel", data=ref_root_vel_array[:])
                            obj_divergence_f.create_dataset(f"data/{episode_name}/{obj_name}/obj_root_vel_divergence", data=obj_root_vel_divergence[:])
                        else:
                            get_default_logger().warning(f"No valid data for object {obj_name}, skipping divergence calculation")
                    else:
                        get_default_logger().warning(f"Object {obj_name} has no data: obj_pose={len(obj_data['obj_root_pose'])}, ref_pose={len(obj_data['ref_root_pose'])}")

            # save pose divergence to hdf5
            with open(replay_json_path, 'w') as f:
                json.dump(success_info, f, indent=4)

        if args_cli.record:
            env.recorder_manager.export_episodes()

    print("Closing process start")
    # Wait for all video processing tasks to complete and cleanup
    if video_processor:
        video_processor.shutdown()
        print("Shut down video processor")

    def save_metrics():
        """Save metrics data to JSON file"""
        metrics_data = env_cfg.isaaclab_arena_env.task.get_checker_results()

        # Save metrics to JSON file
        if metrics_data:
            metrics_file_path = os.path.join(save_dir, "metrics.json")
            try:
                with open(metrics_file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"Metrics saved to: {metrics_file_path}")
            except Exception as e:
                print(f"Failed to save metrics: {e}")

    save_metrics()
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.", flush=True)

    env.close()
    print("Close simulation env")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
