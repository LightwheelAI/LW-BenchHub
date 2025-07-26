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
import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

from isaaclab.app import AppLauncher

from lwlab.utils.isaaclab_utils import get_robot_joint_target_from_scene

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
parser.add_argument("--only_last_clips", action="store_true", default=True, help="only replay the last clips")

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

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import torch

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
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def main():
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
            scene_name=f"robocasakitchen-{env_args['layout_id']}-{env_args['style_id']}" if args_cli.layout is None else args_cli.layout,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True,
            replay_cfgs={"hdf5_path": args_cli.dataset_file, "ep_meta": env_args, "render_resolution": (args_cli.width, args_cli.height)},
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.EVAL if args_cli.replay_mode == "action" else ExecuteMode.EVAL_JOINT_TARGETS,
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
    delattr(env_cfg.terminations, "time_out")

    # create environment from loaded config
    env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped

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

    # reset before starting
    env.reset()

    teleop_interface.reset()
    font = ImageFont.load_default()
    if hasattr(font, 'font_variant'):
        font = font.font_variant(size=20)

    # prepare video writer and json file
    replay_mp4_path = Path(args_cli.dataset_file).parent / f"isaac_replay_action_{args_cli.replay_mode}.mp4"
    replay_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    replay_json_path = replay_mp4_path.with_suffix('.json')
    success_info = {}
    has_success = False

    # read ee_poses from hdf5
    import h5py
    if (ee_poses_path := replay_mp4_path.parent / "replay_state_ee_poses.hdf5").exists():
        gt_ee_poses_f = h5py.File(ee_poses_path, "r")
    else:
        gt_ee_poses_f = None

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    episode_names.sort(key=lambda x: int(x.split("_")[-1]))
    replayed_episode_count = 0
    pose_divergence_ep = []
    with (
        contextlib.suppress(KeyboardInterrupt) and torch.inference_mode(),
        media.VideoWriter(path=replay_mp4_path, shape=(args_cli.height, len(env.cfg.render_cfgs) * args_cli.width), fps=30) as v,
        h5py.File(replay_mp4_path.parent / f"isaac_replay_action_{args_cli.replay_mode}_pose_divergence.hdf5", "w") as pose_divergence_f
    ):
        pose_divergence_f.create_group("data")
        while simulation_app.is_running() and not simulation_app.is_exiting():
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
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if replayed_episode_count:
                            # compare ee_poses with gt_ee_poses, calculate pose divergence
                            if gt_ee_poses_f is not None:
                                gt_ee_poses = gt_ee_poses_f["data"][episode_name]["ee_poses"][:]
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

                            # compare joint_pos_list with gt_joint_pos, calculate joint divergence
                            joint_pos_list = np.concatenate(joint_pos_list, axis=0)
                            joint_divergence = joint_pos_list[:-1] - gt_joint_pos.cpu().numpy()[:len(joint_pos_list) - 1]
                            # print(f"Joint divergence: last step: {joint_divergence[-1]}, mean: {joint_divergence.mean()} max: {joint_divergence.max()}")
                            # save joint divergence to hdf5
                            pose_divergence_f.create_dataset(f"data/{episode_name}/joint_pos", data=joint_pos_list[:])
                            pose_divergence_f.create_dataset(f"data/{episode_name}/joint_pos_divergence", data=joint_divergence[:])
                            if args_cli.replay_mode == "joint_target":
                                joint_target_list = np.concatenate(joint_target_list, axis=0)
                                gt_joint_target_list = np.concatenate(gt_joint_target_list, axis=0)
                                joint_target_divergence = joint_target_list - gt_joint_target_list
                                pose_divergence_f.create_dataset(f"data/{episode_name}/joint_target", data=joint_target_list[:])
                                pose_divergence_f.create_dataset(f"data/{episode_name}/joint_target_divergence", data=joint_target_divergence[:])

                        if next_episode_index is not None:
                            ee_poses = []
                            joint_pos_list = []
                            joint_target_list = []
                            gt_joint_target_list = []
                            replayed_episode_count += 1
                            episode_name = episode_names[next_episode_index]
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode {episode_name} to env_{env_id}")
                            episode_data = dataset_file_handler.load_episode(
                                episode_names[next_episode_index], env.device
                            )
                            env_episode_data_map[env_id] = episode_data

                            # if replayed_episode_count <= 1:
                            if args_cli.replay_mode == "action":
                                gt_joint_pos = episode_data._data["states"]["articulation"]["robot"]["joint_position"]
                            elif args_cli.replay_mode == "joint_target":
                                gt_joint_pos = episode_data._data["states"]["articulation"]["robot"]["joint_position"]
                                gt_joint_target = episode_data._data["joint_targets"]["joint_pos_target"]
                            else:
                                raise ValueError(f"Invalid replay mode: {args_cli.replay_mode}, can only be 'action' or 'joint_target'")
                            # Set initial state for the new episode
                            # env.step(actions)
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=False)
                            # env.step(actions)

                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            if args_cli.replay_mode == "joint_target":
                                env_next_joint_target = env_episode_data_map[env_id].get_next_joint_target()
                                # env_next_joint_target = env_episode_data_map[env_id].get_next_joint_target()
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

                ee_poses.append(obs['policy']['ee_pose'].cpu().numpy())
                joint_pos_list.append(obs["policy"]["joint_pos"].cpu().numpy())
                if args_cli.replay_mode == "joint_target":
                    joint_target_list.append(get_robot_joint_target_from_scene(env.scene)["joint_pos_target"].cpu().numpy())
                    gt_joint_target_list.append(actions.reshape(env.cfg.decimation, -1)[-1:, ...].cpu().numpy())
                if app_launcher._enable_cameras:
                    full_image = np.concatenate([obs['policy'][name].cpu().numpy() for name in env.cfg.render_cfgs.keys()], axis=0).transpose(0, 2, 1, 3)
                    full_image = full_image.reshape(-1, *full_image.shape[2:]).transpose(1, 0, 2)
                    # draw the frame number on the image
                    # cv2.putText(full_image, f"Frame: {env_episode_data_map[env_id].next_state_index - 1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pil_im = Image.fromarray(full_image)
                    draw = ImageDraw.Draw(pil_im)
                    draw.text((20, 30), f"{env_episode_data_map[env_id].next_action_index - 1}", fill=(0, 255, 0), font=font)
                    full_image = np.array(pil_im)

                    v.add_image(full_image)
                    cv2.imshow("replay", full_image[..., ::-1])
                    cv2.waitKey(1)
                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
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

            # save pose divergence to hdf5
            with open(replay_json_path, 'w') as f:
                json.dump(success_info, f, indent=4)
            break
    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
