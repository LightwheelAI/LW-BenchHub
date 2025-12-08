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

import argparse
import ast
import os
from pathlib import Path

import yaml
import cv2
import h5py
import numpy as np
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lw_benchhub.scripts.math_utils import compute_delta_pose, pose_left_multiply


def convert_isaaclab_to_lerobot(args, config):
    # Load configuration: features, robot_type, and default task
    features = config["features"]
    robot_type = config["robot_type"]
    repo_id = args.tgt_repo_id or f"{Path(args.root_path).stem}-lerobot"
    root = Path(args.root_path).parent / repo_id

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=str(root),
        fps=30,
        robot_type=robot_type,
        features=features,
    )

    clip_names = os.listdir(args.root_path)
    dataset_files = [os.path.join(args.root_path, clip_name, 'dataset_success.hdf5') for clip_name in clip_names]

    success = 0
    failed = 0
    failed_list = []

    for i, dataset_file in enumerate(dataset_files):
        try:
            process_hdf5(dataset, dataset_file, args.select_cameras)
            success += 1
            print(f"Processed {i+1}/{len(dataset_files)}: {dataset_file}")
        except Exception as e:
            failed += 1
            failed_list.append(dataset_file)
            print(f"Failed to process {dataset_file}: {e}")
        print(f"Success: {success}, Failed: {failed}")
        print(f"Failed list: {failed_list}")


def process_hdf5(dataset, hdf5_path, cam_names):
    with h5py.File(hdf5_path, "r") as f:
        demo_names = list(f["data"].keys())
        episode_count = len(demo_names)
        print(f"Found {len(demo_names)} demos: {demo_names}")
        demo_names.sort(key=lambda x: int(x.split("_")[-1]))

        for i in tqdm.tqdm(range(0, episode_count), desc="Convert last demo"):
            demo_name = demo_names[i]
            demo_group = f["data"][demo_name]

            actions = np.array(demo_group["eef/relative_left_pose"])    # (T,7) [dx,dy,dz,qw,qx,qy,qz]
            actions_abs = np.array(demo_group["eef/left_pose"])         # (T,7) [x,y,z,  qw,qx,qy,qz]

            pose_curr = pose_left_multiply(actions_abs, actions)

            first_base = np.array([[0.3724, 0.1508, 0.7425, 0, 0, 0, 0]])
            urdf_base = np.array([[0.3725, 0.1508, 0.263, 0, 0, 0, 0]])
            offset = first_base - urdf_base

            pose_urdf_curr = pose_curr - offset

            delta = compute_delta_pose(pose_urdf_curr, actions_abs).astype(np.float32)

            action_gripper = (np.array(demo_group["obs/raw_action/lgrasp"])[:, None] + 1) / 2  # (T,1)
            action_6d = np.concatenate([delta, action_gripper], axis=-1)

            state_gripper = np.array(demo_group["obs/joint_pos"])[:, -1:] / 0.044  # (T,1)
            state_6d = np.concatenate([pose_urdf_curr.astype(np.float32), state_gripper], axis=-1)

            T = action_6d.shape[0]

            video_paths = {
                cam_name: Path(hdf5_path).parent / 'replay_results' / demo_name / f"{cam_name}.mp4" for cam_name in cam_names
            }
            cap_cams = {cam_name: cv2.VideoCapture(str(video_paths[cam_name])) for cam_name in cam_names}

            for j in range(5):
                for cam_name in cam_names:
                    _, _ = cap_cams[cam_name].read()

            for i in tqdm.tqdm(range(5, T), desc="Processing frames"):
                frame = {
                    "observation.state": state_6d[i],   # (7,) = [x,y,z, wx,wy,wz, gripper]
                    "action": action_6d[i],             # (7,) = [dx,dy,dz, wx_rel,wy_rel,wz_rel, gripper]
                }
                for cam_name in cam_names:
                    _, img = cap_cams[cam_name].read()
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frame[f"observation.images.{cam_name}"] = img

                dataset.add_frame(frame, task="Grab the block and lift it up.")
            dataset.save_episode()
            for cam_name in cam_names:
                cap_cams[cam_name].release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_repo_id", type=str, default=None, help="LeRobot dataset repo_id (folder name)")
    parser.add_argument("--config_yaml", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--root_path", type=str, default=None, help="Path to the root directory of the dataset")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config_yaml, "r") as f:
        config = yaml.safe_load(f)
    config['features']['observation.state']['shape'] = ast.literal_eval(config['features']['observation.state']['shape'])
    config['features']['action']['shape'] = ast.literal_eval(config['features']['action']['shape'])
    args.select_cameras = [i.split(".")[-1] for i in config['features'] if config['features'][i]['dtype'] == 'video']
    convert_isaaclab_to_lerobot(args, config)
