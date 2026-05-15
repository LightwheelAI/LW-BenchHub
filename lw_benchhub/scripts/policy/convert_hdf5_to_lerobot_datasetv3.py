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
from pathlib import Path

import h5py
import numpy as np
import tqdm
import yaml
import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CONFIG_DIR = Path(__file__).resolve().parent / "config"


def load_robot_config(robot_config: str) -> dict:
    """Load robot schema and explicit HDF5/video feature mappings."""
    path = Path(robot_config).expanduser()
    candidates = [path] if path.exists() else [CONFIG_DIR / robot_config, CONFIG_DIR / f"{robot_config}.yaml", CONFIG_DIR / f"{robot_config}.yml"]
    config_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"Robot config not found: {robot_config}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    features = config.get("features")
    mapping = config.get("hdf5_mapping")
    video_mapping = config.get("video_mapping", {})
    if not config.get("robot_type") or not isinstance(features, dict) or not isinstance(mapping, dict) or not isinstance(video_mapping, dict):
        raise KeyError(f"{config_path} must define robot_type, features, hdf5_mapping, and optional video_mapping")

    for feature in features.values():
        shape = feature.get("shape")
        if isinstance(shape, str):
            shape = ast.literal_eval(shape)
        feature["shape"] = tuple(shape) if isinstance(shape, list) else shape

    hdf5_features = {name for name, feature in features.items() if feature.get("dtype") != "video"}
    video_features = {name for name, feature in features.items() if feature.get("dtype") == "video"}
    missing = sorted(hdf5_features - set(mapping))
    missing_video = sorted(video_features - set(video_mapping))
    if missing:
        raise KeyError(f"{config_path} missing hdf5_mapping for: {missing}")
    if missing_video:
        raise KeyError(f"{config_path} missing video_mapping for: {missing_video}")
    return config


def discover_hdf5_files(dataset_path: Path, hdf5_name: str) -> list[tuple[Path, Path]]:
    """Find one success HDF5 per episode directory."""
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"dataset_path must be an existing directory: {dataset_path}")

    episodes = []
    for episode_dir in sorted(path for path in dataset_path.iterdir() if path.is_dir()):
        matches = sorted(episode_dir.rglob(hdf5_name))
        if matches:
            if len(matches) > 1:
                print(f"Multiple {hdf5_name} files found under {episode_dir}; using {matches[0]}")
            episodes.append((matches[0], episode_dir))
        else:
            print(f"Skipped {episode_dir}: no {hdf5_name}")

    if not episodes:
        raise FileNotFoundError(f"No {hdf5_name} files found under {dataset_path}")
    return episodes


def demo_sort_key(name: str):
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return name


def demo_names(h5_file: h5py.File) -> list[str]:
    """Only the last demo is the successful rollout to export."""
    if "data" not in h5_file:
        raise KeyError("HDF5 file missing top-level 'data' group")
    names = [name for name, value in h5_file["data"].items() if isinstance(value, h5py.Group)]
    if not names:
        raise ValueError("HDF5 file contains no demo groups under 'data'")
    return [sorted(names, key=demo_sort_key)[-1]]


def video_metadata(video_path: Path) -> tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
    cap.release()
    return height, width, 3, fps


def fill_video_features(config: dict, episode_dir: Path) -> None:
    """Fill null video schema fields from the first matching MP4 before dataset creation."""
    for name, video_name in config.get("video_mapping", {}).items():
        feature = config["features"][name]
        height, width, channels, fps = video_metadata(find_video_file(episode_dir, video_name))
        if feature.get("shape") in (None, (), []):
            feature["shape"] = (height, width, channels)
        video_info = feature.setdefault("video_info", {})
        video_info["video.height"] = video_info.get("video.height") or height
        video_info["video.width"] = video_info.get("video.width") or width
        video_info["video.channels"] = video_info.get("video.channels") or channels
        video_info["video.fps"] = video_info.get("video.fps") or fps


def fps_from_features(features: dict) -> int:
    for feature in features.values():
        video_info = feature.get("video_info", {})
        if "video.fps" in video_info:
            return int(video_info["video.fps"])
    return 30


def find_video_file(episode_dir: Path, video_name: str) -> Path:
    """Resolve a configured MP4 filename recursively inside the episode folder."""
    matches = sorted(episode_dir.rglob(video_name))
    if not matches:
        raise FileNotFoundError(f"Video file not found under {episode_dir}: {video_name}")
    if len(matches) > 1:
        print(f"Multiple {video_name} files found under {episode_dir}; using {matches[0]}")
    return matches[0]


def read_video_frames(video_path: Path, start: int, length: int) -> list[np.ndarray]:
    """Read RGB frames aligned to the kept HDF5 steps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {video_path.name}: total_frames={total_frames}, need to read from frame {start} to {start + length - 1} (length={length})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    try:
        for _ in range(length):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Video {video_path} ended before reading {length} frames from offset {start}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def convert_demo(dataset: LeRobotDataset, demo_group: h5py.Group, episode_dir: Path, config: dict, task: str, skip_frames: int) -> int:
    features = config["features"]
    mapping = config["hdf5_mapping"]
    video_mapping = config["video_mapping"]
    hdf5_features = [name for name, feature in features.items() if feature.get("dtype") != "video"]
    video_features = [name for name, feature in features.items() if feature.get("dtype") == "video"]
    sources = {name: demo_group[mapping[name]] for name in hdf5_features}
    length = sources["action"].shape[0]

    for name, source in sources.items():
        shape = features[name].get("shape")
        if source.shape[0] != length:
            raise ValueError(f"{name} length mismatch: action={length}, {mapping[name]}={source.shape[0]}")
        if shape is not None and tuple(source.shape[1:]) != tuple(shape):
            raise ValueError(f"{name} shape mismatch: config={shape}, hdf5={source.shape[1:]}")

    start = min(skip_frames, length)
    frame_count = length - start
    # Low-dimensional HDF5 data is cheap to preload; MP4 frames are read after the same offset.
    small_arrays = {name: np.asarray(source).astype(np.float32, copy=False) for name, source in sources.items()}
    videos = {name: read_video_frames(find_video_file(episode_dir, video_mapping[name]), start, frame_count) for name in video_features}

    for offset, frame_index in enumerate(tqdm.tqdm(range(start, length), desc="Processing frames", leave=False)):
        frame = {name: small_arrays[name][frame_index] for name in hdf5_features}
        for name in video_features:
            frame[name] = videos[name][offset]
        frame['task'] = task
        dataset.add_frame(frame=frame)

    dataset.save_episode()
    return frame_count


def convert_hdf5_file(dataset: LeRobotDataset, hdf5_path: Path, episode_dir: Path, config: dict, task: str, skip_frames: int) -> tuple[int, int]:
    demos = 0
    frames = 0
    with h5py.File(hdf5_path, "r") as h5_file:
        for name in tqdm.tqdm(demo_names(h5_file), desc=f"Convert {hdf5_path.name}"):
            frames += convert_demo(dataset, h5_file["data"][name], episode_dir, config, task, skip_frames)
            demos += 1
    return demos, frames


def convert_isaaclab_to_lerobot(args: argparse.Namespace) -> None:
    config = load_robot_config(args.robot_config)
    episodes = discover_hdf5_files(args.dataset_path, args.hdf5_name)
    fill_video_features(config, episodes[0][1])
    dataset = LeRobotDataset.create(
        repo_id=args.tgt_repo_id,
        root=str(args.root_path) if args.root_path else None,
        fps=fps_from_features(config["features"]),
        robot_type=config["robot_type"],
        features=config["features"],
        video_backend="auto",
        streaming_encoding=True,
    )

    failed = []
    total_demos = 0
    total_frames = 0
    for hdf5_path, episode_dir in episodes:
        try:
            demos, frames = convert_hdf5_file(dataset, hdf5_path, episode_dir, config, args.task_description, args.skip_frames)
            total_demos += demos
            total_frames += frames
            print(f"Processed {hdf5_path}: {demos} demos, {frames} frames")
        except Exception as exc:
            failed.append(hdf5_path)
            print(f"Failed {hdf5_path}: {type(exc).__name__}: {exc}")
    dataset.finalize()
    print(f"Summary: files={len(episodes) - len(failed)}/{len(episodes)}, demos={total_demos}, frames={total_frames}")
    if failed:
        raise RuntimeError(f"Failed to convert {len(failed)} HDF5 files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LW-BenchHub HDF5 demos to a LeRobot dataset.")
    parser.add_argument("--tgt_repo_id", required=True, type=str, help="LeRobot dataset repo_id.")
    parser.add_argument("--dataset_path", required=True, type=Path, help="Root directory containing episode folders.")
    parser.add_argument("--root_path", required=False, type=Path, help="Output root for the LeRobot dataset.")
    parser.add_argument("--robot_config", required=True, type=str, help="Robot config path, filename, or stem under config/.")
    parser.add_argument("--task_description", required=True, type=str, help="Task description passed to LeRobot frames.")
    parser.add_argument("--hdf5_name", default="dataset_success.hdf5", type=str, help="HDF5 filename to find in each episode folder.")
    parser.add_argument("--skip_frames", default=5, type=int, help="Number of initial frames to skip in each demo.")
    convert_isaaclab_to_lerobot(parser.parse_args())
