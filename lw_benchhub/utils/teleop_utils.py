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

"""
Teleoperation Utilities Module

This module provides utility functions for teleoperation operations including:
- Checkpoint management (save/load)
- Enhanced reset functionality with data preservation
- Episode data manipulation
- State management utilities
"""

import copy
import os
import traceback
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
from requests.adapters import HTTPAdapter
from termcolor import colored
from tqdm import tqdm
from urllib3.util.retry import Retry


def download_file(url, output_dir, file_name="input_dataset.hdf5"):
    """
    Download a file from a URL to the specified output directory.

    Args:
        url (str): The URL of the file to download
        output_dir (str or Path): The directory where the file should be saved

    Returns:
        Path: Path to the downloaded file, or None if download failed
    """
    if requests is None or HTTPAdapter is None or Retry is None or tqdm is None:
        print(colored("Error: Required packages (requests, tqdm) not installed", "red"))
        return None

    try:
        # Parse filename from URL
        path = urlparse(url).path
        fname = file_name

        # Ensure output_dir is a Path object and exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        outfile = output_dir / fname

        # If file exists, it will be overwritten
        if outfile.exists():
            print(colored(f"File already exists at {outfile}, will be overwritten...", "yellow"))

        # Configure automatic retry (useful for temporary network/5xx errors)
        retry = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=["GET"]
        )
        sess = requests.Session()
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        sess.mount("http://", HTTPAdapter(max_retries=retry))

        # Get file size for progress bar (some services may not return this)
        print(colored(f"Downloading {fname}...", "cyan"))
        with sess.get(url, stream=True, timeout=(5, 60)) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=fname)

            with open(outfile, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            pbar.close()

        file_size = outfile.stat().st_size
        print(colored(f"✅ Download completed task hdf5 file: {outfile}  Size: {file_size} bytes", "green"))
        return outfile

    except Exception as e:
        print(colored(f"Download failed: {e}", "red"))
        print(traceback.format_exc())
        return None


def convert_list_to_2d_tensor(data_node):
    if isinstance(data_node, dict):
        converted = {}
        for key, value in data_node.items():
            converted[key] = convert_list_to_2d_tensor(value)
        return converted
    elif isinstance(data_node, list):
        return torch.stack(data_node, dim=0)
    elif isinstance(data_node, torch.Tensor):
        return data_node
    else:
        return data_node


def reset_and_keep_to(env, episode_data, target_frame_index, payload):
    """
    Reset environment to a specific frame while preserving episode data up to that frame.
    This implements a true "load checkpoint" functionality.

    Args:
        env: The environment to reset
        episode_data: The current episode data
        target_frame_index: The frame index to reset to (0-based)

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"[reset_and_keep_to] Resetting to frame {target_frame_index}")

    # Validate target frame index
    if target_frame_index < 0:
        print(f"[reset_and_keep_to] Error: Invalid frame index {target_frame_index}")
        return False

    # Check if episode data is available
    if not hasattr(env.recorder_manager, "_episodes") or not env.recorder_manager._episodes:
        print("[reset_and_keep_to] Error: No episode data available")
        return False

    # 1. Backup current complete episode data
    episodes_backup = {}
    try:
        for env_id, ep_data in env.recorder_manager._episodes.items():
            episodes_backup[env_id] = ep_data._data.copy()
        print(f"[reset_and_keep_to] Backed up episode data for {len(episodes_backup)} environments")
    except Exception as e:
        print(f"[reset_and_keep_to] Error backing up episode data: {e}")
        return False

    # 2. Get target frame state
    target_state = episode_data.get_state(target_frame_index)
    target_state = copy.deepcopy(target_state)
    if target_state is None:
        print(f"[reset_and_keep_to] Error: Failed to get state for frame {target_frame_index}")
        return False

    # Move state to correct device
    if hasattr(env, 'device'):
        target_state = _move_state_to_device(target_state, env.device)
    print(f"[reset_and_keep_to] Retrieved state for frame {target_frame_index}")

    # 3. Reset environment to target state
    try:
        target_state = convert_list_to_2d_tensor(target_state)
        from lw_benchhub.utils.place_utils.env_utils import reset_physx
        reset_physx(env)
        env.reset_to_check_state(target_state, torch.tensor([0], device=env.device), seed=env.cfg.seed, is_relative=True)
        print(f"[reset_and_keep_to] Environment reset to frame {target_frame_index}")
    except Exception as e:
        print(f"[reset_and_keep_to] Error resetting environment: {e}")
        return False

    # 4. Restore and truncate episode data to target frame
    try:
        for env_id, ep_data in env.recorder_manager._episodes.items():
            if env_id in episodes_backup:
                # Truncate data to target frame (inclusive)
                truncated_data = _truncate_episode_data(episodes_backup[env_id], target_frame_index + 1)
                ep_data._data = truncated_data
                print(f"[reset_and_keep_to] Restored episode data for env {env_id} up to frame {target_frame_index}")
            update_checkpoint_to_hdf5(env, target_frame_index, env_id)
    except Exception as e:
        print(f"[reset_and_keep_to] Error restoring episode data: {e}")
        # Even if data restoration fails, environment is reset, so partial success
        print(f"[reset_and_keep_to] Warning: Environment reset but episode data restoration failed")

    try:
        for name, term in env.action_manager._terms.items():
            if hasattr(term, 'load_check_point'):
                term.load_check_point(payload[name])
    except Exception as e:
        print(f"[reset_and_keep_to] Error restoring action terms: {e}")
        return False

    # Render the environment
    if hasattr(env, 'sim') and hasattr(env.sim, 'render'):
        env.sim.render()

    print(f"[reset_and_keep_to] Successfully reset to frame {target_frame_index}")
    return True


def _truncate_episode_data(episode_data, max_frames):
    """
    Truncate episode data to keep only the first max_frames frames.

    Args:
        episode_data: The episode data to truncate
        max_frames: Maximum number of frames to keep

    Returns:
        dict: Truncated episode data
    """
    truncated_data = {}

    for key, value in episode_data.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries
            truncated_data[key] = _truncate_episode_data(value, max_frames)
        elif isinstance(value, list):
            # Truncate list to specified length
            if len(value) > max_frames:
                truncated_data[key] = value[:max_frames]
            else:
                truncated_data[key] = value
        elif isinstance(value, torch.Tensor):
            # Truncate tensor to specified length
            if value.ndim > 0 and value.shape[0] > max_frames:
                truncated_data[key] = value[:max_frames]
            else:
                truncated_data[key] = value
        else:
            # Copy other types directly
            truncated_data[key] = value

    return truncated_data


def _move_state_to_device(state, device):
    """
    Move state data to the specified device.

    Args:
        state: The state data to move
        device: Target device

    Returns:
        State data moved to target device
    """
    if isinstance(state, torch.Tensor):
        return state.to(device)
    if isinstance(state, dict):
        return {k: _move_state_to_device(v, device) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        converted = [_move_state_to_device(v, device) for v in state]
        return type(state)(converted)
    return state


def _save_checkpoint_to_hdf5(env, frame_index, payload):
    """
    Save checkpoint information to HDF5 through recorder_manager.

    Args:
        env: The environment
        frame_index: Current frame index
        payload: Checkpoint payload dictionary
    """
    try:
        if not hasattr(env, 'recorder_manager') or len(env.recorder_manager.active_terms) == 0:
            print("[checkpoint] HDF5 save skipped: recorder_manager not available")
            return

        # Create checkpoint data dictionary
        checkpoint_data = {
            "frame_index": frame_index,
            "frame_index_triggered": 0
        }

        # Add action manager terms data
        for name, value in payload.items():
            if name != "frame_index":  # Skip frame_index as it's already included
                if isinstance(value, torch.Tensor):
                    checkpoint_data[f"action_manager/{name}"] = value.to(env.device)
                elif isinstance(value, (int, float)):
                    checkpoint_data[f"action_manager/{name}"] = torch.tensor([value], device=env.device, dtype=torch.float32)
                elif isinstance(value, (list, tuple)):
                    checkpoint_data[f"action_manager/{name}"] = torch.tensor(value, device=env.device, dtype=torch.float32)
                else:
                    print(f"[checkpoint] Warning: Skipping non-serializable action manager term '{name}' of type {type(value)}")

        # Save each checkpoint component to HDF5
        for key, value in checkpoint_data.items():
            if isinstance(value, torch.Tensor):
                env.recorder_manager.add_to_episodes(f"checkpoints/{key}", value.unsqueeze(0))
            else:
                # Convert non-tensor values to tensor first
                tensor_value = torch.tensor([value], device=env.device, dtype=torch.float32)
                env.recorder_manager.add_to_episodes(f"checkpoints/{key}", tensor_value.unsqueeze(0))

        print(f"[checkpoint] Saved checkpoint at frame {frame_index} to HDF5")

    except Exception as e:
        print(f"[checkpoint] HDF5 save failed: {e}")


def update_checkpoint_to_hdf5(env, frame_index=None, env_id=0):
    """
    Update the most recent checkpoint's frame_index_triggered to 1 in HDF5.

    Args:
        env: The environment
        frame_index: Optional specific frame index to mark as triggered.
                     If None, marks the most recent checkpoint as triggered.
        env_id: The environment ID to update. Defaults to 0.

    Example:
        # Mark the most recent checkpoint as triggered
        update_checkpoint_to_hdf5(env)

        # Mark a specific frame index as triggered
        update_checkpoint_to_hdf5(env, frame_index=100)

        # Mark for a specific environment
        update_checkpoint_to_hdf5(env, env_id=1)
    """
    try:
        if not hasattr(env, 'recorder_manager') or len(env.recorder_manager.active_terms) == 0:
            print("[checkpoint] HDF5 update skipped: recorder_manager not available")
            return

        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or env_id not in episodes:
            print(f"[checkpoint] update failed: recorder episode not ready (env_id={env_id})")
            return

        episode_data = episodes[env_id]

        # Get checkpoint data - nested structure: data["checkpoints"]["frame_index_triggered"]
        if "checkpoints" not in episode_data.data:
            print(f"[checkpoint] update failed: no checkpoint data found")
            return

        checkpoint_data = episode_data.data["checkpoints"]

        if "frame_index_triggered" not in checkpoint_data:
            print(f"[checkpoint] update failed: frame_index_triggered not found in checkpoint data")
            return

        triggered_data = checkpoint_data["frame_index_triggered"]

        if frame_index is not None:
            # Find and update specific frame index
            frame_indices = checkpoint_data.get("frame_index")
            if frame_indices is not None:
                # Convert to list if it's a tensor
                if isinstance(frame_indices, torch.Tensor):
                    indices = frame_indices.squeeze().tolist()
                else:
                    indices = frame_indices

                try:
                    idx = indices.index(frame_index)
                    # Update at that specific index
                    if isinstance(triggered_data, torch.Tensor):
                        triggered_data[idx] = torch.tensor([1], device=env.device, dtype=torch.float32)
                    else:
                        triggered_data[idx] = torch.tensor([1], device=env.device, dtype=torch.float32)
                    print(f"[checkpoint] Updated frame {frame_index} as triggered in HDF5")
                except ValueError:
                    print(f"[checkpoint] Frame {frame_index} not found in checkpoint data")
        else:
            # Update the most recent checkpoint
            try:
                if isinstance(triggered_data, torch.Tensor):
                    if len(triggered_data.shape) > 0 and triggered_data.shape[0] > 0:
                        triggered_data[-1] = torch.tensor([1], device=env.device, dtype=torch.float32)
                        print(f"[checkpoint] Updated most recent checkpoint as triggered in HDF5")
                elif isinstance(triggered_data, list):
                    if len(triggered_data) > 0:
                        triggered_data[-1] = torch.tensor([1], device=env.device, dtype=torch.float32)
                        print(f"[checkpoint] Updated most recent checkpoint as triggered in HDF5")
            except (IndexError, AttributeError) as e:
                print(f"[checkpoint] Failed to update triggered data: {e}")

    except Exception as e:
        print(f"[checkpoint] HDF5 update failed: {e}")


def save_checkpoint(env, checkpoint_path):
    """
    Save current frame index to checkpoint file.

    Args:
        env: The environment
        checkpoint_path: Path to save checkpoint

    Returns:
        int: frame index, -1 if failed
    """
    print("save checkpoint")
    try:
        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or 0 not in episodes:
            print("[checkpoint] save failed: recorder episode not ready (env_id=0)")
            return -1

        episode_data = episodes[0]
        frame_index = _get_current_frame_index(episode_data)
        if frame_index is None:
            print("[checkpoint] save failed: unable to infer current frame index from recorder")
            return -1

        payload = {"frame_index": int(frame_index)}
        for name, term in env.action_manager._terms.items():
            if hasattr(term, 'save_check_point'):
                payload[name] = term.save_check_point()
        torch.save(payload, checkpoint_path)
        print(f"[checkpoint] saved frame_index={frame_index} to: {checkpoint_path}")

        # Save checkpoint information to HDF5
        _save_checkpoint_to_hdf5(env, frame_index, payload)
        # Minimal UI update if env has label binding
        try:
            if hasattr(env, "_task_desc_label") and hasattr(env, "_base_desc") and env._task_desc_label is not None:
                env._task_desc_label.text = env._base_desc + f"\n Checkpoints: saved (frame {frame_index})"
        except Exception:
            pass
        return frame_index
    except Exception as e:
        print(f"[checkpoint] save failed: {e}")
        return -1


def load_checkpoint(env, checkpoint_path):
    """
    Load checkpoint and reset to saved frame with data preservation.

    Args:
        env: The environment
        checkpoint_path: Path to checkpoint file

    Returns:
        bool: True if successful, False otherwise
    """
    print("load checkpoint")
    # Minimal UI update: show loading state
    try:
        if hasattr(env, "_task_desc_label") and hasattr(env, "_base_desc") and env._task_desc_label is not None:
            env._task_desc_label.text = env._base_desc + "\n Checkpoints: loading..."
            if hasattr(env, 'sim') and hasattr(env.sim, 'render'):
                env.sim.render()
    except Exception:
        pass
    if not os.path.exists(checkpoint_path):
        print(f"[checkpoint] file not found: {checkpoint_path}")
        return False

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        frame_index = int(payload.get("frame_index", -1))
        if frame_index < 0:
            print(f"[checkpoint] load failed: invalid frame_index in payload: {payload}")
            return False

        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or 0 not in episodes:
            print("[checkpoint] load failed: recorder episode not ready (env_id=0)")
            return False

        episode_data = episodes[0]

        # Use enhanced reset_and_keep_to method for true load functionality
        success = reset_and_keep_to(env, episode_data, frame_index, payload)
        if success:
            print(f"[checkpoint] Successfully loaded frame_index={frame_index} from: {checkpoint_path}")
        else:
            print(f"[checkpoint] Failed to load frame_index={frame_index}")
        # Minimal UI update: show loaded or failed
        try:
            if hasattr(env, "_task_desc_label") and hasattr(env, "_base_desc") and env._task_desc_label is not None:
                if success:
                    env._task_desc_label.text = env._base_desc + f"\n Checkpoints: loaded (frame {frame_index})"
                else:
                    env._task_desc_label.text = env._base_desc + "\n Checkpoints: load failed"
                if hasattr(env, 'sim') and hasattr(env.sim, 'render'):
                    env.sim.render()
        except Exception:
            pass
        return success
    except Exception as e:
        print(f"[checkpoint] load failed: {e}")
        return False


def _get_current_frame_index(episode_data):
    """
    Infer current frame index from in-memory episode data.

    Strategy:
    1) Prefer the canonical path used by replay: states/articulation/robot/joint_position
    2) Fallback to finding the first leaf list or tensor under states and use its length - 1

    Args:
        episode_data: The episode data

    Returns:
        int or None: Current frame index, or None if cannot determine
    """
    try:
        data = episode_data._data
    except Exception:
        return None

    def get_len(node):
        if isinstance(node, list):
            return (len(node) - 1) if len(node) > 0 else None
        if isinstance(node, torch.Tensor):
            return (int(node.shape[0]) - 1) if node.ndim > 0 and node.shape[0] > 0 else None
        return None

    def get_from_path(d, path):
        cur = d
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return get_len(cur)

    preferred_paths = [
        ["states", "articulation", "robot", "joint_position"],
        ["states", "articulation", "robot", "root_state_w"],
    ]
    for p in preferred_paths:
        idx = get_from_path(data, p)
        if idx is not None:
            return idx

    def find_first_len(node):
        ln = get_len(node)
        if ln is not None:
            return ln
        if isinstance(node, dict):
            for val in node.values():
                res = find_first_len(val)
                if res is not None:
                    return res
        return None

    if isinstance(data, dict) and "states" in data:
        return find_first_len(data["states"])
    return None


def get_state_by_frame(episode_data, frame_index: int):
    """
    Return state dict at frame_index from in-memory episode.

    Prefer EpisodeData.get_state; if it raises due to list-based storage,
    reconstruct by indexing lists/tensors under the "states" subtree.

    Args:
        episode_data: The episode data
        frame_index: The frame index to retrieve

    Returns:
        dict or None: State data for the specified frame
    """
    # Try native helper first
    try:
        state = episode_data.get_state(frame_index)
        return state
    except Exception:
        pass

    # Fallback: reconstruct from raw in-memory structure
    data = getattr(episode_data, "_data", None)
    if not isinstance(data, dict) or "states" not in data:
        return None

    def index_node(node):
        if isinstance(node, dict):
            out = {}
            for k, v in node.items():
                val = index_node(v)
                if val is None:
                    return None
                out[k] = val
            return out
        if isinstance(node, list):
            if frame_index < 0 or frame_index >= len(node):
                return None
            return node[frame_index]
        if isinstance(node, torch.Tensor):
            if node.ndim == 0:
                return node
            if frame_index < 0 or frame_index >= int(node.shape[0]):
                return None
            return node[frame_index, None]
        # Unsupported leaf type
        return None

    return index_node(data["states"])


def quick_rewind(env, frames_back=10):
    """
    Quick rewind function that goes back a specified number of frames.

    Args:
        env: The environment
        frames_back: Number of frames to go back (default: 10)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        episodes = getattr(env.recorder_manager, "_episodes", None)
        if not isinstance(episodes, dict) or 0 not in episodes:
            print("[quick_rewind] Error: No episode data available")
            return False

        episode_data = episodes[0]
        current_frame = _get_current_frame_index(episode_data)
        if current_frame is None:
            print("[quick_rewind] Error: Cannot determine current frame")
            return False

        target_frame = max(0, current_frame - frames_back)
        print(f"[quick_rewind] Going back from frame {current_frame} to frame {target_frame}")

        return reset_and_keep_to(env, episode_data, target_frame, None)
    except Exception as e:
        print(f"[quick_rewind] Error: {e}")
        return False


def load_checkpoints_from_hdf5(episode_data):
    """
    Load all checkpoints from HDF5 episode data.

    Args:
        episode_data: EpisodeData object containing the recorded data

    Returns:
        list: List of checkpoint dictionaries, or empty list if no checkpoints found
    """
    try:
        if "obs" not in episode_data._data or "checkpoints" not in episode_data._data["obs"]:
            print("[checkpoint] No checkpoint data found in HDF5")
            return []

        checkpoint_data = episode_data._data["obs"]["checkpoints"]

        # Extract checkpoint information
        checkpoints = []
        if "frame_index" in checkpoint_data and "checkpoint_id" in checkpoint_data:
            frame_indices = checkpoint_data["frame_index"]
            checkpoint_ids = checkpoint_data["checkpoint_id"]

            for i in range(len(frame_indices)):
                checkpoint = {
                    "frame_index": frame_indices[i].item(),
                    "checkpoint_id": checkpoint_ids[i].item(),
                    "timestamp": checkpoint_ids[i].item()  # Use checkpoint_id as timestamp
                }

                # Add action manager terms if available
                for key, value in checkpoint_data.items():
                    if key.startswith("action_manager/"):
                        term_name = key.replace("action_manager/", "")
                        if i < len(value):
                            checkpoint[f"action_manager_{term_name}"] = value[i]

                checkpoints.append(checkpoint)

        print(f"[checkpoint] Loaded {len(checkpoints)} checkpoints from HDF5")
        return checkpoints

    except Exception as e:
        print(f"[checkpoint] Failed to load checkpoints from HDF5: {e}")
        return []


def get_checkpoint_at_frame(episode_data, target_frame):
    """
    Get the most recent checkpoint at or before the target frame.

    Args:
        episode_data: EpisodeData object containing the recorded data
        target_frame: Target frame index

    Returns:
        dict or None: Checkpoint dictionary, or None if no suitable checkpoint found
    """
    try:
        checkpoints = load_checkpoints_from_hdf5(episode_data)

        # Find the most recent checkpoint at or before target_frame
        suitable_checkpoints = [cp for cp in checkpoints if cp["frame_index"] <= target_frame]

        if not suitable_checkpoints:
            print(f"[checkpoint] No checkpoint found at or before frame {target_frame}")
            return None

        # Return the checkpoint with the highest frame_index
        best_checkpoint = max(suitable_checkpoints, key=lambda x: x["frame_index"])
        print(f"[checkpoint] Found checkpoint at frame {best_checkpoint['frame_index']} for target frame {target_frame}")

        return best_checkpoint

    except Exception as e:
        print(f"[checkpoint] Failed to get checkpoint at frame {target_frame}: {e}")
        return None


def get_action_by_frame(episode_data, frame_index: int):
    """
    Get action by frame index from episode data.

    Args:
        episode_data: The episode data
        frame_index: The frame index to retrieve

    Returns:
        dict or None: Action data for the specified frame
    """
    try:
        data = getattr(episode_data, "_data", None)
        if not isinstance(data, dict) or "actions" not in data:
            return None

        actions = data["actions"]

        def get_action_helper(actions, action_index):
            if isinstance(actions, dict):
                output_actions = {}
                for key, value in actions.items():
                    output_actions[key] = get_action_helper(value, action_index)
                    if output_actions[key] is None:
                        return None
            elif isinstance(actions, torch.Tensor):
                if action_index >= len(actions):
                    return None
                output_actions = actions[action_index]
            elif isinstance(actions, list):
                if action_index >= len(actions):
                    return None
                output_actions = actions[action_index]
            else:
                raise ValueError(f"Invalid action type: {type(actions)}")
            return output_actions

        return get_action_helper(actions, frame_index)
    except Exception as e:
        print(f"Error getting action at frame {frame_index}: {e}")
        return None


def get_joint_target_by_frame(episode_data, frame_index: int):
    """
    Get joint target by frame index from episode data.

    Args:
        episode_data: The episode data
        frame_index: The frame index to retrieve

    Returns:
        dict or None: Joint target data for the specified frame
    """
    try:
        data = getattr(episode_data, "_data", None)
        if not isinstance(data, dict) or "joint_targets" not in data:
            return None

        joint_targets = data["joint_targets"]

        def get_joint_target_helper(joint_targets, joint_target_index):
            if isinstance(joint_targets, dict):
                output_joint_targets = {}
                for key, value in joint_targets.items():
                    output_joint_targets[key] = get_joint_target_helper(value, joint_target_index)
                    if output_joint_targets[key] is None:
                        return None
            elif isinstance(joint_targets, torch.Tensor):
                if joint_target_index >= len(joint_targets):
                    return None
                output_joint_targets = joint_targets[joint_target_index]
            elif isinstance(joint_targets, list):
                if joint_target_index >= len(joint_targets):
                    return None
                output_joint_targets = joint_targets[joint_target_index]
            else:
                raise ValueError(f"Invalid joint target type: {type(joint_targets)}")
            return output_joint_targets

        return get_joint_target_helper(joint_targets, frame_index)
    except Exception as e:
        print(f"Error getting joint target at frame {frame_index}: {e}")
        return None


def get_raw_action_by_frame(episode_data, step_index):
    """Load complete action dictionary structure from HDF5 for replay.

    Args:
        episode_data: EpisodeData object containing the recorded data
        step_index: Index of the step to load

    Returns:
        Complete action dictionary, or None if loading fails
    """
    try:
        action_dict = {}

        # Check if raw_action data exists
        if "obs" not in episode_data._data or "raw_action" not in episode_data._data["obs"]:
            print("Warning: No raw_action data found in episode")
            return None

        raw_action_data = episode_data._data["obs"]["raw_action"]

        # Load each action component
        for key, tensor_data in raw_action_data.items():
            if isinstance(tensor_data, torch.Tensor):
                if step_index < len(tensor_data):
                    # Remove batch dimension and convert back to original format
                    value = tensor_data[step_index].squeeze(0)
                    action_dict[key] = value
                else:
                    print(f"Warning: Step index {step_index} out of range for action component '{key}'")
            else:
                print(f"Warning: Unexpected data type for action component '{key}': {type(tensor_data)}")

        return action_dict if action_dict else None

    except Exception as e:
        print(f"Error loading action dict from HDF5: {e}")
        return None


def get_raw_input_by_frame(episode_data, step_index):
    """Load raw input data from HDF5 for a specific frame.

    Args:
        episode_data: EpisodeData object containing the recorded data
        step_index: Index of the step to load

    Returns:
        dict or None: Raw input dictionary for the specified frame, or None if loading fails
    """
    try:
        raw_input_dict = {}

        # Check if raw_input data exists
        if "obs" not in episode_data._data or "raw_input" not in episode_data._data["obs"]:
            print("Warning: No raw_input data found in episode")
            return None

        raw_input_data = episode_data._data["obs"]["raw_input"]

        # Load each raw input component
        for key, tensor_data in raw_input_data.items():
            if isinstance(tensor_data, torch.Tensor):
                if step_index < len(tensor_data):
                    # Remove batch dimension and convert back to original format
                    value = tensor_data[step_index].squeeze(0)
                    # Convert to numpy if it's a tensor
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    raw_input_dict[key] = value
                else:
                    print(f"Warning: Step index {step_index} out of range for raw input component '{key}'")
            elif isinstance(tensor_data, dict):
                # Handle nested dictionaries (like controller states)
                nested_dict = {}
                for nested_key, nested_tensor in tensor_data.items():
                    if isinstance(nested_tensor, torch.Tensor) and step_index < len(nested_tensor):
                        nested_value = nested_tensor[step_index].squeeze(0)
                        if isinstance(nested_value, torch.Tensor):
                            nested_value = nested_value.cpu().numpy()
                        # Convert single-element arrays to scalars
                        if nested_value.size == 1:
                            nested_value = nested_value.item()
                        nested_dict[nested_key] = nested_value
                    else:
                        print(f"Warning: Step index {step_index} out of range for nested component '{key}/{nested_key}'")
                raw_input_dict[key] = nested_dict
            else:
                print(f"Warning: Unexpected data type for raw input component '{key}': {type(tensor_data)}")

        return raw_input_dict if raw_input_dict else None

    except Exception as e:
        print(f"Error loading raw input dict from HDF5: {e}")
        return None


def replay_from_raw_input(raw_input_dict: dict, device) -> dict | None:
    """Replay action generation from raw input data.

    Args:
        raw_input_dict: Raw input dictionary loaded from HDF5
        device: VR device instance for replay

    Returns:
        Generated action dictionary, or None if replay fails
    """
    try:
        if raw_input_dict is None:
            return None

        # Restore internal state first
        if "internal_state" in raw_input_dict:
            internal_state = raw_input_dict["internal_state"]
            for key, value in internal_state.items():
                if hasattr(device, key):
                    setattr(device, key, value)

        # Extract raw input components
        head_mat = raw_input_dict.get("head_mat")
        abs_left_wrist_mat = raw_input_dict.get("abs_left_wrist_mat")
        abs_right_wrist_mat = raw_input_dict.get("abs_right_wrist_mat")
        rel_left_wrist_mat = raw_input_dict.get("rel_left_wrist_mat")
        rel_right_wrist_mat = raw_input_dict.get("rel_right_wrist_mat")
        left_controller_state = raw_input_dict.get("left_controller_state", {})
        right_controller_state = raw_input_dict.get("right_controller_state", {})

        # Generate action using the same logic as input2action
        state = {}
        reset = state["reset"] = bool(device._reset_state)
        print(f"DEBUG: device._reset_state={device._reset_state}, reset={reset}")
        if reset:
            device._reset_state = False
            print("DEBUG: Returning reset state")
            return state

        # Apply the same processing logic as in input2action
        if right_controller_state.get("a_button", False):
            if device.is_body_moving_last_frame:
                device.is_body_moving = False
                # Note: body_lin_vel_w check would need robot state, skip for now
            device.is_body_moving_last_frame = device.is_body_moving

            if device.is_body_moving:
                abs_right_wrist_mat = device.before_a_button_abs_right_wrist_mat.copy()
                abs_left_wrist_mat = device.before_a_button_abs_left_wrist_mat.copy()
        else:
            device.is_body_moving_last_frame = True
            device.is_body_moving = False

        # Build state dictionary
        state["lpose_abs"] = torch.tensor(abs_left_wrist_mat, device=device.env.device, dtype=torch.float32)
        state["rpose_abs"] = torch.tensor(abs_right_wrist_mat, device=device.env.device, dtype=torch.float32)
        state["lpose_delta"] = torch.tensor(rel_left_wrist_mat, device=device.env.device, dtype=torch.float32)
        state["rpose_delta"] = torch.tensor(rel_right_wrist_mat, device=device.env.device, dtype=torch.float32)

        # Controller inputs
        state["lgrasp"] = left_controller_state.get("trigger", 0.0) * 2 - 1.0
        state["rgrasp"] = right_controller_state.get("trigger", 0.0) * 2 - 1.0
        state["lbase"] = torch.tensor([left_controller_state.get("thumbstick_x", 0.0),
                                       left_controller_state.get("thumbstick_y", 0.0)],
                                      device=device.env.device, dtype=torch.float32)
        state["rbase"] = torch.tensor([right_controller_state.get("thumbstick_x", 0.0),
                                       right_controller_state.get("thumbstick_y", 0.0)],
                                      device=device.env.device, dtype=torch.float32)
        state["lsqueeze"] = left_controller_state.get("squeeze", 0.0)
        state["rsqueeze"] = right_controller_state.get("squeeze", 0.0)
        state["rbase_button"] = right_controller_state.get("thumbstick", 0)
        state["lbase_button"] = left_controller_state.get("thumbstick", 0)

        state["x_button"] = left_controller_state.get("b_button", 0)
        state["y_button"] = left_controller_state.get("a_button", 0)

        # Handle base mode switching (same logic as in input2action)
        if left_controller_state.get("thumbstick", 0) == 1 and device.last_thumbstick_state == 0:
            device.base_mode_flag = 1 - device.base_mode_flag
            print(f"base_mode_flag 切换为: {device.base_mode_flag}")
        device.last_thumbstick_state = left_controller_state.get("thumbstick", 0)
        state["base_mode"] = device.base_mode_flag

        # Button press detection (same logic as in input2action)
        if left_controller_state.get("b_button", 0) == 1 and device.last_x_button_state == 0:
            print("x button pressed")
            state["x_button_pressed"] = True
        else:
            state["x_button_pressed"] = False
        device.last_x_button_state = left_controller_state.get("b_button", 0)

        if left_controller_state.get("a_button", 0) == 1 and device.last_y_button_state == 0:
            print("y button pressed")
            state["y_button_pressed"] = True
        else:
            state["y_button_pressed"] = False
        device.last_y_button_state = left_controller_state.get("a_button", 0)

        # Start/reset logic
        b_pressed = right_controller_state.get("b_button", 0)
        if not b_pressed and device.last_start_state:
            state["reset"] = device.started
            device.started = not device.started
        if state.get("reset", False):
            print("DEBUG: Returning None due to reset state")
            return None
        device.last_start_state = b_pressed

        state["started"] = device.started

        # Handle button press callbacks (same logic as in input2action)
        if state.get("x_button_pressed", False):
            if "N" in device._additional_callbacks:
                device._additional_callbacks["N"]()

        if state.get("y_button_pressed", False):
            if "M" in device._additional_callbacks:
                device._additional_callbacks["M"]()

        # Arm-specific processing
        if device.arm_count == 1:
            # Convert results to tensor
            state[f"{device.active_arm}_abs"] = torch.tensor(
                device.pose2action_xyzw(state["rpose_abs"]),
                device=state["rpose_abs"].device,
                dtype=state["rpose_abs"].dtype
            )
            state[f"{device.active_arm}_delta"] = torch.tensor(
                device.pose2action(state["rpose_delta"]),
                device=state["rpose_delta"].device,
                dtype=state["rpose_delta"].dtype
            )
            state[f"{device.active_arm}_gripper"] = state["rgrasp"]
        else:
            # Convert results to tensor
            state["left_arm_abs"] = torch.tensor(
                device.pose2action_xyzw(state["lpose_abs"]),
                device=state["lpose_abs"].device,
                dtype=state["lpose_abs"].dtype
            )
            state["left_arm_delta"] = torch.tensor(
                device.pose2action(state["lpose_delta"]),
                device=state["lpose_delta"].device,
                dtype=state["lpose_delta"].dtype
            )
            state["right_arm_abs"] = torch.tensor(
                device.pose2action_xyzw(state["rpose_abs"]),
                device=state["rpose_abs"].device,
                dtype=state["rpose_abs"].dtype
            )
            state["right_arm_delta"] = torch.tensor(
                device.pose2action(state["rpose_delta"]),
                device=state["rpose_delta"].device,
                dtype=state["rpose_delta"].dtype
            )
            state["left_gripper"] = state["lgrasp"]
            state["right_gripper"] = state["rgrasp"]

        return state

    except Exception as e:
        print(f"Error replaying from raw input: {e}")
        return None
