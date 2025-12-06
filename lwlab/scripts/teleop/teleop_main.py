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
Teleoperation Main Script with Enhanced Checkpoint System

This script provides a complete teleoperation system with an advanced checkpoint/load system
that allows users to save and load their progress during teleoperation sessions.

Key Features:
1. Standard Checkpoint System (SAVE/LOAD):
   - SAVE (M key): Save current frame index to checkpoint file
   - LOAD (N key): Load from checkpoint file and reset to saved frame

2. Enhanced Load System (reset_and_keep_to):
   - Preserves all episode data up to the target frame
   - Implements true "read checkpoint" functionality
   - Maintains recording history for replay and analysis

3. Quick Rewind (B key / REWIND):
   - Instantly go back 10 frames while preserving data
   - Useful for quick corrections during teleoperation

Usage:
- M key: Save checkpoint
- N key: Load checkpoint (with data preservation)
- B key: Quick rewind 10 frames (with data preservation)
- R key: Reset recording instance (clears all data)

The reset_and_keep_to function is the core of the enhanced system, providing:
- Complete episode data backup before reset
- Environment state restoration to target frame
- Episode data truncation to maintain consistency
- Comprehensive error handling and user feedback
"""

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
from pathlib import Path
import argparse
import os
import time
import torch
import json
import select
import sys
import copy
from datetime import datetime
from collections.abc import Callable
from lwlab.utils.log_utils import log_scene_rigid_objects, handle_exception_and_log, get_default_logger
from lwlab.utils.env import ExecuteMode
from isaaclab.app import AppLauncher


from lwlab.utils.profile_utils import trace_profile, DEBUG_FRAME_ANALYZER, debug_print
from lwlab.utils.config_loader import config_loader
from termcolor import colored
from lwlab.utils.render_utils import optimize_rendering
from robocasa_upload.joylo_uploader import JoyLoUploader
from lwlab.utils.teleop_utils import download_file


def _check_no_task_signal():
    """check if task is available"""
    try:
        import json
        from multiprocessing import shared_memory

        try:
            shm = shared_memory.SharedMemory(name="teleop_task_signal", create=False)

            data_bytes = bytes(shm.buf).split(b'\x00')[0]

            if data_bytes:
                signal_data = json.loads(data_bytes.decode('utf-8'))
                has_task = signal_data.get("has_task", True)

                if not has_task:
                    shm.buf[:] = b'\x00' * len(shm.buf)
                    return True

        except FileNotFoundError:
            return False
        except (json.JSONDecodeError, Exception) as e:
            pass

    except Exception as e:
        print(f"Error checking no-task signal: {e}")

    return False


def get_joint_pos_offset(env):
    import re
    joint_pos_offsets = {}
    init_state = env.cfg.robot_cfg.init_state
    if hasattr(init_state, 'joint_pos') and init_state.joint_pos:
        # Get all joint names from the robot
        robot = env.scene.articulations.get('robot')
        if robot is not None:
            all_joint_names = robot.joint_names
            # Process each pattern in joint_pos config
            for pattern, value in init_state.joint_pos.items():
                # Compile regex pattern
                regex = re.compile(pattern)
                # Match against all joint names
                for joint_name in all_joint_names:
                    if regex.match(joint_name):
                        joint_pos_offsets[joint_name] = float(value)
    return joint_pos_offsets


# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default=None, help="task config")
parser.add_argument("--checkpoint_path", type=str, default=None, help="path to save/load teleop checkpoint (env state)")
parser.add_argument("--auto_load_checkpoint", action="store_true", default=False, help="auto-load checkpoint on start if available")
parser.add_argument("--batch_name", type=str, default="default-batch", help="batch name for data collection")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)

app_launcher_args = vars(args_cli)
if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

if 'hand' in args_cli.teleop_device.lower() or args_cli.enable_pinocchio:
    import pinocchio

app_launcher_args["profiler_backend"] = ["tracy"]


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        current_time = time.time()
        next_wakeup_time = self.last_time + self.sleep_duration

        # if loop is too slow, jump to the next wakeup time
        if current_time >= next_wakeup_time:
            while self.last_time < current_time:
                self.last_time += self.sleep_duration
            return

        # calculate the sleep time
        sleep_time = next_wakeup_time - current_time

        # if sleep time is too short (less than 5ms), return
        if sleep_time < 0.005:
            self.last_time = next_wakeup_time
            return

        # for longer sleep time, use shorter sleep interval to keep responsiveness
        while time.time() < next_wakeup_time:
            remaining = next_wakeup_time - time.time()
            if remaining <= 0.001:  # if remaining time is less than 1ms, break
                break
            # use shorter sleep interval to avoid long blocking
            time.sleep(min(0.001, remaining * 0.5))

        self.last_time = next_wakeup_time


def log_sliding_window_delay_statistics(delay_stats, sliding_window_manager, args_cli):
    """Log sliding window delay statistics for async mode.

    Args:
        delay_stats: Dictionary containing delay statistics
        sliding_window_manager: Sliding window manager instance
        args_cli: Command line arguments containing delay configuration
    """
    if not args_cli.action_delay_async or delay_stats["async_actions_buffered"] <= 0:
        return

    logger = get_default_logger()
    logger.info("[Sliding Window Action Delay Statistics]")
    logger.info("  Sliding window mode: ENABLED")
    logger.info(f"  Actions buffered: {delay_stats['async_actions_buffered']}")
    logger.info(f"  Actions executed: {delay_stats['async_actions_executed']}")
    logger.info(f"  Window utilization: {delay_stats['buffer_utilization']:.2%}")
    logger.info(f"  Window size: {args_cli.action_buffer_size}")

    # Get detailed window statistics
    window_stats = sliding_window_manager.get_window_stats()
    if window_stats['window_size'] > 0:
        logger.info(f"  Current window size: {window_stats['window_size']}")
        logger.info(f"  Average delay in window: {window_stats['avg_delay']:.3f}s")
        logger.info(f"  Oldest delay: {window_stats['oldest_delay']:.3f}s")
        logger.info(f"  Newest delay: {window_stats['newest_delay']:.3f}s")
        logger.info(f"  Total executions: {window_stats['execution_count']}")

    logger.info(f"  Delay type: {args_cli.action_delay_type}")
    logger.info(f"  Base delay: {args_cli.action_delay_time:.3f}s")


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    teleoperation_active = None
    should_reset_env = None
    should_reset_env_keep_placement = None
    should_remake_env = None
    should_remake_env_with_fixed_layout = None
    flush_recorder_manager_flag = None
    rollback_to_checkpoint_flag = None

    def reset_env():
        print("reset env and random placement", flush=True)
        nonlocal should_reset_env
        should_reset_env = True

    def reset_env_keep_placement():
        print("reset env and keep placement", flush=True)
        nonlocal should_reset_env_keep_placement
        should_reset_env_keep_placement = True

    def remake_env_with_random_layout():
        print("remake env with random layout", flush=True)
        nonlocal should_remake_env
        should_remake_env = True

    def remake_env_with_fixed_layout():
        print("remake env with fixed layout", flush=True)
        nonlocal should_remake_env_with_fixed_layout
        should_remake_env_with_fixed_layout = True

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True
        if teleop_interface is not None and hasattr(teleop_interface, "send_robot_state"):
            teleop_interface.reset()

    def stop_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = False

    def flush_recorder_manager():
        nonlocal flush_recorder_manager_flag
        flush_recorder_manager_flag = True

    def call_save_checkpoint():
        frame_index = save_checkpoint(teleop_interface.env, args_cli.checkpoint_path)
        if isinstance(teleop_interface, VRController):
            teleop_interface.set_checkpoint_frame_idx(frame_index)

    class SlidingWindowActionDelay:
        """Asynchronous action delay system with sliding window implementation"""

        def __init__(self):
            self.action_window = []  # Sliding window of actions
            self.window_size = args_cli.action_buffer_size
            self.current_time = 0.0
            self.last_executed_time = 0.0
            self.execution_history = []  # Track execution timing

        def calculate_delay(self):
            """Calculate delay based on configuration"""
            if args_cli.action_delay_time <= 0.0:
                return 0.0

            import random
            import numpy as np

            delay = args_cli.action_delay_time

            if args_cli.action_delay_type == "random":
                delay = random.uniform(0, args_cli.action_delay_time)

            return delay

        def add_action_to_window(self, action, current_time):
            """Add action to sliding window with calculated delay"""
            if not args_cli.action_delay_async or args_cli.action_delay_time <= 0.0:
                return action

            delay = self.calculate_delay()
            execute_time = current_time + delay

            # Add to sliding window
            window_entry = {
                'action': action,
                'execute_time': execute_time,
                'delay': delay,
                'added_time': current_time,
                'index': len(self.action_window)
            }

            self.action_window.append(window_entry)

            # Maintain sliding window size
            if len(self.action_window) > self.window_size:
                # Remove oldest entry (sliding window behavior)
                removed = self.action_window.pop(0)
                # Update indices
                for i, entry in enumerate(self.action_window):
                    entry['index'] = i

            return None  # No immediate action

        def get_action_from_window(self, current_time):
            """Get action from sliding window based on timing and window strategy"""
            if not self.action_window:
                return None

            # Strategy 1: Execute ready actions (FIFO)
            ready_actions = [entry for entry in self.action_window
                             if entry['execute_time'] <= current_time]

            if ready_actions:
                # Execute the oldest ready action
                ready_actions.sort(key=lambda x: x['execute_time'])
                executed_entry = ready_actions[0]

                # Remove from window
                self.action_window = [entry for entry in self.action_window
                                      if entry['index'] != executed_entry['index']]

                # Update indices
                for i, entry in enumerate(self.action_window):
                    entry['index'] = i

                # Record execution
                self.execution_history.append({
                    'execute_time': current_time,
                    'delay': executed_entry['delay'],
                    'window_position': executed_entry['index']
                })

                return executed_entry['action']

            # Strategy 2: Use sliding window interpolation
            return self._interpolate_from_window(current_time)

        def _interpolate_from_window(self, current_time):
            """Interpolate action from sliding window based on timing"""
            if not self.action_window:
                return None

            # Find the closest actions in time
            future_actions = [entry for entry in self.action_window
                              if entry['execute_time'] > current_time]
            past_actions = [entry for entry in self.action_window
                            if entry['execute_time'] <= current_time]

            if past_actions:
                # Use the most recent past action
                past_actions.sort(key=lambda x: x['execute_time'], reverse=True)
                return past_actions[0]['action']

            if future_actions:
                # Use the nearest future action
                future_actions.sort(key=lambda x: x['execute_time'])
                return future_actions[0]['action']

            # Fallback: use the latest action in window
            return self.action_window[-1]['action']

        def get_window_stats(self):
            """Get sliding window statistics"""
            if not self.action_window:
                return {
                    'window_size': 0,
                    'oldest_delay': 0,
                    'newest_delay': 0,
                    'avg_delay': 0,
                    'execution_count': len(self.execution_history)
                }

            delays = [entry['delay'] for entry in self.action_window]
            return {
                'window_size': len(self.action_window),
                'oldest_delay': min(delays),
                'newest_delay': max(delays),
                'avg_delay': sum(delays) / len(delays),
                'execution_count': len(self.execution_history)
            }

    # Create sliding window delay manager
    sliding_window_manager = SlidingWindowActionDelay()

    def rollback_to_checkpoint():
        nonlocal rollback_to_checkpoint_flag
        rollback_to_checkpoint_flag = True

    def save_metrics(env):
        """Save metrics data to JSON file"""
        metrics_data = env.cfg.isaaclab_arena_env.task.get_checker_results()

        # Save metrics to JSON file
        if metrics_data:
            metrics_file_path = os.path.join(output_dir, "metrics.json")
            try:
                with open(metrics_file_path, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                print(f"Metrics saved to: {metrics_file_path}")
            except Exception as e:
                print(f"Failed to save metrics: {e}")

    # Global variable to track upload dialog state
    upload_dialog_state = {"shown": False, "result": None, "window": None}

    def create_teleop_interface(env):
        """Create teleoperation interface based on device type."""
        if args_cli.headless:
            return None

        env_cfg = env.cfg
        nonlocal teleoperation_active

        teleoperation_callbacks: dict[str, Callable[[], None]] = {
            "X": reset_env,
            "R": reset_env_keep_placement,
            "Y": remake_env_with_random_layout,
            "U": remake_env_with_fixed_layout,
            "T": flush_recorder_manager,
            "M": call_save_checkpoint,
            "N": rollback_to_checkpoint,
            "B": start_teleoperation,
            # TODO not enable now
            # Add new shortcut: quick rewind 10 frames
            "P": lambda: quick_rewind(teleop_interface.env, 10),
        }

        if hasattr(env_cfg.isaaclab_arena_env.embodiment, "teleop_devices") and args_cli.teleop_device in env_cfg.isaaclab_arena_env.embodiment.teleop_devices.devices:
            teleoperation_active = False
            teleop_interface = create_teleop_device(
                env, args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
            if hasattr(teleop_interface, "init_internal_state"):
                teleop_interface.init_internal_state(get_joint_pos_offset(env), env.scene.articulations["robot"].data.joint_names)
        else:
            if args_cli.teleop_device.lower() == "keyboard":
                device_type = KEYCONTROLLER_MAP[args_cli.teleop_device.lower() + "-" + args_cli.robot.lower().split("-")[0]]
                teleop_interface = device_type(
                    pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity,
                    base_sensitivity=0.5 * args_cli.sensitivity, base_yaw_sensitivity=0.8 * args_cli.sensitivity
                )
                teleop_interface.env = env
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(env,
                                                 pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.2 * args_cli.sensitivity
                                                 )
                teleop_interface.env = env
            # elif args_cli.teleop_device.lower() == "gamepad":
            #     teleop_interface = Se3Gamepad(
            #         pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
            #     )
            elif args_cli.teleop_device.lower() == "so101leader":
                from lwlab.core.devices.lerobot import SO101Leader
                teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
            elif args_cli.teleop_device.lower() == "bi_so101leader":
                from lwlab.core.devices.lerobot import BiSO101Leader
                teleop_interface = BiSO101Leader(env, left_port=args_cli.left_port, right_port=args_cli.right_port, recalibrate=args_cli.recalibrate)
            elif args_cli.teleop_device.lower() == "dualhandtracking_abs" and args_cli.robot.lower().endswith("hand"):
                # Create hand tracking device with retargeter
                teleop_interface = LwOpenXRDevice(
                    env_cfg.xr,
                    retargeters=[],
                    env=env,
                )
                teleoperation_active = False
            elif args_cli.teleop_device.lower().startswith("vr"):
                teleoperation_active = True
                image_size = (720, 1280)
                shm = shared_memory.SharedMemory(
                    create=True,
                    size=image_size[0] * image_size[1] * 3 * np.uint8().itemsize,
                )
                vr_device_type = {
                    "vr-controller": VRController,
                    "vr-hand": VRHand,
                }[args_cli.teleop_device.lower()]
                teleop_interface = vr_device_type(env,
                                                  img_shape=image_size,
                                                  shm_name=shm.name,
                                                  relative_control=args_cli.relative_control)
            else:
                raise ValueError(
                    f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
                )

            # Add callbacks to teleop device
            if not args_cli.headless and isinstance(teleop_interface, LwOpenXRDevice):
                teleoperation_active = False
                teleoperation_callbacks: dict[str, Callable[[], None]] = {
                    "RESET": reset_env,
                    "START": start_teleoperation,
                    "STOP": stop_teleoperation,
                    "SAVE": lambda: save_checkpoint(teleop_interface.env, args_cli.checkpoint_path),
                    "LOAD": lambda: load_checkpoint(teleop_interface.env, args_cli.checkpoint_path),
                    # Add new shortcut: quick rewind 10 frames
                    "REWIND": lambda: quick_rewind(teleop_interface.env, 10),
                }
            elif teleop_interface is not None:
                if args_cli.teleop_device.lower().startswith("vr"):
                    teleoperation_active = True
                else:
                    teleoperation_active = False
                for key, callback in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, callback)
                    except (ValueError, TypeError) as e:
                        omni.log.warn(f"Failed to add callback for key {key}: {e}")
            else:
                teleoperation_active = False

        if args_cli.headless:
            teleoperation_active = True

        print(teleop_interface)

        return teleop_interface

    def create_env_config(object_cfgs=None, cache_usd_version=None, scene_name=None, initial_state=None):
        """Create environment configuration based on task type."""
        import omni.usd
        omni.usd.get_context().new_stage()
        with trace_profile("parse_env_cfg"):
            # Load replay dataset
            if hasattr(args_cli, "project_name") and hasattr(args_cli, "batch_name"):
                uploader = JoyLoUploader(project_name=args_cli.project_name)
                task_info = uploader.pull_task(batch_name=args_cli.batch_name)
                if task_info is None:
                    raise ValueError(f"Unable to obtain task information: batch_name='{args_cli.batch_name}' May not exist or may have been completed")
                if task_info.get("hdf5_url", None) is not None:
                    tmp_dir = str(Path(__file__).parent.parent.parent.parent / 'tmp')
                    download_file(task_info["hdf5_url"], tmp_dir, "input_dataset.hdf5")
                    args_cli.input_dataset_file = str(Path(tmp_dir) / "input_dataset.hdf5")
                else:
                    if hasattr(args_cli, "input_dataset_file"):
                        delattr(args_cli, 'input_dataset_file')
            if hasattr(args_cli, "input_dataset_file") and os.path.exists(args_cli.input_dataset_file):
                dataset_file_handler = HDF5DatasetFileHandler()
                dataset_file_handler.open(args_cli.input_dataset_file)
                env_args = json.loads(dataset_file_handler._hdf5_data_group.attrs["env_args"])
                scene_backend = env_args["scene_backend"] if "scene_backend" in env_args else "robocasa"
                task_backend = env_args["task_backend"] if "task_backend" in env_args else "robocasa"
                task_name = env_args["task_name"].strip()
                robot_name = env_args["robot_name"]
                args_cli.robot = robot_name
                scene_name = f"{env_args['scene_type']}-{env_args['layout_id']}-{env_args['style_id']}"
                usd_simplify = env_args["usd_simplify"] if 'usd_simplify' in env_args else False
                episode_count = dataset_file_handler.get_num_episodes()
                episode_indices_to_replay = list(range(episode_count))
                # prepare video writer
                episode_names = list(dataset_file_handler.get_episode_names())
                episode_names.sort(key=lambda x: int(x.split("_")[-1]))

                episode_names_to_replay = []
                for idx in episode_indices_to_replay:
                    episode_names_to_replay.append(episode_names[idx])
                dataset_file_handler.close()
                env_cfg = parse_env_cfg(
                    scene_backend=scene_backend,
                    task_backend=task_backend,
                    task_name=task_name,
                    robot_name=robot_name,
                    scene_name=scene_name,
                    robot_scale=args_cli.robot_scale,
                    device=args_cli.device,
                    num_envs=args_cli.num_envs,
                    use_fabric=not args_cli.disable_fabric,
                    replay_cfgs={"hdf5_path": args_cli.input_dataset_file, "ep_meta": env_args, "ep_names": episode_names_to_replay},
                    first_person_view=args_cli.first_person_view,
                    enable_cameras=app_launcher._enable_cameras,
                    execute_mode=ExecuteMode.REPLAY_TELEOP,
                    usd_simplify=usd_simplify,
                    seed=env_args["seed"] if "seed" in env_args else None,
                    sources=env_args["sources"] if "sources" in env_args else None,
                    object_projects=env_args["object_projects"] if "object_projects" in env_args else None,
                    headless_mode=args_cli.headless,
                )
            else:
                # Build replay_cfgs from YAML if initial pose is provided
                ep_meta = {}
                if hasattr(args_cli, "init_robot_base_pos") and args_cli.init_robot_base_pos is not None:
                    try:
                        pos = [float(v) for v in args_cli.init_robot_base_pos]
                        if len(pos) == 3:
                            ep_meta["init_robot_base_pos"] = pos
                    except Exception:
                        pass
                if hasattr(args_cli, "init_robot_base_ori") and args_cli.init_robot_base_ori is not None:
                    try:
                        ori = [float(v) for v in args_cli.init_robot_base_ori]
                        if len(ori) == 3:
                            ep_meta["init_robot_base_ori"] = ori
                    except Exception:
                        pass
                if object_cfgs is not None:
                    ep_meta["object_cfgs"] = object_cfgs
                if cache_usd_version is not None:
                    ep_meta["cache_usd_version"] = cache_usd_version
                replay_cfgs = None
                if len(ep_meta) > 0:
                    replay_cfgs = {"ep_meta": ep_meta}
                object_init_offset = [0.0, 0.0]
                if hasattr(args_cli, "object_init_offset") and args_cli.object_init_offset is not None:
                    try:
                        object_init_offset = [float(v) for v in args_cli.object_init_offset]
                    except Exception:
                        pass
                if scene_name is None:
                    scene_name = args_cli.layout
                kwargs = {"debug_assets": args_cli.debug_assets}
                if args_cli.debug_assets == "object":
                    execute_mode = ExecuteMode.TEST_OBJECT
                    kwargs["test_object_paths"] = args_cli.test_object_paths
                elif args_cli.debug_assets == "fixture":
                    execute_mode = ExecuteMode.TEST_FIXTURE
                    kwargs["test_fixture_path"] = args_cli.test_fixture_path
                    kwargs["test_fixture_type"] = args_cli.test_fixture_type
                else:
                    execute_mode = ExecuteMode.TELEOP
                env_cfg = parse_env_cfg(
                    scene_backend=args_cli.scene_backend,
                    task_backend=args_cli.task_backend,
                    task_name=args_cli.task,
                    robot_name=args_cli.robot,
                    scene_name=scene_name,
                    robot_scale=args_cli.robot_scale,
                    device=args_cli.device,
                    num_envs=args_cli.num_envs,
                    use_fabric=not args_cli.disable_fabric,
                    replay_cfgs=replay_cfgs,
                    first_person_view=args_cli.first_person_view,
                    enable_cameras=app_launcher._enable_cameras,
                    execute_mode=execute_mode,
                    usd_simplify=args_cli.usd_simplify,
                    object_init_offset=object_init_offset,
                    max_scene_retry=args_cli.max_scene_retry,
                    max_object_placement_retry=args_cli.max_object_placement_retry,
                    seed=args_cli.seed,
                    sources=args_cli.sources,
                    object_projects=args_cli.object_projects,
                    headless_mode=args_cli.headless,
                    initial_state=initial_state,
                    teleop_device=args_cli.teleop_device,
                    resample_objects_placement_on_reset=args_cli.resample_objects_placement_on_reset,
                    resample_robot_placement_on_reset=args_cli.resample_robot_placement_on_reset,
                    **kwargs,
                )
            env_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"
            env_cfg.env_name = env_name
            env_cfg.terminations.time_out = None
            if args_cli.record:
                env_cfg.recorders.dataset_export_dir_path = output_dir
                env_cfg.recorders.dataset_filename = output_file_name

        # Unregister existing environment if it exists
        if env_name in gym.envs.registry:
            gym.envs.registry.pop(env_name)

        # Register the environment
        gym.register(
            id=env_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )
        # Apply configuration modifications
        # Create new stage and environment
        new_env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped
        set_seed(env_cfg.seed, new_env)
        return new_env

    def remake_env(env, teleop_interface, viewports, scene_name=None):
        """Reset environment by creating a new one with fresh configuration."""
        # Reset object cache to ensure reload new object samples
        reset_obj_cache()

        # Close current environment
        env.close()
        # Create fresh configuration using the same logic
        new_env = create_env_config(scene_name=scene_name)
        get_default_logger().info(f"env_cfg: {new_env.cfg}")

        reset_physx(new_env)
        new_env.reset(seed=new_env.cfg.seed)

        if args_cli.enable_rendering_optimization:
            optimize_rendering(new_env, args_cli)
        if teleop_interface is not None:
            teleop_interface.refresh_env(new_env)
            teleop_interface.reset()
        if not args_cli.headless:
            viewports, overlay_window = setup_env_config_with_args(new_env, viewports)
        else:
            viewports, overlay_window = None, None
        return new_env, teleop_interface, viewports, overlay_window

    def setup_env_config_with_args(env, viewports=None):
        isaaclab_arena_env = env.cfg.isaaclab_arena_env
        if not args_cli.headless and args_cli.enable_cameras and args_cli.enable_multiple_viewports:
            viewports = setup_cameras(env, viewports)
            for key, v_p in viewports.items():
                res = v_p.viewport_api.get_texture_resolution()
                sca = v_p.viewport_api.get_texture_resolution_scale()
                v_p.viewport_api.set_texture_resolution((426, 240))
                env.sim.render()
                res_new = v_p.viewport_api.get_texture_resolution()
                print(f"Viewport {key} resolution: {res}, scale: {sca}, new resolution: {res_new}")
        if not args_cli.headless:
            overlay_window = setup_task_description_ui(env)
        return viewports, overlay_window

    def run_simulation(env, teleop_interface):
        env_cfg = env.cfg
        # add teleoperation key for env reset
        nonlocal should_reset_env
        nonlocal should_reset_env_keep_placement
        nonlocal should_remake_env
        nonlocal should_remake_env_with_fixed_layout
        nonlocal flush_recorder_manager_flag
        nonlocal rollback_to_checkpoint_flag
        nonlocal teleoperation_active
        should_reset_env = False
        should_reset_env_keep_placement = False
        should_remake_env = False
        should_remake_env_with_fixed_layout = False
        flush_recorder_manager_flag = False
        rollback_to_checkpoint_flag = False

        ci_start_flag = None
        rate_limiter = RateLimiter(args_cli.step_hz)

        # reset environment
        initial_state = None
        reset_physx(env)
        env.reset(seed=env.cfg.seed)
        print(colored("Reset env successfully", "green"))
        initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data.get("initial_state", None))

        if teleop_interface is not None:
            teleop_interface.reset()
        # auto load checkpoint if enabled
        if getattr(args_cli, "auto_load_checkpoint", False) and os.path.exists(args_cli.checkpoint_path):
            load_checkpoint(env, args_cli.checkpoint_path)

        viewports = None
        overlay_window = None
        if not args_cli.headless:
            viewports, overlay_window = setup_env_config_with_args(env)

        print(colored(env.cfg.isaaclab_arena_env.orchestrator.get_ep_meta()["lang"], "green"))

        current_recorded_demo_count = 0
        success_step_count = 0
        frame_count = 0
        start_record_state = False
        # Initialize video recorder
        video_recorder = None
        if args_cli.save_video:
            video_recorder = VideoRecorder(args_cli.video_save_dir, args_cli.video_fps, args_cli.task, args_cli.robot, args_cli.layout)
        if args_cli.enable_debug_log:
            log_path = log_scene_rigid_objects(env)
        # add frame rate analyzer (only in debug mode)
        frame_analyzer = DEBUG_FRAME_ANALYZER
        debug_print("Frame rate analyzer initialized in debug mode")

        vis_helper_prims = []
        initial_state = None

        # Initialize delay statistics (only for async mode)
        delay_stats = {
            "async_actions_buffered": 0,
            "async_actions_executed": 0,
            "buffer_utilization": 0.0
        }
        action_idx = 0

        # simulate environment
        while simulation_app.is_running():
            if _check_no_task_signal():
                print(colored("No task available signal received, stopping teleoperation", "yellow"))
                break

            if args_cli.headless and teleop_interface is None:  # for CI test
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    command = sys.stdin.readline().strip()
                    print(f"get command is {command}", flush=True)
                    if command.lower() == "r":
                        reset_env()
                        action_idx = 0

                    if command.lower() == "x":
                        reset_env_keep_placement()
                        action_idx = 0

                    if command.lower() == "u":
                        remake_env_with_fixed_layout()

                    if command.lower() == "b":
                        start_teleoperation()
                        ci_start_flag = True

                    if command.lower() == "t":
                        flush_recorder_manager()

            if flush_recorder_manager_flag:
                env.recorder_manager.export_episodes()
                env.recorder_manager.reset()
                flush_recorder_manager_flag = False

            # start frame analysis (debug mode only)
            frame_analyzer.start_frame()

            # run everything in inference mode
            # with torch.inference_mode():
            teleop_start = time.time()
            if teleop_interface is None and args_cli.teleop_device == "vr-controller":
                if ci_start_flag:
                    with open("ci_run/g1wbc_action_data/actions.txt", "r") as f:
                        actions_history = [torch.tensor(eval(line.strip())) for line in f.readlines()]
                    if action_idx < len(actions_history):
                        actions = actions_history[action_idx]
                        action_idx += 1
                    else:
                        break
                else:
                    actions = None

            elif teleop_interface is None:
                if env.cfg.isaaclab_arena_env.embodiment.get_default_action(env.device, env.num_envs) is not None:
                    actions = env.cfg.isaaclab_arena_env.embodiment.get_default_action(env.device, env.num_envs)
                else:
                    actions = torch.zeros(env.action_space.shape)
            else:
                actions = teleop_interface.advance()
                # safety coding
                if isinstance(actions, torch.Tensor) and actions.shape[0] == 1 and env.num_envs != actions.shape[0]:
                    actions = actions.repeat(env.num_envs, 1)

            if actions is not None and hasattr(actions, 'clone'):
                env.latest_action = actions.clone()
            else:
                env.latest_action = None

            teleop_time = time.time() - teleop_start
            frame_analyzer.record_stage('teleop_advance', teleop_time)

            # Handle sliding window action delay (only async mode supported)
            current_time = time.time()
            if actions is not None and args_cli.action_delay_async:
                # Add action to sliding window
                delayed_action = sliding_window_manager.add_action_to_window(actions, current_time)
                if delayed_action is not None:
                    actions = delayed_action
                    delay_stats["async_actions_executed"] += 1
                else:
                    delay_stats["async_actions_buffered"] += 1
                    # Get action from sliding window
                    window_action = sliding_window_manager.get_action_from_window(current_time)
                    if window_action is not None:
                        actions = window_action
                        delay_stats["async_actions_executed"] += 1

                # Update window statistics
                window_stats = sliding_window_manager.get_window_stats()
                delay_stats["buffer_utilization"] = window_stats['window_size'] / args_cli.action_buffer_size

            if (
                actions is None
                or should_reset_env
                or should_reset_env_keep_placement
                or should_remake_env_with_fixed_layout
                or should_remake_env
            ):
                destroy_robot_vis_helper(vis_helper_prims, env)
                if should_reset_env_keep_placement:
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = True
                    reset_physx(env)
                    from lwlab.utils.teleop_utils import convert_list_to_2d_tensor
                    initial_state_converted = convert_list_to_2d_tensor(initial_state)
                    env.reset_to(initial_state_converted, torch.tensor([0], device=env.device), seed=env.cfg.seed, is_relative=True)
                    if teleop_interface is not None:
                        teleop_interface.reset()

                    should_reset_env_keep_placement = False
                    action_idx = 0
                    ci_start_flag = None
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = False
                elif should_reset_env:
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = True
                    reset_physx(env)
                    env.reset(seed=env.cfg.seed)
                    if teleop_interface is not None:
                        teleop_interface.reset()
                    should_reset_env = False
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = False
                elif should_remake_env:
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = True
                    env, teleop_interface, viewports, overlay_window = remake_env(env, teleop_interface, viewports)
                    env_cfg = env.cfg
                    should_remake_env = False
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = False
                elif should_remake_env_with_fixed_layout:
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = True
                    if env.cfg.layout_id and env.cfg.style_id and env.cfg.scene_type:
                        scene_name = f"{env.cfg.scene_type}-{env.cfg.layout_id}-{env.cfg.style_id}"
                    else:
                        scene_name = args_cli.layout
                    env, teleop_interface, viewports, overlay_window = remake_env(env, teleop_interface, viewports, scene_name)
                    env_cfg = env.cfg
                    should_remake_env_with_fixed_layout = False
                    env.cfg.isaaclab_arena_env.force_reset_env_enabled = False
                initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data.get("initial_state", None))

                if not args_cli.headless:
                    update_task_desc(env)

                frame_count = 0
                # Reset demo counter for new session
                current_recorded_demo_count = 0
                if start_record_state:
                    print("Stop Recording!!!")
                    save_metrics(env)
                    start_record_state = False
                    if video_recorder is not None:
                        video_recorder.stop_recording()

            elif not args_cli.headless and ((isinstance(actions, bool) and not actions) or (not teleoperation_active)):
                env.render()
            # apply actions
            else:
                if not start_record_state:
                    print("Start Recording!!!", flush=True)
                    start_record_state = True

                    if not args_cli.headless:
                        vis_helper_prims = spawn_robot_vis_helper_general(env)
                    # Initialize video recording
                    if video_recorder is not None:
                        camera_data, camera_name = get_camera_images(env)
                        if camera_name is not None:
                            image_shape = (camera_data.shape[0], camera_data.shape[1])  # (height, width)
                            video_recorder.start_recording(camera_name, image_shape)

                warmup_rendering(env)
                frame_count += 1
                step_start = time.time()
                if rollback_to_checkpoint_flag:
                    load_checkpoint(env, args_cli.checkpoint_path)
                    rollback_to_checkpoint_flag = False

                if not args_cli.headless and isinstance(teleop_interface, VRController):
                    if teleop_interface.tv.left_controller_state["b_button"]:
                        if teleop_interface.get_rollback_action() is not None:
                            actions = teleop_interface.get_rollback_action()
                        else:
                            continue

                with trace_profile(f"env_step_frame_{frame_count:06d}"):
                    carb.profiler.begin(1, "env_step")
                    if actions is None:
                        continue
                    obs, *_ = env.step(actions)
                    if initial_state is None:
                        initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data.get("initial_state", None))
                    carb.profiler.end(1)
                step_time = time.time() - step_start
                frame_analyzer.record_stage('env_step', step_time)

                update_checkers_status(env, env_cfg.isaaclab_arena_env.task.get_warning_text())

                # Recorded
                if args_cli.enable_cameras and start_record_state and video_recorder is not None:
                    camera_data, camera_name = get_camera_images(env)
                    if camera_name is not None:
                        video_recorder.add_frame(camera_data)
                    video_recorder.frame_count += 1

                # print out the current demo count if it has changed
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

                if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                    save_metrics(env)
                    print(f"All {args_cli.num_demos} demonstrations recorded. Resetting environment for new session.")

                    if (not args_cli.continue_teleop_after_success) or args_cli.headless:
                        break

                    # Show upload dialog to user (non-blocking)
                    if not upload_dialog_state["shown"]:
                        show_upload_dialog(upload_dialog_state)
                        print("Upload dialog shown. Please make your choice in the dialog window.")

                    # Check if user has made a choice
                    if upload_dialog_state["result"] is not None:
                        upload_choice = upload_dialog_state["result"]
                        # Hide the dialog window before resetting state
                        if upload_dialog_state["window"] is not None:
                            upload_dialog_state["window"].visible = False
                        # Reset dialog state
                        upload_dialog_state["shown"] = False
                        upload_dialog_state["result"] = None
                        upload_dialog_state["window"] = None
                    else:
                        continue

                    if args_cli.teleop_device.lower().startswith("vr"):
                        teleoperation_active = True
                    else:
                        teleoperation_active = False

                    if start_record_state:
                        start_record_state = False

                    if upload_choice:
                        # User chose to upload - continue with existing upload logic
                        try:
                            import shutil
                            original_hdf5_path = args_cli.dataset_file
                            success_hdf5_path = os.path.splitext(original_hdf5_path)[0] + "_success.hdf5"

                            if os.path.exists(original_hdf5_path):
                                shutil.move(original_hdf5_path, success_hdf5_path)
                                print(f"Renamed dataset file to: {success_hdf5_path}")

                                # Send signal to teleop_launcher for upload using upload shared memory
                                try:
                                    from multiprocessing import shared_memory
                                    import json

                                    # Create upload signal data
                                    upload_signal_data = {
                                        "action": "upload_success",
                                        "hdf5_path": success_hdf5_path,
                                        "timestamp": time.time(),
                                        "has_success_file": True
                                    }

                                    # Serialize signal data
                                    signal_json = json.dumps(upload_signal_data)
                                    signal_bytes = signal_json.encode('utf-8')

                                    # Create or access upload shared memory
                                    try:
                                        # Try to access existing upload shared memory
                                        shm = shared_memory.SharedMemory(name="teleop_upload_signal", create=False)
                                    except FileNotFoundError:
                                        # Create new upload shared memory if it doesn't exist
                                        shm = shared_memory.SharedMemory(name="teleop_upload_signal", create=True, size=4096)

                                    # Write signal data to upload shared memory
                                    shm.buf[:len(signal_bytes)] = signal_bytes
                                    shm.buf[len(signal_bytes)] = 0  # Null terminator

                                    print(f"Upload signal sent via upload shared memory: {upload_signal_data}")

                                except Exception as signal_e:
                                    print(f"Error sending upload signal via shared memory: {signal_e}")
                            else:
                                print(f"Warning: Original dataset file not found: {original_hdf5_path}")
                        except Exception as e:
                            print(f"Error renaming dataset file: {e}")
                        remake_env_with_random_layout()
                    else:
                        env, teleop_interface, viewports, overlay_window = remake_env(env, teleop_interface, viewports)
                        initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data.get("initial_state", None))
                        env_cfg = env.cfg

            if not args_cli.headless and teleoperation_active:
                env.sim.render()

            # only use rate_limiter when needed, don't let it limit rendering
            if rate_limiter and teleoperation_active:
                rate_start = time.time()
                with trace_profile("rate_limiter"):
                    rate_limiter.sleep(env)
                rate_time = time.time() - rate_start
                frame_analyzer.record_stage('rate_limiter', rate_time)

            # send robot state if needed
            if hasattr(teleop_interface, "send_robot_state"):
                teleop_interface.send_robot_state()

            # end frame analysis (debug mode only)
            frame_analyzer.end_frame()

        # ensure to stop recording before exiting
        if video_recorder is not None:
            video_recorder.stop_recording()

        # Print sliding window delay statistics (only for async mode)
        log_sliding_window_delay_statistics(delay_stats, sliding_window_manager, args_cli)

        return env

        # launch omniverse app
    start_time = time.time()
    print("starting isaacsim")
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app
    print(f"isaacsim started in {time.time() - start_time:.2f}s")
    from lwlab.utils.env import parse_env_cfg

    """Rest everything follows."""
    import gymnasium as gym
    import numpy as np
    if not args_cli.headless:
        from lwlab.core.devices import VRController, VRHand, LwOpenXRDevice, KEYCONTROLLER_MAP
        from isaaclab.devices import Se3Keyboard, Se3SpaceMouse
    if app_launcher_args.get("xr"):
        from isaacsim.xr.openxr import OpenXRSpec
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.envs.ui import ViewportCameraController
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.manipulation.lift import mdp
    from isaaclab.devices.teleop_device_factory import create_teleop_device
    from multiprocessing import Process, shared_memory
    from lwlab.utils.video_recorder import VideoRecorder, get_camera_images
    from lwlab.utils.teleop_utils import save_checkpoint, load_checkpoint, quick_rewind
    from lwlab.utils.place_utils.env_utils import reset_obj_cache, reset_physx, warmup_rendering, set_seed
    import carb

    from omni.log import get_log
    import omni.log

    channels = [
        "omni.physx.tensors.plugin",
        "omni.usd",
        "omni.physicsschema.plugin",
        "omni.physx.plugin",
        "omni.graph.core.plugin",
        "rtx.neuraylib.plugin",
        "rtx.scenedb.plugin",
        "omni.hydra",
        "rtx.scenedb.plugin",
        "omni.isaac.dynamic_control",
        "omni.replicator.core.scripts.extension",
        "omni.usd-abi.plugin",
        "omni.kit.menu.utils.app_menu"
    ]

    # omni.log.set_log_level(omni.log.LogLevel.ERROR)

    for channel in channels:
        get_log().set_channel_enabled(channel, False, omni.log.SettingBehavior.OVERRIDE)
    # get directory path and file name (without extension) from cli arguments
    print(f"teleop_main working directory: {os.getcwd()}")
    print(f"teleop_main dataset_file: {args_cli.dataset_file}")
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # setup default checkpoint path if not provided
    if getattr(args_cli, "checkpoint_path", None) in (None, ""):
        args_cli.checkpoint_path = os.path.join(output_dir, f"{output_file_name}.ckpt.pt")

    env = create_env_config()

    if args_cli.enable_rendering_optimization:
        optimize_rendering(env, args_cli)

    get_default_logger().info(f"env_cfg: {env.cfg}")

    teleop_interface = create_teleop_interface(env)

    from lwlab.utils.ui_utils import (
        setup_cameras,
        setup_task_description_ui,
        spawn_robot_vis_helper_general,
        destroy_robot_vis_helper,
        update_checkers_status,
        update_task_desc,
        hide_ui_windows,
        show_upload_dialog
    )

    # Global variable for batch name GUI
    batch_name_gui = None

    if not args_cli.headless:
        hide_ui_windows(simulation_app)
        # Setup batch name GUI
        # TODO disable batch name GUI for now, since it's not used in teleop
        # batch_name_gui = setup_batch_name_gui(getattr(args_cli, 'batch_name', 'default-batch'))
    print("Starting teleoperation - will check for task signals during execution")

    try:
        with trace_profile("mainloop"):
            env = run_simulation(env, teleop_interface)
    except Exception as e:
        print(f"Error during mainloop execution: {e}")
        import traceback
        traceback.print_exc()
        handle_exception_and_log()

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
