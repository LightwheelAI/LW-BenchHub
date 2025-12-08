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

import argparse
import copy
import json
import multiprocessing
import os
import select
import sys
import time
from collections.abc import Callable

from termcolor import colored

from isaaclab.app import AppLauncher

from lw_benchhub.utils.config_loader import config_loader
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.log_utils import get_default_logger, handle_exception_and_log, log_scene_rigid_objects
from lw_benchhub.utils.profile_utils import DEBUG_FRAME_ANALYZER, debug_print, trace_profile

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


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


def optimize_rendering(env):
    import carb
    settings = carb.settings.get_settings()
    # enable async rendering
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)
    settings.set_bool("/app/asyncRendering", False)
    settings.set_bool("/app/asyncRenderingLowLatency", False)
    settings.set_bool("/app/asyncRendering", True)
    settings.set_bool("/app/asyncRenderingLowLatency", True)

    # use the USD / Fabric only for poses
    if args_cli.use_fabric:
        settings.set_bool("/physics/updateToUsd", False)
        settings.set_bool("/physics/updateParticlesToUsd", True)
        settings.set_bool("/physics/updateVelocitiesToUsd", False)
        settings.set_bool("/physics/updateForceSensorsToUsd", False)
        settings.set_bool("/physics/updateResidualsToUsd", False)
        settings.set_bool("/physics/outputVelocitiesLocalSpace", False)
        settings.set_bool("/physics/fabricUpdateTransformations", True)
        settings.set_bool("/physics/fabricUpdateVelocities", False)
        settings.set_bool("/physics/fabricUpdateForceSensors", False)
        settings.set_bool("/physics/fabricUpdateJointStates", False)
        settings.set_bool("/physics/fabricUpdateResiduals", False)
        settings.set_bool("/physics/fabricUseGPUInterop", True)

    # enable DLSS and performance optimization
    settings.set_bool("/rtx-transient/dlssg/enabled", True)
    settings.set_int("/rtx/post/dlss/execMode", 0)  # "Performance"
    settings.set_bool("/rtx/raytracing/fractionalCutoutOpacity", False)

    # TODO this option affects the rendering of dynamic spawned objects
    settings.set_bool("/app/renderer/skipMaterialLoading", False)
    settings.set_bool("/app/renderer/skipTextureLoading", False)

    # Setup timeline
    import omni.timeline as timeline
    timeline = timeline.get_timeline_interface()
    # Configure Kit to not wait for wall clock time to catch up between updates
    # This setting is effective only with Fixed time stepping
    timeline.set_play_every_frame(True)

    # enable fast mode and ensure fixed time stepping
    settings.set_bool("/app/player/useFastMode", True)
    settings.set_bool("/app/player/useFixedTimeStepping", True)

    # configure all run loops, disable rate limiting
    for run_loop in ["present", "main", "rendering_0"]:
        settings.set_bool(f"/app/runLoops/{run_loop}/rateLimitEnabled", False)
        settings.set_int(f"/app/runLoops/{run_loop}/rateLimitFrequency", 120)
        settings.set_bool(f"/app/runLoops/{run_loop}/rateLimitUseBusyLoop", False)

    # disable vertical sync to improve frame rate
    settings.set_bool("/app/vsync", False)
    settings.set_bool("/exts/omni.kit.renderer.core/present/enabled", False)

    # enable gpu dynamics
    if args_cli.device != "cpu":
        physics_context = env.sim.get_physics_context()
        physics_context.enable_gpu_dynamics(True)
        physics_context.set_broadphase_type("GPU")

    # settings.set_int("/persistent/physics/numThreads", 0)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    teleoperation_active = None
    should_reset_recording_instance = None
    should_reset_env_instance = None
    should_reset_env_keep_placement = None
    flush_recorder_manager_flag = None
    rollback_to_checkpoint_flag = None

    def reset_recording_instance():
        print("reset recording instance", flush=True)
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def reset_env_instance_no_keep_placement():
        print("reset env instance", flush=True)
        nonlocal should_reset_env_instance
        should_reset_env_instance = True
        nonlocal should_reset_env_keep_placement
        should_reset_env_keep_placement = False

    def reset_env_instance_keep_placement():
        print("reset env instance keep placement", flush=True)
        nonlocal should_reset_env_instance
        should_reset_env_instance = True
        nonlocal should_reset_env_keep_placement
        should_reset_env_keep_placement = True

    def start_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        nonlocal teleoperation_active
        teleoperation_active = False

    def flush_recorder_manager():
        nonlocal flush_recorder_manager_flag
        flush_recorder_manager_flag = True

    def call_save_checkpoint():
        frame_index = save_checkpoint(env, args_cli.checkpoint_path)
        if isinstance(teleop_interface, VRController):
            teleop_interface.set_checkpoint_frame_idx(frame_index)

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

    def create_teleop_interface(env):
        """Create teleoperation interface based on device type."""
        if args_cli.headless:
            return None

        env_cfg = env.cfg
        nonlocal teleoperation_active

        teleoperation_callbacks: dict[str, Callable[[], None]] = {
            "X": reset_recording_instance,
            "Y": reset_env_instance_no_keep_placement,
            "R": reset_env_instance_keep_placement,
            "T": flush_recorder_manager,
            "M": call_save_checkpoint,
            "N": rollback_to_checkpoint,
            # Add new shortcut: quick rewind 10 frames
            "B": lambda: quick_rewind(env, 10),
        }

        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleoperation_active = True
            teleop_interface = create_teleop_device(
                env, args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            if args_cli.teleop_device.lower() == "keyboard":
                device_type = KEYCONTROLLER_MAP[args_cli.teleop_device.lower() + "-" + args_cli.robot.lower().split("-")[0]]
                teleop_interface = device_type(
                    pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity,
                    base_sensitivity=0.5 * args_cli.sensitivity, base_yaw_sensitivity=0.8 * args_cli.sensitivity
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(env,
                                                 pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.2 * args_cli.sensitivity
                                                 )
            # elif args_cli.teleop_device.lower() == "gamepad":
            #     teleop_interface = Se3Gamepad(
            #         pos_sensitivity=0.1 * args_cli.sensitivity, rot_sensitivity=0.1 * args_cli.sensitivity
            #     )
            elif args_cli.teleop_device.lower() == "so101leader":
                from lw_benchhub.core.devices.lerobot import SO101Leader
                teleop_interface = SO101Leader(env, port=args_cli.port, recalibrate=args_cli.recalibrate)
            elif args_cli.teleop_device.lower() == "bi_so101leader":
                from lw_benchhub.core.devices.lerobot import BiSO101Leader
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
                                                  shm_name=shm.name,)
            else:
                raise ValueError(
                    f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse''handtracking'."
                )

            # Add callbacks to teleop device
            if not args_cli.headless and isinstance(teleop_interface, LwOpenXRDevice):
                teleoperation_active = False
                teleoperation_callbacks: dict[str, Callable[[], None]] = {
                    "RESET": reset_recording_instance,
                    "START": start_teleoperation,
                    "STOP": stop_teleoperation,
                    "SAVE": lambda: save_checkpoint(env, args_cli.checkpoint_path),
                    "LOAD": lambda: load_checkpoint(env, args_cli.checkpoint_path),
                    # Add new shortcut: quick rewind 10 frames
                    "REWIND": lambda: quick_rewind(env, 10),
                }
            elif teleop_interface is not None:
                teleoperation_active = True
                for key, callback in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, callback)
                    except (ValueError, TypeError) as e:
                        omni.log.warn(f"Failed to add callback for key {key}: {e}")
            else:
                teleoperation_active = False

        print(teleop_interface)

        return teleop_interface

    def create_env_config(initial_state=None, object_cfgs=None, cache_usd_version=None, scene_name=None):
        """Create environment configuration based on task type."""
        import omni.usd
        omni.usd.get_context().new_stage()
        if "-" in args_cli.task:
            env_cfg = parse_env_cfg(
                args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
            )
            env_name = args_cli.task
        else:  # isaac-robocasa
            with trace_profile("parse_env_cfg"):
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
                env_cfg = parse_env_cfg(
                    task_name=args_cli.task,
                    robot_name=args_cli.robot,
                    scene_name=scene_name,
                    robot_scale=args_cli.robot_scale,
                    device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
                    replay_cfgs=replay_cfgs,
                    first_person_view=args_cli.first_person_view,
                    enable_cameras=app_launcher._enable_cameras,
                    execute_mode=ExecuteMode.TELEOP,
                    usd_simplify=args_cli.usd_simplify,
                    object_init_offset=object_init_offset,
                    max_scene_retry=args_cli.max_scene_retry,
                    max_object_placement_retry=args_cli.max_object_placement_retry,
                    seed=args_cli.seed,
                    sources=args_cli.sources,
                    object_projects=args_cli.object_projects,
                    initial_state=initial_state,
                    headless_mode=args_cli.headless,
                    resample_objects_placement_on_reset=args_cli.resample_objects_placement_on_reset,
                    resample_robot_placement_on_reset=args_cli.resample_robot_placement_on_reset,
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

    def reset_env(env, teleop_interface, viewports, keep_placement=True, initial_state=None):
        """Reset environment by creating a new one with fresh configuration."""
        initial_state = None
        cache_usd_version = None
        object_cfgs = None
        scene_name = None
        if keep_placement:
            # if initial_state is None:
            #     initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data["initial_state"])
            if "robocasalibero" in env.cfg.isaaclab_arena_env.scene.usd_path:
                scene_name = "robocasalibero"
            else:
                scene_name = "robocasakitchen"
            scene_name = f"{scene_name}-{env.cfg.isaaclab_arena_env.scene.layout_id}-{env.cfg.isaaclab_arena_env.scene.style_id}"
            object_cfgs = copy.deepcopy(env.cfg.isaaclab_arena_env.task.object_cfgs)
            cache_usd_version = {
                "floorplan_version": env.cfg.isaaclab_arena_env.scene.floorplan_version,
                "objects_version": env.cfg.isaaclab_arena_env.task.objects_version
            }
            cache_usd_version["keep_placement"] = True
            from lw_benchhub.utils.robocasa_utils import convert_fixture_to_name
            for obj in object_cfgs:
                obj["placement"] = convert_fixture_to_name(obj["placement"])
        else:
            initial_state = None
            # Close current environment
        env.close()
        print(f"reset env, initial_state: {initial_state}, object_cfgs: {object_cfgs}, scene_name: {scene_name}")
        # Create fresh configuration using the same logic
        new_env = create_env_config(initial_state=initial_state,
                                    cache_usd_version=cache_usd_version,
                                    object_cfgs=object_cfgs,
                                    scene_name=scene_name)

        get_default_logger().info(f"env_cfg: {new_env.cfg}")

        new_env.reset(seed=new_env.cfg.seed)

        if args_cli.enable_rendering_optimization:
            optimize_rendering(new_env)
        if teleop_interface is not None:
            teleop_interface.refresh_env(new_env)
            teleop_interface.reset()
        if not args_cli.headless:
            viewports, overlay_window = setup_env_config_with_args(new_env, viewports)
        else:
            viewports, overlay_window = None, None
        return new_env, teleop_interface, viewports, overlay_window

    def setup_env_config_with_args(env, viewports=None):
        env_cfg = env.cfg
        if not args_cli.headless and env_cfg.enable_cameras and args_cli.enable_multiple_viewports:
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
        nonlocal should_reset_recording_instance
        nonlocal flush_recorder_manager_flag
        nonlocal rollback_to_checkpoint_flag
        nonlocal should_reset_env_instance
        nonlocal should_reset_env_keep_placement
        should_reset_recording_instance = False
        flush_recorder_manager_flag = False
        rollback_to_checkpoint_flag = False
        should_reset_env_instance = False
        should_reset_env_keep_placement = True  # Default to keep placement
        ci_start_flag = None

        rate_limiter = RateLimiter(args_cli.step_hz)

        # reset environment
        env.reset()
        if teleop_interface is not None:
            teleop_interface.reset()
        # auto load checkpoint if enabled
        if getattr(args_cli, "auto_load_checkpoint", False) and os.path.exists(args_cli.checkpoint_path):
            load_checkpoint()

        viewports = None
        overlay_window = None
        if not args_cli.headless:
            viewports, overlay_window = setup_env_config_with_args(env)

        print(colored(env_cfg.isaaclab_arena_env.orchestrator.get_ep_meta()["lang"], "green"))

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
        action_idx = 0

        # simulate environment
        initial_state = None
        while simulation_app.is_running():
            if _check_no_task_signal():
                print(colored("No task available signal received, stopping teleoperation", "yellow"))
                break

            if args_cli.headless and teleop_interface is None:  # for CI test
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    command = sys.stdin.readline().strip()
                    print(f"get command is {command}", flush=True)
                    if command.lower() == "r":
                        reset_env_instance_keep_placement()
                        action_idx = 0

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
                import torch
                # actions = torch.zeros(env.action_space.shape)
                if ci_start_flag:
                    with open("configs/data_collection/teleop/g1wbc_action_data/actions.txt", "r") as f:
                        actions_history = [torch.tensor(eval(line.strip())) for line in f.readlines()]
                    if action_idx < len(actions_history):
                        actions = actions_history[action_idx]
                        action_idx += 1
                    else:
                        break
                else:
                    actions = None
            else:
                actions = teleop_interface.advance()

            teleop_time = time.time() - teleop_start
            frame_analyzer.record_stage('teleop_advance', teleop_time)
            if actions is None or should_reset_recording_instance or should_reset_env_instance:
                destroy_robot_vis_helper(vis_helper_prims, env)
                if should_reset_env_instance:
                    env, teleop_interface, viewports, overlay_window = reset_env(env, teleop_interface, viewports=viewports,
                                                                                 initial_state=initial_state,
                                                                                 keep_placement=should_reset_env_keep_placement)
                    env_cfg = env.cfg
                    should_reset_env_instance = False
                    should_reset_env_keep_placement = True
                    action_idx = 0
                    ci_start_flag = None
                elif should_reset_recording_instance:
                    env.reset()
                    should_reset_recording_instance = False

                if not args_cli.headless:
                    update_task_desc(env, env_cfg)

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

                # warmup rendering
                if not args_cli.headless and env.common_step_counter <= 1:
                    for _ in range(env.cfg.warmup_steps):
                        update_sensors(env, env.physics_dt)

                if args_cli.enable_debug_log:
                    try:
                        env.step(actions)
                        if initial_state is None:
                            initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data["initial_state"])
                        update_checkers_status(env, env_cfg.isaaclab_arena_env.task.get_warning_text())
                    except Exception as e:
                        print(f"Error during env step: {traceback.format_exc()}")
                        handle_exception_and_log(e, log_path)
                        break

                else:
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
                        carb.profiler.end(1)
                    if initial_state is None:
                        initial_state = copy.deepcopy(env.recorder_manager.get_episode(0).data["initial_state"])
                    step_time = time.time() - step_start
                    frame_analyzer.record_stage('env_step', step_time)

                    update_checkers_status(env, env_cfg.isaaclab_arena_env.task.get_warning_text())

                # Recorded
                if env_cfg.enable_cameras and start_record_state and video_recorder is not None:
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

                    if not args_cli.continue_teleop_after_success:
                        break

                    # Rename hdf5 file to indicate successful completion
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

                    # Reset environment with keep_placement=False for new session
                    reset_env_instance_no_keep_placement()

            if not args_cli.headless and teleoperation_active:
                env.sim.render()

            # only use rate_limiter when needed, don't let it limit rendering
            if rate_limiter and teleoperation_active:
                rate_start = time.time()
                with trace_profile("rate_limiter"):
                    rate_limiter.sleep(env)
                rate_time = time.time() - rate_start
                frame_analyzer.record_stage('rate_limiter', rate_time)

            # end frame analysis (debug mode only)
            frame_analyzer.end_frame()

        # ensure to stop recording before exiting
        if video_recorder is not None:
            video_recorder.stop_recording()

        return env

        # launch omniverse app
    start_time = time.time()
    print("starting isaacsim")
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app
    print(f"isaacsim started in {time.time() - start_time:.2f}s")
    from lw_benchhub.utils.env import parse_env_cfg

    """Rest everything follows."""
    import gymnasium as gym
    import numpy as np
    if not args_cli.headless:
        from lw_benchhub.core.devices import VRController, VRHand, LwOpenXRDevice, KEYCONTROLLER_MAP
        from isaaclab.devices import Se3Keyboard, Se3SpaceMouse
    if app_launcher_args.get("xr"):
        from isaacsim.xr.openxr import OpenXRSpec
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.envs.ui import ViewportCameraController
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.manipulation.lift import mdp
    from isaaclab.devices.teleop_device_factory import create_teleop_device
    from multiprocessing import Process, shared_memory
    from lw_benchhub.utils.video_recorder import VideoRecorder, get_camera_images
    from lw_benchhub.utils.teleop_utils import save_checkpoint, load_checkpoint, quick_rewind
    from lw_benchhub.utils.place_utils.env_utils import set_seed
    from lw_benchhub.utils.isaaclab_utils import update_sensors
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
        optimize_rendering(env)

    get_default_logger().info(f"env_cfg: {env.cfg}")

    teleop_interface = create_teleop_interface(env)

    from lw_benchhub.utils.ui_utils import (
        setup_cameras,
        setup_task_description_ui,
        spawn_robot_vis_helper_general,
        destroy_robot_vis_helper,
        update_checkers_status,
        update_task_desc,
        hide_ui_windows,
        setup_batch_name_gui
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

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
