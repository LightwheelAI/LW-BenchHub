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

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
from pathlib import Path
import argparse
import os
import time
import yaml
import json
from datetime import datetime
from lwlab.utils.log_utils import log_scene_rigid_objects, handle_exception_and_log


from lwlab.utils.log_utils import get_default_logger

from isaaclab.app import AppLauncher

import torch

from lwlab.utils.profile_utils import trace_profile, DEBUG_FRAME_ANALYZER, debug_print
from lwlab.utils.config_loader import config_loader

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default=None, help="task config")

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
    # launch omniverse app
    start_time = time.time()
    print("starting isaacsim")
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app
    print(f"isaacsim started in {time.time() - start_time:.2f}s")

    from lwlab.utils.env import parse_env_cfg, ExecuteMode

    """Rest everything follows."""
    import gymnasium as gym
    import numpy as np
    from lwlab.core.devices import VRController, VRHand, LwOpenXRDevice, KEYCONTROLLER_MAP
    from isaaclab.devices import Se3Keyboard, Se3SpaceMouse, Se3SpaceMouseCfg
    if app_launcher_args.get("xr"):
        from isaacsim.xr.openxr import OpenXRSpec
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.envs.ui import ViewportCameraController
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.manipulation.lift import mdp
    from multiprocessing import Process, shared_memory
    from lwlab.utils.video_recorder import VideoRecorder, get_camera_images

    import carb

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if "-" in args_cli.task:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        env_name = args_cli.task
    else:  # isaac-robocasa

        print(f"\n\n\nargs_cli: {args_cli}\n\n\n")
        from lwlab.utils.env import parse_env_cfg, ExecuteMode
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
            replay_cfgs = None
            if len(ep_meta) > 0:
                replay_cfgs = {"ep_meta": ep_meta}
            env_cfg = parse_env_cfg(
                task_name=args_cli.task,
                robot_name=args_cli.robot,
                scene_name=args_cli.layout,
                robot_scale=args_cli.robot_scale,
                device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
                replay_cfgs=replay_cfgs,
                first_person_view=args_cli.first_person_view,
                enable_cameras=app_launcher._enable_cameras,
                execute_mode=ExecuteMode.TELEOP,
                usd_simplify=args_cli.usd_simplify,
            )
        env_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"

        print(f"\n\n\nenv_name: {env_name}\n\n\n")
        
        gym.register(
            id=env_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

    # modify configuration
    env_cfg.env_name = env_name
    env_cfg.terminations.time_out = None
    if args_cli.record:
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
    # create environment
    with trace_profile("gymmake"):
        env: ManagerBasedRLEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    if args_cli.enable_optimization:
        optimize_rendering(env)

    get_default_logger().info(f"env_cfg: {env_cfg}")

    # create controller
    if True: # args_cli.teleop_device.lower() == "keyboard":
        # device_type = KEYCONTROLLER_MAP[args_cli.teleop_device.lower() + "-" + args_cli.robot.lower().split("-")[0]]
        device_type = KEYCONTROLLER_MAP["keyboard-pandaomron"]
        teleop_interface = device_type(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity,
            base_sensitivity=0.5 * args_cli.sensitivity, base_yaw_sensitivity=0.8 * args_cli.sensitivity
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
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
    elif args_cli.teleop_device.lower() == "handtracking":
        from isaacsim.xr.openxr import OpenXRSpec

        teleop_interface = Se3HandTracking(OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True)
        teleop_interface.add_callback("RESET", env.reset)
        viewer = ViewerCfg(eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer")
        ViewportCameraController(env, viewer)

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

    def run_simulation():
        # add teleoperation key for env reset
        should_reset_recording_instance = False
        flush_recorder_manager_flag = False

        def reset_recording_instance():
            nonlocal should_reset_recording_instance
            should_reset_recording_instance = True

        def start_teleoperation():
            nonlocal teleoperation_active
            teleoperation_active = True

        def stop_teleoperation():
            nonlocal teleoperation_active
            teleoperation_active = False

        def flush_recorder_manager():
            nonlocal flush_recorder_manager_flag
            flush_recorder_manager_flag = True

        if isinstance(teleop_interface, LwOpenXRDevice):
            teleoperation_active = False
            teleop_interface.add_callback("RESET", reset_recording_instance)
            teleop_interface.add_callback("START", start_teleoperation)
            teleop_interface.add_callback("STOP", stop_teleoperation)
        else:
            teleoperation_active = True
            teleop_interface.add_callback("R", reset_recording_instance)
            teleop_interface.add_callback("T", flush_recorder_manager)
        print(teleop_interface)

        rate_limiter = RateLimiter(args_cli.step_hz)

        # reset environment
        env.reset()
        teleop_interface.reset()

        from lwlab.utils.env import setup_cameras, setup_task_description_ui, spawn_robot_vis_helper_general, destroy_robot_vis_helper

        if env_cfg.enable_cameras and args_cli.enable_multiple_viewports:
            viewports = setup_cameras(env)
            for key, v_p in viewports.items():
                res = v_p.viewport_api.get_texture_resolution()
                sca = v_p.viewport_api.get_texture_resolution_scale()
                print(f"Viewport {key} resolution: {res}, scale: {sca}")

        overlay_window = setup_task_description_ui(env_cfg, env)

        current_recorded_demo_count = 0
        success_step_count = 0
        frame_count = 0
        start_record_state = False
        # Initialize video recorder
        video_recorder = None
        if args_cli.save_video:
            video_recorder = VideoRecorder(args_cli.video_save_dir, args_cli.video_fps, args_cli.task, args_cli.robot, args_cli.layout)

        if args_cli.enable_log:
            log_path = log_scene_rigid_objects(env)

        # add frame rate analyzer (only in debug mode)
        frame_analyzer = DEBUG_FRAME_ANALYZER
        debug_print("Frame rate analyzer initialized in debug mode")

        vis_helper_prims = []

        # simulate environment
        while simulation_app.is_running():
            if flush_recorder_manager_flag:
                env.recorder_manager.export_episodes()
                env.recorder_manager.reset()
                flush_recorder_manager_flag = False

            # start frame analysis (debug mode only)
            frame_analyzer.start_frame()

            # run everything in inference mode
            # with torch.inference_mode():
            teleop_start = time.time()
            actions = teleop_interface.advance()
            
            # print(actions.shape)

            """
            left/right arm pos/quat (wzyx) (holding arms out in front of the body):

            left_wrist_pos: [0.20248358 0.17661604 0.13682538]
            left_wrist_quat: [ 0.99800018  0.04580273 -0.03275206  0.02872376]

            right_wrist_pos: [ 0.21232468 -0.18499057  0.13347319]
            right_wrist_quat: [ 0.99572691 -0.05724299 -0.05909783 -0.04193568]
            """

            """
            [ left hand, right hand,
              left arm pos,
              left arm quat,
              right arm pos,
              right arm quat ]
            """
            actions_upper = torch.tensor(
                [1, 1,
                 0.20248358, 0.17661604, 0.13682538, 
                 0.99800018,  0.04580273, -0.03275206,  0.02872376,
                 0.21232468, -0.18499057,  0.13347319,
                 0.99572691, -0.05724299, -0.05909783, -0.04193568]
            ).unsqueeze(0)

            # Create a [:2] tensor where the values cycle between -1 to 1 continuously for the hand actions
            t = time.time()
            cycle = torch.tensor([np.sin(t), np.cos(t)])
            actions_upper[:, :2] = cycle
            # Add some noise to the wrist poses
            noise_std = 0.05
            noise = noise_std * torch.randn_like(actions_upper)
            noise[:, 0:2] = 0
            actions_upper = actions_upper + noise

            actions_lower = torch.zeros(env.num_envs, 4)
            actions = torch.cat([actions_upper, actions_lower], dim=1)
            actions[:, -1] = 0.7

            teleop_time = time.time() - teleop_start
            frame_analyzer.record_stage('teleop_advance', teleop_time)
            if actions is None or should_reset_recording_instance:
                if args_cli.enable_log:
                    try:
                        destroy_robot_vis_helper(vis_helper_prims, env)
                        env.reset()
                    except Exception as e:
                        handle_exception_and_log(e, log_path)
                        break
                else:
                    destroy_robot_vis_helper(vis_helper_prims, env)
                    env.reset()
                should_reset_recording_instance = False
                frame_count = 0
                if start_record_state == True:
                    print("Stop Recording!!!")
                    start_record_state = False
                    if video_recorder is not None:
                        video_recorder.stop_recording()

            elif (isinstance(actions, bool) and actions == False) or (not teleoperation_active):
                env.render()
            # apply actions
            else:
                if start_record_state == False:
                    print("Start Recording!!!")
                    start_record_state = True

                    vis_helper_prims = spawn_robot_vis_helper_general(env)
                    # Initialize video recording
                    if video_recorder is not None:
                        camera_data, camera_name = get_camera_images(env)
                        if camera_name is not None:
                            image_shape = (camera_data.shape[0], camera_data.shape[1])  # (height, width)
                            video_recorder.start_recording(camera_name, image_shape)

                # # warmup rendering
                # if env.common_step_counter <= 1:
                #     for _ in range(env.cfg.warmup_steps):
                #         env.sim.step()
                #         env.scene.update(env.physics_dt)

                if args_cli.enable_log:
                    try:
                        env.step(actions)
                    except Exception as e:
                        handle_exception_and_log(e, log_path)
                        break
                else:
                    frame_count += 1  # increase frame counter
                    # measure env_step time
                    step_start = time.time()
                    with trace_profile(f"env_step_frame_{frame_count:06d}"):
                        carb.profiler.begin(1, "evn_step")
                        obs, *_ = env.step(actions)
                        carb.profiler.end(1)
                    step_time = time.time() - step_start
                    frame_analyzer.record_stage('env_step', step_time)

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
                    print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                    break

            if teleoperation_active:
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

    try:
        with trace_profile("mainloop"):
            run_simulation()
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
