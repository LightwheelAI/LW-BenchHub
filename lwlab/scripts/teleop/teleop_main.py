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

from isaaclab.app import AppLauncher

from lwlab.utils.func import trace_profile
from lwlab import CONFIGS_PATH


def load_yaml_to_namespace(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default=None, help="task config")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
task_config_path = CONFIGS_PATH / 'data_collection' / 'teleop' / f'{args_cli.task_config}.yml'
yaml_args = load_yaml_to_namespace(task_config_path)
args_cli.__dict__.update(yaml_args.__dict__)


app_launcher_args = vars(args_cli)
if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = f'{os.environ["ISAACLAB_PATH"]}/apps/isaaclab.python.xr.openxr.kit'

if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True
    import pinocchio


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
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


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
    from lwlab.utils.devices import LwOpenXRDevice
    from lwlab.utils.devices.keyboard.se3_keyboard import KEYCONTROLLER_MAP
    if args_cli.enable_pinocchio:
        from isaacsim.xr.openxr import OpenXRSpec
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.envs.ui import ViewportCameraController
    from lwlab.utils.video_recorder import VideoRecorder, get_camera_images

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
        from lwlab.utils.env import parse_env_cfg, ExecuteMode
        with trace_profile("parse_env_cfg"):
            env_cfg = parse_env_cfg(
                task_name=args_cli.task,
                robot_name=args_cli.robot,
                scene_name=args_cli.layout,
                robot_scale=args_cli.robot_scale,
                device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric,
                first_person_view=args_cli.first_person_view,
                enable_cameras=app_launcher._enable_cameras,
                execute_mode=ExecuteMode.TELEOP
            )
        env_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"
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

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        device_type = KEYCONTROLLER_MAP[args_cli.teleop_device.lower() + "-" + args_cli.robot.lower().split("-")[0]]
        teleop_interface = device_type(
            pos_sensitivity=0.05 * args_cli.sensitivity, rot_sensitivity=0.05 * args_cli.sensitivity,
            base_sensitivity=0.5 * args_cli.sensitivity, base_yaw_sensitivity=0.8 * args_cli.sensitivity
        )

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
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'handtracking'."
        )

    def run_simulation():
        # add teleoperation key for env reset
        should_reset_recording_instance = False

        def reset_recording_instance():
            nonlocal should_reset_recording_instance
            should_reset_recording_instance = True

        def start_teleoperation():
            nonlocal teleoperation_active
            teleoperation_active = True

        def stop_teleoperation():
            nonlocal teleoperation_active
            teleoperation_active = False

        if isinstance(teleop_interface, LwOpenXRDevice):
            teleoperation_active = False
            teleop_interface.add_callback("RESET", reset_recording_instance)
            teleop_interface.add_callback("START", start_teleoperation)
            teleop_interface.add_callback("STOP", stop_teleoperation)
        else:
            teleoperation_active = True
            teleop_interface.add_callback("R", reset_recording_instance)
        print(teleop_interface)

        rate_limiter = RateLimiter(args_cli.step_hz)

        # reset environment
        env.reset()
        teleop_interface.reset()

        from lwlab.utils.env import setup_cameras, setup_task_description_ui

        if env_cfg.enable_cameras:
            viewports = setup_cameras(env)
            for key, v_p in viewports.items():
                res = v_p.viewport_api.get_texture_resolution()
                sca = v_p.viewport_api.get_texture_resolution_scale()
                print(f"Viewport {key} resolution: {res}, scale: {sca}")

        overlay_window = setup_task_description_ui("Task name: {}\nLayout id: {}\nStyle id: {}\nDesc: {}".format(env_cfg.task_name, env_cfg.layout_id, env_cfg.style_id, env_cfg.get_ep_meta()["lang"]), env)

        current_recorded_demo_count = 0
        success_step_count = 0

        start_record_state = False
        # Initialize video recorder
        video_recorder = None
        if args_cli.save_video:
            video_recorder = VideoRecorder(args_cli.video_save_dir, args_cli.video_fps, args_cli.task, args_cli.robot, args_cli.layout)

        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            # with torch.inference_mode():
            actions = teleop_interface.advance()
            if actions is None or should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False
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
                    # Initialize video recording
                    if video_recorder is not None:
                        camera_data, camera_name = get_camera_images(env)
                        if camera_name is not None:
                            image_shape = (camera_data.shape[0], camera_data.shape[1])  # (height, width)
                            video_recorder.start_recording(camera_name, image_shape)

                env.step(actions)

                # Recorded
                if start_record_state and video_recorder is not None:
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
            if rate_limiter:
                rate_limiter.sleep(env)

        # ensure to stop recording before exiting
        if video_recorder is not None:
            video_recorder.stop_recording()

    with trace_profile("mainloop"):
        run_simulation()

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
