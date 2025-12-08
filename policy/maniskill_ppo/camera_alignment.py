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

import argparse
import math

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher

from lw_benchhub.sim2real.lerobot_follower.so100_follower import SO100Follower
from lw_benchhub.sim2real.lerobot_follower.so101_follower import SO101Follower
from lw_benchhub.utils.config_loader import config_loader
from lw_benchhub.utils.place_utils.env_utils import set_seed

EPS = np.finfo(float).eps * 4.0

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default="default", help="task config")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)

app_launcher_args = vars(args_cli)
# launch omniverse app
# os.environ["ROBOCASA_ASSETS_ROOT"] = str(args_cli.asset_base_path)
app_launcher = AppLauncher(app_launcher_args)

simulation_app = app_launcher.app

args_cli.device = f"cuda:0"


def find_available_camera(max_cams=5):
    for cam_index in range(max_cams):
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            print(f"Using camera index: {cam_index}")
            return cap
        cap.release()
    print("No available camera found.")
    return None


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def set_camera_follow_pose(env, pos_offset, rot_offset_wxyz):
    # 1) Get the world pose of the robot base
    robot_q_wxyz = env.scene.articulations["robot"].data.body_quat_w[..., 2, :][0].cpu().numpy()  # (w,x,y,z)
    robot_pos = env.scene.articulations["robot"].data.body_pos_w[..., 2, :][0].cpu().numpy()      # (3,)

    # 2) Position offset -> world frame
    robot_q_xyzw = robot_q_wxyz[[1, 2, 3, 0]]
    robot_R = quat2mat(robot_q_xyzw)  # 3x3
    pos_offset = np.asarray(pos_offset, dtype=np.float32).reshape(3)
    cam_pos_world = robot_pos + (robot_R @ pos_offset)
    # 3) Orientation offset -> world frame: cam_q = robot_q ⊗ rot_offset
    # Use scipy.Rotation to combine quaternions (its input format is xyzw)
    rot_offset_wxyz = np.asarray(rot_offset_wxyz, dtype=np.float32).reshape(4)
    r_robot = R.from_quat(robot_q_xyzw)                                 # xyzw
    r_off = R.from_quat(rot_offset_wxyz[[1, 2, 3, 0]])                # wxyz -> xyzw
    r_cam = r_robot * r_off
    cam_q_xyzw = r_cam.as_quat()                                        # xyzw
    cam_q_wxyz = cam_q_xyzw[[3, 0, 1, 2]]                               # back wxyz

    global_cam = env.scene.sensors["global_camera"]
    positions = torch.tensor([cam_pos_world], device=env.device, dtype=torch.float32)
    orientations = torch.tensor([cam_q_wxyz], device=env.device, dtype=torch.float32)
    positions[0][2] += 0.03047
    print("positions: ", pos_offset)
    print("orientations:", rot_offset_wxyz)
    global_cam.set_world_poses(positions=positions,
                               orientations=orientations,
                               env_ids=[0],
                               convention="opengl")
    # global_cam.set_world_poses_from_view(robot_pos_tensor, robot_lookat_tensor)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    import gymnasium as gym
    import torch
    import numpy as np
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab.envs.ui import ViewportCameraController
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab_tasks.manager_based.manipulation.lift import mdp
    from isaaclab_tasks.utils import parse_env_cfg
    from multiprocessing import Process, shared_memory

    if "-" in args_cli.task:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task

    else:  # robocasa
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        env_cfg = parse_env_cfg(
            scene_backend=args_cli.scene_backend,
            task_backend=args_cli.task_backend,
            task_name=args_cli.task,
            robot_name=args_cli.robot,
            scene_name=args_cli.layout,
            rl_name=args_cli.rl,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.TRAIN,
            usd_simplify=args_cli.usd_simplify,
            seed=args_cli.seed,
            sources=args_cli.sources,
            object_projects=args_cli.object_projects,
            headless_mode=args_cli.headless,
        )
        task_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",  # lw_benchhub.enhance.envs:ManagerBasedRLDigitalTwinEnv
            kwargs={
            },
            disable_env_checker=True,
        )

    # modify configuration
    env_cfg.terminations.time_out = None
    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg)  # .unwrapped
    set_seed(env_cfg.seed, env.unwrapped)
    from policy.maniskill_ppo.agent import PPOArgs, PPO, observation

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # reset environment before any step calls
    env.reset()

    # don't connect camera
    if args_cli.uid == "so101":
        follower = SO101Follower(port="/dev/ttyACM0", calibration_file_name="so101_follower.json", use_degrees=True)
        reset_qpos = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.87])
    elif args_cli.uid == "so100":
        follower = SO100Follower(port="/dev/ttyACM0", calibration_file_name="so100_follower.json", use_degrees=True)
        reset_qpos = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])
    follower.reset(reset_qpos)
    # ---------------------------------------------------------------
    # Optional: Camera alignment before training (keyboard controlled)
    # Controls: w/s/a/d (x/y), up/down (z), left/right (FOV), backspace (reset)
    # Close the figure window to finish alignment and start training
    # ---------------------------------------------------------------

    import matplotlib.pyplot as plt
    import numpy as np
    import time as _time
    import cv2
    import os
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.envs.mdp.observations import image

    # base camera offsets from config if present
    base_pos_offset = None
    base_rot_offset = None

    if hasattr(env.env.isaaclab_arena_env.embodiment.cfg.observation_cameras['global_camera']['camera_cfg'].offset, "pos"):
        base_pos_offset = np.array(env.env.isaaclab_arena_env.embodiment.cfg.observation_cameras['global_camera']['camera_cfg'].offset.pos, dtype=np.float32)
        base_rot_offset = np.array(env.env.isaaclab_arena_env.embodiment.cfg.observation_cameras['global_camera']['camera_cfg'].offset.rot, dtype=np.float32)

    camera_offset = np.zeros(3, dtype=np.float32)
    active_keys = set()
    MOVEMENT_SPEED = 0.1
    FOV_CHANGE_SPEED = 0.1
    ROT_SPEED = 0.5  # radians per second
    # current rotation offset (wxyz), initialize from base_rot_offset or identity
    # initialize camera to base pose
    # set_camera_follow_pose(env.env, base_offset, base_lookat)

    def _on_key_press(event):
        print("Pressed:", event.key)
        if event.key is not None:
            active_keys.add(event.key)

    def _on_key_release(event):
        print("Released:", event.key)
        if event.key is not None and event.key in active_keys:
            active_keys.discard(event.key)

    def _update_camera(last_frame_time, fov_offset, cur_rot_offset):
        now = _time.time()
        dt = now - last_frame_time
        last_frame_time = now

        # movement
        if "w" in active_keys:
            camera_offset[0] -= MOVEMENT_SPEED * dt
        if "s" in active_keys:
            camera_offset[0] += MOVEMENT_SPEED * dt
        if "a" in active_keys:
            camera_offset[1] -= MOVEMENT_SPEED * dt
        if "d" in active_keys:
            camera_offset[1] += MOVEMENT_SPEED * dt
        if "up" in active_keys:
            camera_offset[2] += MOVEMENT_SPEED * dt
        if "down" in active_keys:
            camera_offset[2] -= MOVEMENT_SPEED * dt

        # fov
        if "left" in active_keys:
            fov_offset -= FOV_CHANGE_SPEED * dt
        if "right" in active_keys:
            fov_offset += FOV_CHANGE_SPEED * dt

        # rotation (quaternion), keys: i/k (pitch ±X), o/p (roll ±Y), j/l (yaw ±Z)
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        if "i" in active_keys:
            pitch += ROT_SPEED * dt
        if "j" in active_keys:
            pitch -= ROT_SPEED * dt
        if "o" in active_keys:
            roll += ROT_SPEED * dt
        if "k" in active_keys:
            roll -= ROT_SPEED * dt
        if "p" in active_keys:
            yaw += ROT_SPEED * dt
        if "l" in active_keys:
            yaw -= ROT_SPEED * dt

        if pitch != 0.0 or roll != 0.0 or yaw != 0.0:
            # compose delta rotation in local camera frame
            r_cur = R.from_quat(cur_rot_offset[[1, 2, 3, 0]])  # wxyz -> xyzw
            r_delta = R.from_euler('xyz', [pitch, roll, yaw], degrees=False)
            r_new = r_delta * r_cur
            q_xyzw = r_new.as_quat()
            cur_rot_offset = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)  # back to wxyz

        # reset
        if "backspace" in active_keys:
            camera_offset[:] = 0.0
            fov_offset = 0.0
            cur_rot_offset = base_rot_offset.astype(np.float32).copy() if base_rot_offset is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        update_offset = base_pos_offset + camera_offset
        set_camera_follow_pose(env.env, update_offset, cur_rot_offset)
        zero_action = torch.zeros(env.env.action_space.shape, device=env.env.device)
        obs, _, _, _, _ = env.step(zero_action)
        # env.env.sim.reset()

    # setup real image source (camera or image file)
    cap = None
    if os.environ.get("REAL_OVERLAY_CAM") is not None:
        try:
            cam_index = int(os.environ.get("REAL_OVERLAY_CAM"))
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                print(f"Camera {cam_index} not available, searching automatically...")
                cap = find_available_camera()
        except Exception:
            cap = find_available_camera()
    else:
        cap = find_available_camera()
    if cap is None:
        print("No camera detected, please check your device or provide an image path.")

    # helper to get current overlay frame (blend real and sim)
    def _get_overlay_frame():
        # get sim image via observation API (normalized [0,1])
        try:
            sim_tensor = env.env.scene.sensors["global_camera"].data.output["rgb"]
            # print(env.env.scene.sensors["global_camera"].data.pos_w)
        except Exception as e:
            print(f"Error getting sim image: {e}")
            return None, None, None

        # expect shape [num_env, H, W, C]; take first env
        sim_np = sim_tensor[0].detach().cpu().numpy()

        # resize sim image to 224x224
        target_w, target_h = 224, 224
        sim_rgb = cv2.resize(sim_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        real_rgb = None
        if cap is not None and cap.isOpened():
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                real_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # resize real image to 224x224
        h, w, c = real_rgb.shape
        start_w = (w - h) // 2
        real_rgb = real_rgb[:, start_w:start_w + h, :]
        real_rgb = cv2.resize(real_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        sim_f = sim_rgb.astype(np.float32)
        sim_f = sim_f / 255.0
        real_f = real_rgb.astype(np.float32)
        real_f = real_f / 255.0

        # 2) 融合 (可调权重)
        alpha = 0.5
        overlay_f = alpha * sim_f + (1.0 - alpha) * real_f
        # 3) 转回 uint8
        overlay = (overlay_f * 255.0).clip(0, 255).astype(np.uint8)

        return overlay

    # simple viewer using matplotlib to capture key events
    fig = plt.figure()
    ax = fig.add_subplot()
    # turn off default key bindings to avoid conflicts
    try:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.manager.key_press_handler_id = None
    except Exception:
        pass
    ax.text(0.02, 0.98,
            "Align camera → WASD/Arrows; Left/Right=FOV; Backspace=Reset\nClose window to start training",
            transform=ax.transAxes, va='top', ha='left')
    ax.axis('off')

    # initial image content
    init_frame = _get_overlay_frame()
    if init_frame is None:
        init_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    im = ax.imshow(init_frame)
    fig.canvas.mpl_connect("key_press_event", _on_key_press)
    fig.canvas.mpl_connect("key_release_event", _on_key_release)

    # loop until window closed
    while plt.fignum_exists(fig.number):

        last_frame_time, fov_offset, cur_rot_offset = _time.time(), 0.0, base_rot_offset.astype(np.float32).copy() if base_rot_offset is not None else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        _update_camera(last_frame_time, fov_offset, cur_rot_offset)
        frame = _get_overlay_frame()
        if frame is not None:
            im.set_data(frame)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    try:
        plt.close(fig)
    except Exception:
        pass
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    # close the simulator
    # so101_follower.disconnect()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    main()
