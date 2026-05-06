"""Sweep EE target Z (oven frame) and test IK reachability.

Keeps X, Y, and orientation fixed at the current g1_loco_left config values.
Teleports the robot to --robot_dist metres from the object (matching sampling_radius),
squats, then sweeps Z.

Usage:
    python z_target_sweep.py \
        --pipeline_id LWBenchhub-Autosim-G1CloseOvenPipeline-v0 \
        --headless \
        --robot_dist 1.15 \
        --squat_steps 40 \
        --z_min -0.5 --z_max 0.2 --z_step 0.05
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline_id",  required=True, type=str)
parser.add_argument("--robot_dist",   default=1.15,  type=float,
                    help="Distance to teleport robot from oven (should match sampling_radius)")
parser.add_argument("--x_offset",     default=-0.3,  type=float,
                    help="Robot base X offset relative to oven X (matches init_state_pos_delta[0])")
parser.add_argument("--squat_steps",  default=40,    type=int)
parser.add_argument("--z_min",        default=-0.5,  type=float)
parser.add_argument("--z_max",        default=0.2,   type=float)
parser.add_argument("--z_step",       default=0.05,  type=float)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

import lw_benchhub.autosim  # noqa: F401
from autosim import make_pipeline


def _teleport_robot(env, pos_w: torch.Tensor, yaw_rad: float, settle_steps: int = 20):
    half = yaw_rad / 2.0
    quat = torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=torch.float32, device=env.device,
    )
    pose = torch.zeros(1, 7, device=env.device)
    pose[0, :3] = pos_w.to(env.device)
    pose[0, 3:]  = quat
    robot = env.scene["robot"]
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(settle_steps):
        env.sim.step()
    env.scene.update(env.sim.get_physics_dt())


def _settle_squat(env, squat_steps: int) -> None:
    if squat_steps <= 0:
        return
    action = torch.zeros(env.action_space.shape, device=env.device)
    action[0, 0] = -1.0
    action[0, 3] = 1.0
    env.action_manager.process_action(action)
    for _ in range(squat_steps):
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.cfg.motion_planner.use_cuda_graph = False
    pipeline.initialize()

    env        = pipeline._env
    extra_info = pipeline._env_extra_info

    object_name = list(extra_info.object_reach_target_poses.keys())[0]
    obj = env.scene[object_name]
    obj_pos_w  = obj.data.root_pos_w[0]
    obj_quat_w = obj.data.root_quat_w[0]

    base_offset = extra_info.object_reach_target_poses[object_name][0]
    fix_x    = base_offset[0].item()
    fix_y    = base_offset[1].item()
    fix_quat = base_offset[3:].unsqueeze(0)   # [1, 4]

    print(f"\nObject '{object_name}' world pos : {obj_pos_w.cpu().numpy().round(3)}")
    print(f"Fixed oven-frame X={fix_x:.4f}  Y={fix_y:.4f}")
    print(f"Fixed quat (w,x,y,z) = {fix_quat[0].numpy().round(4)}")

    # Teleport robot to robot_dist from oven, facing +Y (yaw=90°)
    robot_z   = env.scene["robot"].data.root_pos_w[0, 2].item()
    robot_pos = torch.tensor(
        [obj_pos_w[0].item() + args_cli.x_offset,
         obj_pos_w[1].item() - args_cli.robot_dist,
         robot_z],
        dtype=torch.float32,
    )
    approach_yaw = math.pi / 2.0
    print(f"Teleporting robot to {robot_pos.numpy().round(3)}  (dist={args_cli.robot_dist}m from oven)")
    _teleport_robot(env, robot_pos, approach_yaw)

    # Squat
    _settle_squat(env, args_cli.squat_steps)

    robot    = env.scene["robot"]
    r_pos_w  = robot.data.root_pos_w[:1]
    r_quat_w = robot.data.root_quat_w[:1]
    print(f"Robot root pos (after squat): {r_pos_w[0].cpu().numpy().round(3)}")

    z_values = np.arange(args_cli.z_min, args_cli.z_max + args_cli.z_step / 2, args_cli.z_step)

    header = f"\n{'Z (oven)':>10}  {'IK':>4}  {'tgt_x':>7}  {'tgt_y':>7}  {'tgt_z':>7}  {'pos_err':>8}  {'rot_err':>9}"
    print(header)
    print("-" * len(header))

    for z in z_values:
        off_pos = torch.tensor([[fix_x, fix_y, z]], dtype=torch.float32)
        ee_pos_w, ee_quat_w = combine_frame_transforms(
            obj_pos_w.unsqueeze(0), obj_quat_w.unsqueeze(0),
            off_pos, fix_quat,
        )

        t_robot, q_robot = subtract_frame_transforms(
            r_pos_w, r_quat_w,
            ee_pos_w, ee_quat_w,
        )

        result = pipeline._motion_planner.solve_ik_batch(
            target_pos=t_robot,
            target_quat=q_robot,
        )

        ok      = bool(result.success[0].item())
        pos_err = float(result.position_error[0].item())
        rot_err = float(result.rotation_error[0].item())
        mark    = "✓" if ok else "✗"
        tx, ty, tz = t_robot[0].cpu().tolist()
        print(f"{z:>10.3f}  {mark:>4}  {tx:>7.3f}  {ty:>7.3f}  {tz:>7.3f}  {pos_err:>8.4f}  {rot_err:>9.4f}")

    print("\nSweep complete.")
    simulation_app.close()


if __name__ == "__main__":
    main()
