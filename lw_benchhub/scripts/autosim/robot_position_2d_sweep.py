"""Sweep G1 base position (x_offset, distance) and test IK to a fixed EE target.

2D sweep over x_offset and distance from object, testing IK reachability.

Usage:
    python robot_position_2d_sweep.py \
        --pipeline_id LWBenchhub-Autosim-G1OpenFridgePipeline-v0 \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="2D sweep of robot position (x_offset, distance) and test IK.")
parser.add_argument("--pipeline_id", required=True, type=str)
parser.add_argument("--object_name", default=None, type=str,
                    help="Scene object name (default: first object in pipeline)")
parser.add_argument("--x_min",    default=-0.6,  type=float, help="Minimum x_offset (m)")
parser.add_argument("--x_max",    default=0.2,   type=float, help="Maximum x_offset (m)")
parser.add_argument("--x_step",   default=0.1,   type=float, help="X step (m)")
parser.add_argument("--d_min",    default=0.4,   type=float, help="Minimum distance (m)")
parser.add_argument("--d_max",    default=1.5,   type=float, help="Maximum distance (m)")
parser.add_argument("--d_step",   default=0.1,   type=float, help="Distance step (m)")
parser.add_argument("--squat_steps", default=40, type=int, help="Squat steps (0=no squat)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import math

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

    env = pipeline._env
    extra_info = pipeline._env_extra_info

    object_name = args_cli.object_name
    if object_name is None:
        object_name = list(extra_info.object_reach_target_poses.keys())[0]

    obj = env.scene[object_name]
    obj_pos_w = obj.data.root_pos_w[0]
    obj_quat_w = obj.data.root_quat_w[0]

    offset = extra_info.object_reach_target_poses[object_name][0]
    offset_pos = offset[:3].unsqueeze(0)
    offset_quat = offset[3:].unsqueeze(0)

    ee_pos_w, ee_quat_w = combine_frame_transforms(
        obj_pos_w.unsqueeze(0), obj_quat_w.unsqueeze(0),
        offset_pos, offset_quat,
    )
    ee_pos_w = ee_pos_w[0]
    ee_quat_w = ee_quat_w[0]

    print(f"\nObject '{object_name}' world pos : {obj_pos_w.cpu().numpy().round(3)}")
    print(f"EE target world pos             : {ee_pos_w.cpu().numpy().round(3)}")
    print(f"EE target world quat (w,x,y,z)  : {ee_quat_w.cpu().numpy().round(3)}")

    robot_z = env.scene["robot"].data.root_pos_w[0, 2].item()
    approach_yaw = math.pi / 2.0

    x_offsets = np.arange(args_cli.x_min, args_cli.x_max + args_cli.x_step / 2, args_cli.x_step)
    distances = np.arange(args_cli.d_min, args_cli.d_max + args_cli.d_step / 2, args_cli.d_step)

    print(f"\nScanning {len(x_offsets)} x_offsets × {len(distances)} distances = {len(x_offsets) * len(distances)} positions\n")

    results = []

    for x_off in x_offsets:
        for d in distances:
            robot_pos = torch.tensor(
                [obj_pos_w[0].item() + x_off, obj_pos_w[1].item() - d, robot_z],
                dtype=torch.float32,
            )
            _teleport_robot(env, robot_pos, approach_yaw)
            _settle_squat(env, args_cli.squat_steps)

            robot = env.scene["robot"]
            r_pos_w = robot.data.root_pos_w[:1]
            r_quat_w = robot.data.root_quat_w[:1]

            t_robot, q_robot = subtract_frame_transforms(
                r_pos_w, r_quat_w,
                ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0),
            )

            result = pipeline._motion_planner.solve_ik_batch(
                target_pos=t_robot,
                target_quat=q_robot,
            )

            ok = bool(result.success[0].item())
            pos_err = float(result.position_error[0].item())
            rot_err = float(result.rotation_error[0].item())

            results.append((x_off, d, ok, pos_err, rot_err))

    # Print results
    print(f"\n{'x_offset':>9}  {'dist':>6}  {'IK':>4}  {'pos_err':>8}  {'rot_err':>9}")
    print("-" * 50)
    for x_off, d, ok, pos_err, rot_err in results:
        mark = "✓" if ok else "✗"
        print(f"{x_off:>9.2f}  {d:>6.2f}  {mark:>4}  {pos_err:>8.4f}  {rot_err:>9.4f}")

    # Summary: show successful positions
    success_list = [(x, d) for x, d, ok, _, _ in results if ok]
    if success_list:
        print(f"\n✓ Found {len(success_list)} successful positions:")
        for x, d in success_list:
            print(f"  x_offset={x:.2f}, distance={d:.2f}")
    else:
        print("\n✗ No successful IK solutions found in the scanned range.")

    print("\nSweep complete.")
    simulation_app.close()
    import sys
    sys.exit(0)


if __name__ == "__main__":
    main()

