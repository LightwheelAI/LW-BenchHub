"""Sweep G1 base distance from object and test IK to a fixed EE target.

Teleports the robot to a series of distances from the object along the
approach axis, runs IK-only to the configured reach target, and reports
which distances yield a successful IK solution.

Usage:
    python robot_distance_sweep.py \
        --pipeline_id LWBenchhub-Autosim-G1CloseOvenPipeline-v0 \
        --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep robot base distance and test IK to fixed EE target.")
parser.add_argument("--pipeline_id", required=True, type=str)
parser.add_argument("--object_name", default=None, type=str,
                    help="Scene object name to measure distance from (default: first object in pipeline)")
parser.add_argument("--d_min",    default=0.4,  type=float, help="Minimum distance from object (m)")
parser.add_argument("--d_max",    default=1.5,  type=float, help="Maximum distance from object (m)")
parser.add_argument("--d_step",   default=0.1,  type=float, help="Distance step (m)")
parser.add_argument("--x_offset", default=0.0,  type=float,
                    help="Robot base X offset relative to object X (e.g. -0.21 for G1)")
parser.add_argument("--squat_steps", default=0, type=int,
                    help="Steps to squat before IK (0 = no squat; use 40 for G1)")
parser.add_argument("--n_orient_samples", default=1, type=int,
                    help="Number of EE orientation samples per distance (default 1 = use base quaternion only). "
                         "Samples are evenly spaced yaw rotations around world Z applied to the base quaternion.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import math

import torch
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms, quat_mul

import lw_benchhub.autosim  # noqa: F401
from autosim import make_pipeline


def _gen_orient_samples(base_quat_wxyz: torch.Tensor, n: int) -> list[torch.Tensor]:
    """Return n quaternions covering evenly-spaced yaw rotations (world Z) applied to base_quat.

    When n==1 returns [base_quat] unchanged (original orientation only).
    """
    if n <= 1:
        return [base_quat_wxyz]
    samples = []
    for i in range(n):
        yaw = 2 * math.pi * i / n
        half = yaw / 2.0
        dq = torch.tensor([math.cos(half), 0.0, 0.0, math.sin(half)],
                          dtype=torch.float32)  # rotation around world Z
        q = quat_mul(dq.unsqueeze(0), base_quat_wxyz.unsqueeze(0))[0]
        samples.append(q)
    return samples


def _get_object_pose(env, object_name: str):
    obj = env.scene[object_name]
    pos_w  = obj.data.root_pos_w[0]   # [3]
    quat_w = obj.data.root_quat_w[0]  # [w, x, y, z]
    return pos_w, quat_w


def _settle_squat(env, squat_steps: int) -> None:
    """Switch to squat/stance mode using low-level stepping (bypasses recorder)."""
    if squat_steps <= 0:
        return
    action = torch.zeros(env.action_space.shape, device=env.device)
    action[0, 0] = -1.0  # negative cmd drives squat_cmd[0] down toward min height
    action[0, 3] = 1.0   # mode=1: squat/stance
    env.action_manager.process_action(action)
    for _ in range(squat_steps):
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())


def _teleport_robot(env, pos_w: torch.Tensor, yaw_rad: float, settle_steps: int = 20):
    half = yaw_rad / 2.0
    quat = torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=torch.float32, device=env.device,
    )  # [w, x, y, z]
    pose = torch.zeros(1, 7, device=env.device)
    pose[0, :3] = pos_w.to(env.device)
    pose[0, 3:]  = quat

    robot = env.scene["robot"]
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
    for _ in range(settle_steps):
        env.sim.step()
    env.scene.update(env.sim.get_physics_dt())


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.cfg.motion_planner.use_cuda_graph = False
    pipeline.initialize()

    env          = pipeline._env
    extra_info   = pipeline._env_extra_info

    # Pick the target object
    object_name = args_cli.object_name
    if object_name is None:
        object_name = list(extra_info.object_reach_target_poses.keys())[0]

    # Get the object's world pose
    obj_pos_w, obj_quat_w = _get_object_pose(env, object_name)

    # Compute EE target in world frame
    offset = extra_info.object_reach_target_poses[object_name][0]  # [x,y,z, qw,qx,qy,qz]
    offset_pos  = offset[:3].unsqueeze(0)
    offset_quat = offset[3:].unsqueeze(0)

    ee_pos_w, ee_quat_w = combine_frame_transforms(
        obj_pos_w.unsqueeze(0), obj_quat_w.unsqueeze(0),
        offset_pos, offset_quat,
    )
    ee_pos_w  = ee_pos_w[0]   # [3]
    ee_quat_w = ee_quat_w[0]  # [4]

    print(f"\nObject '{object_name}' world pos : {obj_pos_w.cpu().numpy().round(3)}")
    print(f"EE target world pos             : {ee_pos_w.cpu().numpy().round(3)}")
    print(f"EE target world quat (w,x,y,z)  : {ee_quat_w.cpu().numpy().round(3)}")

    n_orient = args_cli.n_orient_samples
    orient_samples = _gen_orient_samples(ee_quat_w, n_orient)
    print(f"Orientation samples per distance: {n_orient}")

    # Robot spawn height from the G1 profile default
    robot_z   = env.scene["robot"].data.root_pos_w[0, 2].item()
    # Approach from -y side (robot south of object), facing +y
    approach_yaw = math.pi / 2.0

    import numpy as np
    distances = np.arange(args_cli.d_min, args_cli.d_max + args_cli.d_step / 2, args_cli.d_step)

    header = f"\n{'Dist (m)':>10}  {'IK':>4}  {'tgt_x':>7}  {'tgt_y':>7}  {'tgt_z':>7}  {'pos_err':>8}  {'rot_err':>9}  {'idx':>4}"
    print(header)
    print("-" * len(header))

    for d in distances:
        # Teleport robot
        robot_pos = torch.tensor(
            [obj_pos_w[0].item() + args_cli.x_offset, obj_pos_w[1].item() - d, robot_z],
            dtype=torch.float32,
        )
        _teleport_robot(env, robot_pos, approach_yaw)
        _settle_squat(env, args_cli.squat_steps)

        # Express EE position in robot root frame (read AFTER squatting — pelvis may have dropped)
        robot = env.scene["robot"]
        r_pos_w  = robot.data.root_pos_w[:1]   # [1, 3]
        r_quat_w = robot.data.root_quat_w[:1]  # [1, 4]

        best_ok, best_pos_err, best_rot_err, best_idx = False, float("inf"), float("inf"), -1
        best_t_robot = None

        for i, q_sample in enumerate(orient_samples):
            t_robot, q_robot = subtract_frame_transforms(
                r_pos_w, r_quat_w,
                ee_pos_w.unsqueeze(0), q_sample.unsqueeze(0).to(r_pos_w.device),
            )  # [1,3], [1,4]

            result = pipeline._motion_planner.solve_ik_batch(
                target_pos=t_robot,
                target_quat=q_robot,
            )

            ok      = bool(result.success[0].item())
            pos_err = float(result.position_error[0].item())
            rot_err = float(result.rotation_error[0].item())

            if ok and not best_ok:
                best_ok, best_pos_err, best_rot_err, best_idx = True, pos_err, rot_err, i
                best_t_robot = t_robot[0].cpu()
                break  # first success is enough
            if not best_ok and pos_err < best_pos_err:
                best_pos_err, best_rot_err, best_idx = pos_err, rot_err, i
                best_t_robot = t_robot[0].cpu()

        mark = "✓" if best_ok else "✗"
        tx, ty, tz = best_t_robot.tolist() if best_t_robot is not None else (0, 0, 0)
        print(f"{d:>10.2f}  {mark:>4}  {tx:>7.3f}  {ty:>7.3f}  {tz:>7.3f}  {best_pos_err:>8.4f}  {best_rot_err:>9.4f}  {best_idx:>4}")

    print("\nSweep complete.")


if __name__ == "__main__":
    main()
