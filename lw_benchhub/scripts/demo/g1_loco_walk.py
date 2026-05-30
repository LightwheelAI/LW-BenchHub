# Copyright 2025 Lightwheel Team
# SPDX-License-Identifier: Apache-2.0
"""Standalone G1 locomotion demo.

Drives the bundled `loco.onnx` / `squat.onnx` whole-body controllers (the exact
``Controller_loco`` / ``Controller_squat`` used by ``LegPositionAction``) with a
scripted forward-walk command -- no teleop device needed -- and records an mp4.

Run:
    OMNI_KIT_ACCEPT_EULA=YES LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
    python ./lw_benchhub/scripts/demo_g1_loco_walk.py --out results/g1_loco_walk.mp4
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 loco walking demo recorder.")
parser.add_argument("--out", type=str, default="results/g1_loco_walk.mp4", help="output mp4 path")
parser.add_argument("--vx", type=float, default=0.3, help="forward velocity command (m/s)")
parser.add_argument("--settle_steps", type=int, default=60, help="control steps to stand before walking")
parser.add_argument("--walk_steps", type=int, default=300, help="control steps to walk forward")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- after app is up, imports that need omni / isaaclab ----
import os
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

from lw_benchhub.core.robots.unitree.assets_cfg import G1_Loco_CFG
from lw_benchhub.core.mdp.config import Config
from lw_benchhub.core.mdp.actions.controller import Controller_loco, Controller_squat
from lw_benchhub.core.mdp.helpers.rotation_helper import get_gravity_orientation

# Leg joints in the policy's expected order (matches LegPositionAction.loco_joint_ids /
# g1_loco.yaml default_angles: left leg then right leg, each hip_pitch,roll,yaw,knee,ankle_pitch,ankle_roll)
LEG_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def main():
    def log(m):
        print(f"[demo] {m}", flush=True)

    loco_cfg = Config("lw_benchhub/core/mdp/configs/g1_loco.yaml")
    squat_cfg = Config("lw_benchhub/core/mdp/configs/g1_squat.yaml")
    sim_dt = float(loco_cfg.simulation_dt)              # 0.02 / 4 = 0.005
    decimation = int(loco_cfg.control_decimation)       # 4

    # --- sim ---
    log("creating SimulationContext")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device))
    log("spawning ground + light")
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    log("spawning G1 articulation")
    robot_cfg = G1_Loco_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    log("spawning camera")
    cam_cfg = CameraCfg(
        prim_path="/World/cam",
        height=360, width=480,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 1000.0)),
    )
    camera = Camera(cam_cfg)

    log("calling sim.reset() (first render -> may compile shaders, can take minutes)")
    sim.reset()
    log("sim.reset() done")
    print("[demo] joint names:", robot.joint_names, flush=True)
    leg_ids = [robot.joint_names.index(n) for n in LEG_ORDER]
    print("[demo] leg ids (policy order):", leg_ids, [robot.joint_names[i] for i in leg_ids])
    pelvis_idx = robot.find_bodies("pelvis")[0][0]

    # default joint targets (hold arms/waist), legs overwritten by policy
    default_q = robot.data.default_joint_pos.clone()

    # --- controllers (same classes LegPositionAction uses) ---
    loco = Controller_loco(loco_cfg, None)
    squat = Controller_squat(squat_cfg, None)
    loco.reset()
    squat.reset()
    loco.stance_command = False
    target_dof_pos = loco_cfg.default_angles.copy()          # (12,) legs in policy order
    policy_mode = "squat"

    frames = []
    total_ctrl = args_cli.settle_steps + args_cli.walk_steps

    def read_obs():
        jp = robot.data.joint_pos[0, leg_ids].cpu().numpy().astype(np.float32)
        jv = robot.data.joint_vel[0, leg_ids].cpu().numpy().astype(np.float32)
        quat = robot.data.body_link_quat_w[0, pelvis_idx].cpu().numpy()        # wxyz
        angw = robot.data.body_link_ang_vel_w[0, pelvis_idx].cpu().numpy()
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        from scipy.spatial.transform import Rotation as R
        angb = R.from_quat(xyzw).inv().apply(angw).astype(np.float32)
        grav = get_gravity_orientation(quat).astype(np.float32)
        return jp, jv, grav, angb

    print(f"[demo] running {total_ctrl} control steps ({decimation} phys steps each, dt={sim_dt})")
    for c in range(total_ctrl):
        walking = c >= args_cli.settle_steps
        vx = args_cli.vx if walking else 0.0
        cmd = np.array([vx, 0.0, 0.0], dtype=np.float32)
        squat_flag = 0.0 if walking else 1.0

        # mode switch (squat <-> loco) like LegPositionAction.check_mode_switch
        if squat_flag:
            if policy_mode != "squat":
                policy_mode = "squat"; squat.set_transition_count(); loco.reset_gait()
        else:
            if policy_mode != "loco":
                policy_mode = "loco"; loco.set_transition_count()

        jp, jv, grav, angb = read_obs()
        if policy_mode == "loco":
            target_dof_pos = loco.run(cmd, grav, angb, jp, jv, target_dof_pos.copy())
        # else: settle phase -- hold legs at default_angles (robot stands via stiff PD)

        q_target = default_q.clone()
        q_target[0, leg_ids] = torch.tensor(np.asarray(target_dof_pos, dtype=np.float32), device=q_target.device)

        # apply over `decimation` physics steps
        for _ in range(decimation):
            robot.set_joint_position_target(q_target)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

        # follow camera on the robot, capture every 2nd control step (~25 fps)
        base = robot.data.root_pos_w[0].cpu().numpy()
        eye = torch.tensor([[base[0] + 2.2, base[1] - 2.2, base[2] + 1.0]], device=args_cli.device, dtype=torch.float32)
        tgt = torch.tensor([[base[0], base[1], base[2] - 0.2]], device=args_cli.device, dtype=torch.float32)
        camera.set_world_poses_from_view(eye, tgt)
        camera.update(sim_dt)
        if c % 2 == 0:
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            frames.append(rgb)
        if c % 25 == 0:
            print(f"[demo] step {c}/{total_ctrl} mode={policy_mode} base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) frames={len(frames)}")

    os.makedirs(os.path.dirname(args_cli.out) or ".", exist_ok=True)
    import imageio
    imageio.mimsave(args_cli.out, frames, fps=25, quality=8)
    print(f"[demo] wrote {len(frames)} frames -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        # always release the Isaac Sim process so a crash never leaks GPU memory
        simulation_app.close()
