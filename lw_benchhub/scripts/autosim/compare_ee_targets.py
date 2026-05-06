"""Compare world-frame EE reach targets between X7S and G1 in the CloseOven scene.

Loads a single pipeline, reads the oven's world pose, then computes the EE world
target for both robot profiles using their oven-frame offsets and prints the
difference.  Only one Isaac Sim instance is needed.

Usage:
    python compare_ee_targets.py \
        --pipeline_id LWBenchhub-Autosim-CloseOvenPipeline-v0 \
        --headless
    # or use the G1 pipeline — same scene, same oven pose
    python compare_ee_targets.py \
        --pipeline_id LWBenchhub-Autosim-G1CloseOvenPipeline-v0 \
        --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare X7S and G1 EE world targets in CloseOven scene.")
parser.add_argument("--pipeline_id", required=True, type=str)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import torch
from isaaclab.utils.math import combine_frame_transforms

import lw_benchhub.autosim  # noqa: F401
from autosim import make_pipeline


# Oven-frame EE offsets as defined in close_oven.py TASK_ROBOT_OVERRIDES
_OFFSETS = {
    "x7s_joint_left": torch.tensor([-0.176, -0.739, -0.180, 0.707, -0.00, -0.00, 0.707]),
    "g1_loco_left":   torch.tensor([-0.176, -0.739, -0.180, 0.707,  0.00,  0.00, 0.707]),
}


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.cfg.motion_planner.use_cuda_graph = False
    pipeline.initialize()

    env        = pipeline._env
    extra_info = pipeline._env_extra_info

    object_name = list(extra_info.object_reach_target_poses.keys())[0]
    obj         = env.scene[object_name]
    obj_pos_w   = obj.data.root_pos_w[0]   # [3]
    obj_quat_w  = obj.data.root_quat_w[0]  # [w, x, y, z]

    # Print robot actual spawn position (after delta applied)
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[0].cpu().numpy().round(4)
    print(f"\nRobot actual spawn pos (world): {robot_pos_w}")
    print(f"Object '{object_name}'")
    print(f"  world pos  : {obj_pos_w.cpu().numpy().round(4)}")
    print(f"  world quat : {obj_quat_w.cpu().numpy().round(4)}  (w,x,y,z)")
    y_dist = abs(float(robot_pos_w[1]) - float(obj_pos_w[1].item()))
    x_dist = float(robot_pos_w[0]) - float(obj_pos_w[0].item())
    print(f"  Robot->Oven  Y distance: {y_dist:.3f} m  (sweep d value)")
    print(f"  Robot->Oven  X offset:   {x_dist:.3f} m\n")

    results = {}
    for profile, offset in _OFFSETS.items():
        off_pos  = offset[:3].unsqueeze(0)
        off_quat = offset[3:].unsqueeze(0)

        ee_pos_w, ee_quat_w = combine_frame_transforms(
            obj_pos_w.unsqueeze(0), obj_quat_w.unsqueeze(0),
            off_pos, off_quat,
        )
        results[profile] = (ee_pos_w[0], ee_quat_w[0])

        print(f"[{profile}]")
        print(f"  oven-frame offset pos  : {offset[:3].numpy().round(4)}")
        print(f"  oven-frame offset quat : {offset[3:].numpy().round(4)}  (w,x,y,z)")
        print(f"  => EE world pos        : {ee_pos_w[0].cpu().numpy().round(4)}")
        print(f"  => EE world quat       : {ee_quat_w[0].cpu().numpy().round(4)}  (w,x,y,z)\n")

    x7s_pos, x7s_quat = results["x7s_joint_left"]
    g1_pos,  g1_quat  = results["g1_loco_left"]

    pos_diff  = (x7s_pos - g1_pos).abs().max().item()
    quat_diff = (x7s_quat - g1_quat).abs().max().item()

    print(f"Max pos  difference (X7S - G1): {pos_diff:.6f} m")
    print(f"Max quat difference (X7S - G1): {quat_diff:.6f}")

    if pos_diff < 1e-4:
        print("=> EE world POSITIONS are identical.")
    else:
        print("=> EE world positions DIFFER — oven may be placed differently per profile.")

    simulation_app.close()


if __name__ == "__main__":
    main()
