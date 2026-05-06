"""Script to run reach_plan_sweep."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sample dx/dy/dz/yaw around base reach pose and batch-plan with cuRobo.")
parser.add_argument("--pipeline_id", required=True, type=str)
parser.add_argument(
    "--reach_skill_index",
    default=0,
    type=int,
    help="Which reach skill to sweep at (0-based, globally across all subtasks)",
)
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--dx", default=0.01, type=float, help="dx range is [-dx, dx] meters")
parser.add_argument("--dy", default=0.01, type=float, help="dy range is [-dy, dy] meters")
parser.add_argument("--dz", default=0.01, type=float, help="dz range is [-dz, dz] meters")
parser.add_argument("--yaw_deg", default=10.0, type=float, help="yaw range is [-yaw_deg, yaw_deg] degrees")
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument(
    "--ik_only",
    action="store_true",
    help="Use IK-only solving instead of full motion planning (faster reachability check)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim / IsaacLab app
app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import math

import lw_benchhub.autosim  # noqa: F401

from autosim import make_pipeline
from autosim.calibration.plan_sweep import ReachPlanSweepCfg, reach_plan_sweep
from autosim.calibration.pose_sampler import OffsetSampler


def main() -> None:
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.cfg.motion_planner.use_cuda_graph = False
    reach_plan_sweep(
        pipeline,
        ReachPlanSweepCfg(
            reach_skill_index=args_cli.reach_skill_index,
            sampling=OffsetSampler(
                num_samples=args_cli.num_samples,
                dx_range=(-args_cli.dx, args_cli.dx),
                dy_range=(-args_cli.dy, args_cli.dy),
                dz_range=(-args_cli.dz, args_cli.dz),
                yaw_range_rad=(-math.radians(args_cli.yaw_deg), math.radians(args_cli.yaw_deg)),
                seed=args_cli.seed,
            ),
            top_k=args_cli.top_k,
            ik_only=args_cli.ik_only,
        ),
    )


if __name__ == "__main__":
    main()
