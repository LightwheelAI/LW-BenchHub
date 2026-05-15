from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

import torch
import random
from autosim.decomposers import LLMDecomposerCfg

from ..prompt_utils import render_additional_prompt
from ..robot_profiles import (
    TaskRobotOverride,
    build_env_extra_info,
    configure_robot_runtime_settings,
    resolve_robot_settings,
)

TASK_ROBOT_OVERRIDES: dict[str, TaskRobotOverride] = {
    "x7s_joint_left": TaskRobotOverride(
        extra_target_link_names=("link20_tip",),
        reach_extra_target_mode="keep_current",
        object_reach_target_poses={
            "toaster_oven_main_group": [
                 torch.tensor([ 
                random.uniform(-0.03, 0.03),
                -0.1899999105930328,
                0.08999999865889549,
                0.5,
                -0.5,
                -0.5,
                0.5   
        ]),
            ],
        },
    ),
}


def get_task_robot_override(robot_profile: str) -> TaskRobotOverride:
    try:
        return TASK_ROBOT_OVERRIDES[robot_profile]
    except KeyError as exc:
        supported = ", ".join(tuple(TASK_ROBOT_OVERRIDES))
        raise ValueError(
            f"OpenToasterOvenDoorPipeline does not support robot profile '{robot_profile}'. Supported profiles: {supported}"
        ) from exc


@configclass
class OpenToasterOvenDoorPipelineCfg(AutoSimPipelineCfg):
    """Configuration for the OpenToasterOvenDoorPipeline."""

    robot_profile: str = "x7s_joint_left"

    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    def __post_init__(self):
        resolved_robot = resolve_robot_settings(
            self.robot_profile,
            override=get_task_robot_override(self.robot_profile),
        )
        configure_robot_runtime_settings(self, resolved_robot)

        self.skills.moveto.extra_cfg.local_planner.max_linear_velocity = 0.1
        self.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.4
        self.skills.moveto.extra_cfg.local_planner.predict_time = 0.4
        self.skills.moveto.extra_cfg.global_planner.safety_distance = 0.8
        self.skills.moveto.extra_cfg.global_planner.proximity_weight = 3.0
        self.skills.moveto.extra_cfg.waypoint_tolerance = 0.2
        self.skills.moveto.extra_cfg.goal_tolerance = 0.1
        self.skills.moveto.extra_cfg.yaw_tolerance = 0.008
        self.skills.moveto.extra_cfg.uws_dwa = False
        self.skills.moveto.extra_cfg.sampling_radius = 1.2
        self.max_steps = 1000

        self.skills.pull.extra_cfg.move_offset = 0.38
        self.skills.pull.extra_cfg.move_axis = "-x"

        self.skills.lift.extra_cfg.move_offset = 0.10
        self.skills.lift.extra_cfg.move_axis = "+z"

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.collision_enable_substrings = ["Collisions", "top_geometry"]
        self.motion_planner.world_only_subffixes = [
            "Scene/toaster_oven_main_group",
            "Scene/counter_main_main_group",
            "Scene/counter_1_main_group",
            "Scene/stack_1_main_group_4",
            "Scene/island_island_group",
        ]


class OpenToasterOvenDoorPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        super().__init__(cfg)
        robot_profile = cfg.robot_profile
        self._resolved_robot = resolve_robot_settings(
            robot_profile,
            override=get_task_robot_override(robot_profile),
        )

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import ExecuteMode, parse_env_cfg

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="OpenToasterOvenDoor",
            robot_name=self._resolved_robot.profile.robot_name,
            scene_name="robocasakitchen-3-5",
            robot_scale=1.0,
            device="cpu",
            num_envs=1,
            use_fabric=False,
            first_person_view=False,
            enable_cameras=False,
            execute_mode=ExecuteMode.TELEOP,
            usd_simplify=False,
            seed=42,
            sources=["objaverse", "lightwheel", "aigen_objs"],
            object_projects=[],
            rl_name=None,
            headless_mode=False,
            replay_cfgs={"add_camera_to_observation": True, "render_resolution": (640, 480)},
            resample_robot_placement_on_reset=False,
        )

        env_cfg.terminations.time_out = None

        env_cfg.scene.robot.init_state.pos[0] -= 0.4
        env_cfg.scene.robot.init_state.pos[1] -= 0.8

        env_id = f"Robocasa-OpenToasterOvenDoor-{self._resolved_robot.profile.robot_name}-v0"
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

        env = gym.make(env_id, cfg=env_cfg).unwrapped

        return env

    def get_env_extra_info(self):
        return build_env_extra_info(
            task_name="Robocasa-Task-OpenToasterOvenDoor",
            objects=["toaster_oven_main_group"],
            additional_prompt_contents=(
                f"{render_additional_prompt()}"
            ),
            resolved_robot=self._resolved_robot,
        )
