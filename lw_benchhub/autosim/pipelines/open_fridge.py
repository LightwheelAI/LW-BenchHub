import numpy as np
import torch
from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from autosim.decomposers import LLMDecomposerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from ..prompt_utils import render_additional_prompt
from ..robot_profiles import (
    TaskRobotOverride,
    apply_robot_env_cfg,
    build_env_extra_info,
    configure_robot_runtime_settings,
    resolve_robot_settings,
)


def _x7s_skill_cfg(cfg) -> None:
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 0.1
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.1
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 0.8
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.2
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.3
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.005
    cfg.skills.moveto.extra_cfg.uws_dwa                            = False
    cfg.skills.moveto.extra_cfg.sampling_radius                    = 1.0


def _g1_skill_cfg(cfg) -> None:
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 1.0
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.4
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 0.3
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.1
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.1
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.2
    cfg.skills.moveto.extra_cfg.use_dwa                            = False
    cfg.skills.moveto.extra_cfg.sampling_radius                    = 0.7
    cfg.max_steps = 1000
    cfg.action_adapter.squat_settle_steps = 0


TASK_ROBOT_OVERRIDES: dict[str, TaskRobotOverride] = {
    "x7s_joint_right": TaskRobotOverride(
        object_reach_target_poses={
            "fridge_main_group": [
                torch.tensor([0.047, -0.429, 0.125, 0.707, 0.0, 0.0, 0.707]),
            ],
        },
        init_state_pos_delta=(-0.45, -0.7, 0.0),
        skill_cfg_fn=_x7s_skill_cfg,
    ),
    "g1_loco_right": TaskRobotOverride(
        object_reach_target_poses={
            "fridge_main_group": [
                torch.tensor([0.07, -0.429, 0.25, 0.707, 0.0, 0.0, 0.707]),
            ],
        },
        init_state_pos_delta=(-0.15, -1.5, 0.0),
        skill_cfg_fn=_g1_skill_cfg,
        skill_finger_configs={
            "right_hand": {
                "grasp": (1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8),
                "push": (1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8),
            }
        },
    ),
}


def get_task_robot_override(robot_profile: str) -> TaskRobotOverride:
    try:
        return TASK_ROBOT_OVERRIDES[robot_profile]
    except KeyError as exc:
        supported = ", ".join(TASK_ROBOT_OVERRIDES)
        raise ValueError(
            f"OpenFridgePipeline does not support robot profile '{robot_profile}'. Supported: {supported}"
        ) from exc


@configclass
class OpenFridgePipelineCfg(AutoSimPipelineCfg):
    robot_profile: str = "x7s_joint_right"
    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    def __post_init__(self):
        resolved_robot = resolve_robot_settings(self.robot_profile, override=get_task_robot_override(self.robot_profile))
        configure_robot_runtime_settings(self, resolved_robot)

        if resolved_robot.override.skill_cfg_fn:
            resolved_robot.override.skill_cfg_fn(self)

        self.skills.push.extra_cfg.move_offset = 0.3
        self.skills.push.extra_cfg.move_axis   = "-x"

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes = [
            "Scene/island_island_group",
        ]


@configclass
class G1OpenFridgePipelineCfg(OpenFridgePipelineCfg):
    robot_profile: str = "g1_loco_right"


class OpenFridgePipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        self._resolved_robot = resolve_robot_settings(
            cfg.robot_profile, override=get_task_robot_override(cfg.robot_profile)
        )
        super().__init__(cfg)

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        from lw_benchhub.utils.env import ExecuteMode, parse_env_cfg

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="OpenFridge",
            robot_name=self._resolved_robot.profile.robot_name,
            scene_name="robocasakitchen-2-2",
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

        apply_robot_env_cfg(env_cfg, self._resolved_robot)
        env_cfg.terminations.time_out = None

        env_id = f"Robocasa-OpenFridge-{self._resolved_robot.profile.robot_name}-v0"
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

        return gym.make(env_id, cfg=env_cfg).unwrapped

    def get_env_extra_info(self):
        return build_env_extra_info(
            task_name="Robocasa-Task-OpenFridge",
            objects=["fridge_main_group", "robot"],
            additional_prompt_contents=render_additional_prompt(),
            resolved_robot=self._resolved_robot,
            object_navigate_sample_range={
                "fridge_main_group": (3 / 2 * np.pi, 7 / 4 * np.pi),
            },
        )
