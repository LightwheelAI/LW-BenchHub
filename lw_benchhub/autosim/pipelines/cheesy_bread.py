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
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 3.5
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 3.0
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 1.1
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.2
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.1
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.005
    cfg.skills.moveto.extra_cfg.uws_dwa                            = False


def _g1_skill_cfg(cfg) -> None:
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 1.0
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.4
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 0.5
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.25
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.30
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.01
    cfg.skills.moveto.extra_cfg.use_dwa                            = False
    cfg.skills.moveto.extra_cfg.per_object_sampling_radius         = {"cheese": 0.53, "bread": 0.53}
    cfg.skills.moveto.extra_cfg.per_object_yaw_tolerance           = {"cheese": 0.01, "bread": 0.1}


TASK_ROBOT_OVERRIDES: dict[str, TaskRobotOverride] = {
    "x7s_joint_left": TaskRobotOverride(
        extra_target_link_names=("link20_tip",),
        object_reach_target_poses={
            "cheese": [
                torch.tensor([0.003, -0.049, 0.025, 0.705, -0.002, 0.05, 0.707]),
            ],
            "bread": [
                torch.tensor([-0.05, -0.05, 0.13, 0.9238, 0.0, 0.0, 0.3827]),
            ],
        },
        init_state_pos_delta=(0.0, -0.8, 0.0),
        skill_cfg_fn=_x7s_skill_cfg,
    ),
    "g1_loco_left": TaskRobotOverride(
        object_reach_target_poses={
            "cheese": [
                torch.tensor([0.003, -0.049, 0.025, 0.705, -0.002, 0.05, 0.707]),
            ],
            "bread": [
                torch.tensor([-0.05, -0.05, 0.13, 0.9238, 0.0, 0.0, 0.3827]),
            ],
        },
        init_state_pos_delta=(0.0, -0.8, 0.01),
        skill_cfg_fn=_g1_skill_cfg,
    ),
}


def get_task_robot_override(robot_profile: str) -> TaskRobotOverride:
    try:
        return TASK_ROBOT_OVERRIDES[robot_profile]
    except KeyError as exc:
        supported = ", ".join(TASK_ROBOT_OVERRIDES)
        raise ValueError(
            f"CheesyBreadPipeline does not support robot profile '{robot_profile}'. Supported: {supported}"
        ) from exc


@configclass
class CheesyBreadPipelineCfg(AutoSimPipelineCfg):
    """Configuration for the CheesyBreadPipeline."""

    robot_profile: str = "x7s_joint_left"
    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    def __post_init__(self):
        resolved_robot = resolve_robot_settings(self.robot_profile, override=get_task_robot_override(self.robot_profile))
        configure_robot_runtime_settings(self, resolved_robot)

        if resolved_robot.override.skill_cfg_fn:
            resolved_robot.override.skill_cfg_fn(self)

        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"
        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes = [
            "Scene/bread",
            "Scene/cheese",
            "Scene/counter_main_main_group",
            "Scene/counter_1_front_group",
        ]


@configclass
class G1CheesyBreadPipelineCfg(CheesyBreadPipelineCfg):
    robot_profile: str = "g1_loco_left"


class CheesyBreadPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        self._resolved_robot = resolve_robot_settings(
            cfg.robot_profile, override=get_task_robot_override(cfg.robot_profile)
        )
        super().__init__(cfg)

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        import lw_benchhub_tasks.lightwheel_robocasa_tasks.multi_stage.making_toast.cheesy_bread as cb
        from lw_benchhub.utils.env import ExecuteMode, parse_env_cfg

        cb.CheesyBread._get_obj_cfgs = _get_obj_cfgs

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="CheesyBread",
            robot_name=self._resolved_robot.profile.robot_name,
            scene_name="robocasakitchen-9-8",
            robot_scale=1.0,
            device="cpu",
            num_envs=1,
            use_fabric=False,
            first_person_view=False,
            enable_cameras=False,
            execute_mode=ExecuteMode.TRAIN,
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

        env_id = f"Robocasa-CheesyBread-{self._resolved_robot.profile.robot_name}-v0"
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

        return gym.make(env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped

    def get_env_extra_info(self):
        return build_env_extra_info(
            task_name="Robocasa-Task-CheesyBread",
            objects=self._env.scene.keys(),
            additional_prompt_contents=(
                f"{render_additional_prompt()}\n\n After grasp the cheess, you need to lift it up and place it on the bread."
            ),
            resolved_robot=self._resolved_robot,
        )


def _get_obj_cfgs(self):
    return [
        dict(
            name="bread",
            obj_groups="bread_flat",
            asset_name="Bread013.usd",
            object_scale=1.5,
            placement=dict(
                fixture=self.counter,
                size=(0.5, 0.4),
                pos=(0, -1.0),
                rotation=(0.0, 0.0),
                try_to_place_in="cutting_board",
            ),
        ),
        dict(
            name="cheese",
            obj_groups="cheese",
            asset_name="Cheese003.usd",
            init_robot_here=True,
            placement=dict(
                ref_obj="bread_container",
                fixture=self.counter,
                size=(1.0, 0.08),
                pos=(-0.8, -1.0),
                rotation=(0.0, 0.0),
            ),
        ),
        dict(
            name="distr_counter",
            obj_groups="all",
            placement=dict(fixture=self.counter, size=(1.0, 0.20), pos=(0, 1.0)),
        ),
    ]
