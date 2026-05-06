import numpy as np
import torch
from autosim.core.pipeline import AutoSimPipeline, AutoSimPipelineCfg
from autosim.decomposers import LLMDecomposerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass

from lw_benchhub.autosim.prompt_utils import render_additional_prompt
from lw_benchhub.autosim.robot_profiles import (
    TaskRobotOverride,
    apply_robot_env_cfg,
    build_env_extra_info,
    configure_robot_runtime_settings,
    resolve_robot_settings,
)


def _x7s_skill_cfg(cfg) -> None:
    cfg.skills.moveto.extra_cfg.local_planner.max_linear_velocity  = 0.4
    cfg.skills.moveto.extra_cfg.local_planner.max_angular_velocity = 0.3
    cfg.skills.moveto.extra_cfg.local_planner.predict_time         = 0.4
    cfg.skills.moveto.extra_cfg.global_planner.safety_distance     = 1.0
    cfg.skills.moveto.extra_cfg.global_planner.proximity_weight    = 3.0
    cfg.skills.moveto.extra_cfg.waypoint_tolerance                 = 0.2
    cfg.skills.moveto.extra_cfg.goal_tolerance                     = 0.07
    cfg.skills.moveto.extra_cfg.yaw_tolerance                      = 0.01
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
    cfg.skills.moveto.extra_cfg.per_object_sampling_radius         = {"obj": 0.53, "stovetop_main_group": 0.45}
    cfg.skills.moveto.extra_cfg.per_object_yaw_tolerance           = {"obj": 0.01, "stovetop_main_group": 0.4}


def _x7s_get_obj_cfgs(self):
    return [dict(
        name="obj",
        obj_groups=("kettle_non_electric"),
        asset_name="Kettle073.usd",
        graspable=True,
        placement=dict(
            fixture=self.counter,
            sample_region_kwargs={},
            size=(0.4, 0.28),
            pos=(0.0, -1.0),
            rotation=(7 / 8 * np.pi, np.pi),
        ),
    )]


def _x7s_reset_env(pipeline) -> None:
    obj = pipeline._env.scene["obj"]
    obj.write_root_pose_to_sim(
        torch.tensor([[2.0, -0.56, 1.1, 0.0, 0.0, 0.0, 1.0]], device=pipeline._env.device)
    )
    obj.reset()


def _g1_get_obj_cfgs(self):
    return [
        dict(
            name="obj",
            obj_groups=("kettle_non_electric"),
            asset_name="Kettle073.usd",
            graspable=True,
            placement=dict(
                fixture=self.counter,
                sample_region_kwargs=dict(ref=self.stove),
                size=(0.35, 0.35),
                pos=("ref", -1),
            ),
        ),
        dict(
            name="stove_distr",
            obj_groups=("pan"),
            asset_name="Pan023.usd",
            placement=dict(
                fixture=self.stove,
                ensure_object_boundary_in_range=False,
            ),
        ),
    ]


def _g1_reset_env(pipeline) -> None:
    obj = pipeline._env.scene["obj"]
    obj.write_root_pose_to_sim(
        torch.tensor([[2.0, -0.56, 1.1, 0.0, 0.0, 0.0, 1.0]], device=pipeline._env.device)
    )
    obj.reset()

    stove_distr = pipeline._env.scene["stove_distr"]
    pose = stove_distr.data.root_pose_w.clone()
    pose[:, 3:] = torch.tensor([0.707, 0.0, 0.0, 0.707], device=pipeline._env.device)
    pose[:, 0] += 0.3
    pose[:, 1] += 0.2
    stove_distr.write_root_pose_to_sim(pose)
    stove_distr.reset()


def _g1_after_env_created(pipeline, env) -> None:
    env.cfg.isaaclab_arena_env.task.init_fixtures(env)


TASK_ROBOT_OVERRIDES: dict[str, TaskRobotOverride] = {
    "x7s_joint_left": TaskRobotOverride(
        extra_target_link_names=("link20_tip",),
        reach_extra_target_mode="keep_initial_relative_offset",
        object_reach_target_poses={
            "obj": [
                torch.tensor([0.0, 0.09, 0.10, 0.707, 0.0, 0.0, -0.707]),
            ],
            "stovetop_main_group": [
                torch.tensor([-0.0, -0.045, 0.24, 0.707, 0.0, 0.0, 0.707]),
            ],
        },
        init_state_pos_delta=(0.0, -0.8, 0.01),
        skill_cfg_fn=_x7s_skill_cfg,
        get_obj_cfgs_fn=_x7s_get_obj_cfgs,
        reset_env_fn=_x7s_reset_env,
    ),
    "g1_loco_left": TaskRobotOverride(
        object_reach_target_poses={
            "obj": [
                torch.tensor([-0.05, 0.22, 0.08, 0.707, 0.0, 0.0, -0.707]),
            ],
            "stovetop_main_group": [
                torch.tensor([-0.0, -0.20, 0.2, 0.707, 0.0, 0.0, 0.707]),
            ],
        },
        init_state_pos_delta=(0.0, -0.8, 0.01),
        skill_cfg_fn=_g1_skill_cfg,
        get_obj_cfgs_fn=_g1_get_obj_cfgs,
        reset_env_fn=_g1_reset_env,
        after_env_created_fn=_g1_after_env_created,
        skill_finger_configs={
            "left_hand": {
                "grasp": (-1.2, -1.2, -1.2, -1.2, -1.0, -1.0, -1.0),
                "lift": (-1.2, -1.2, -1.2, -1.2, -1.0, -1.0, -1.0),
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
            f"KettleBoilingPipeline does not support robot profile '{robot_profile}'. Supported: {supported}"
        ) from exc


@configclass
class KettleBoilingPipelineCfg(AutoSimPipelineCfg):
    robot_profile: str = "x7s_joint_left"
    decomposer: LLMDecomposerCfg = LLMDecomposerCfg()

    def __post_init__(self):
        resolved_robot = resolve_robot_settings(self.robot_profile, override=get_task_robot_override(self.robot_profile))
        configure_robot_runtime_settings(self, resolved_robot)

        if resolved_robot.override.skill_cfg_fn:
            resolved_robot.override.skill_cfg_fn(self)

        self.skills.lift.extra_cfg.move_offset = 0.10
        self.skills.lift.extra_cfg.move_axis   = "+z"

        self.motion_planner.enable_dynamic_world_sync = True
        self.occupancy_map.floor_prim_suffix = "Scene/floor_room"
        self.max_steps = 1500

        self.motion_planner.world_ignore_subffixes = ["Scene/floor_room"]
        self.motion_planner.world_only_subffixes   = [
            "Scene/obj",
            "Scene/stovetop_main_group",
            "Scene/counter_main_main_group",
            "Scene/counter_1_main_group",
        ]


@configclass
class G1KettleBoilingPipelineCfg(KettleBoilingPipelineCfg):
    robot_profile: str = "g1_loco_left"


class KettleBoilingPipeline(AutoSimPipeline):
    def __init__(self, cfg: AutoSimPipelineCfg):
        self._resolved_robot = resolve_robot_settings(
            cfg.robot_profile, override=get_task_robot_override(cfg.robot_profile)
        )
        super().__init__(cfg)

    def reset_env(self):
        super().reset_env()
        if self._resolved_robot.override.reset_env_fn:
            self._resolved_robot.override.reset_env_fn(self)

    def load_env(self) -> ManagerBasedEnv:
        import gymnasium as gym
        import lw_benchhub_tasks.lightwheel_robocasa_tasks.multi_stage.brewing.kettle_boiling as kb
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        if self._resolved_robot.override.get_obj_cfgs_fn:
            kb.KettleBoiling._get_obj_cfgs = self._resolved_robot.override.get_obj_cfgs_fn

        env_cfg = parse_env_cfg(
            scene_backend="robocasa",
            task_backend="robocasa",
            task_name="KettleBoiling",
            robot_name=self._resolved_robot.profile.robot_name,
            scene_name="robocasakitchen-7-3",
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

        env_id = f"Robocasa-KettleBoiling-{self._resolved_robot.profile.robot_name}-v0"
        gym.register(
            id=env_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={},
            disable_env_checker=True,
        )

        env = gym.make(env_id, cfg=env_cfg, render_mode="rgb_array").unwrapped

        if self._resolved_robot.override.after_env_created_fn:
            self._resolved_robot.override.after_env_created_fn(self, env)

        return env

    def get_env_extra_info(self):
        return build_env_extra_info(
            task_name="Robocasa-Task-KettleBoiling",
            objects=["obj", "stovetop_main_group"],
            additional_prompt_contents=f"{render_additional_prompt()}\n\n You don't need to turn on burner.",
            resolved_robot=self._resolved_robot,
        )
