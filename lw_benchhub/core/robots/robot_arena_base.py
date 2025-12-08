# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING
from typing import Optional

import numpy as np
import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg, RecorderTerm, RecorderTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.utils.cameras import make_camera_observation_cfg
from isaaclab_arena.utils.configclass import combine_configclass_instances
from isaaclab_arena.utils.pose import Pose

import lw_benchhub.core.mdp as mdp
import lw_benchhub.core.mdp as lw_benchhub_mdp
import lw_benchhub.utils.math_utils.transform_utils.numpy_impl as Tn
import lw_benchhub.utils.place_utils.env_utils as EnvUtils
from lw_benchhub.core.context import get_context
from lw_benchhub.utils.csv_loader import csv_loader
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.isaaclab_utils import get_robot_joint_target_from_scene
from lw_benchhub.utils.log_utils import get_action_space_def, get_camera_metadata
from lw_benchhub.utils.place_utils.env_utils import get_safe_robot_anchor


@configclass
class EmbodimentBaseActionsCfg:
    """Action specifications for the MDP."""
    """Action specifications for the MDP."""

    # arm_action: Optional[mdp.ActionTermCfg] = None
    # gripper_action: Optional[mdp.ActionTermCfg] = None
    # base_action: Optional[mdp.ActionTermCfg] = None
    pass


class JointReplayPositionAction(mdp.JointPositionAction):
    def __init__(self, cfg: mdp.JointPositionActionCfg, env: ManagerBasedEnv):
        self.decimation = env.cfg.decimation
        self.apply_action_count = 0
        super().__init__(cfg, env)
        self._offset = 0

    @property
    def action_dim(self):
        return self._num_joints * self.decimation

    def apply_actions(self):
        # TODO: if self.processed_actions shape has no decimation, means that's an old data, no need to slice it
        # to_apply_action = self.processed_actions[self.apply_action_count % self.decimation]
        to_apply_action = self.processed_actions.reshape(self.decimation, -1, self._num_joints)[self.apply_action_count % self.decimation]

        self._asset.set_joint_position_target(to_apply_action, joint_ids=self._joint_ids)
        self.apply_action_count += 1


@configclass
class JointReplayPositionActionCfg(mdp.JointPositionActionCfg):
    """Joint targets for the MDP."""
    class_type: type[mdp.ActionTerm] = JointReplayPositionAction
    asset_name: str = "robot"
    joint_names = ".*"
    # joint_targets: mdp.ActionTermCfg = mdp.JointActionCfg()


class PrePhysicsStepJointTargetsRecorder(RecorderTerm):
    """Recoder term that records the IK Target Pos in the beginning of each step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.decimation = env.cfg.decimation
        self._record_count = 0
        self._buffer_joint_targets = []

    def stack_joint_targets(self):
        # TODO: stack joint targets
        if not self._buffer_joint_targets:
            raise RuntimeError("No joint targets to stack")
        stacked_joint_targets = dict()
        joint_targets = self._buffer_joint_targets[0]
        for name in joint_targets:
            stacked = torch.stack([joint_target[name] for joint_target in self._buffer_joint_targets], dim=1)
            stacked_joint_targets[name] = stacked.reshape(stacked.shape[0], -1)
        self._buffer_joint_targets = []
        return stacked_joint_targets

    def record_pre_physics_step(self):
        self._record_count += 1
        self._buffer_joint_targets.append(get_robot_joint_target_from_scene(self._env.scene))
        if self._record_count % self.decimation == 0:
            return "joint_targets", self.stack_joint_targets()
        return None, None

    def record_pre_reset(self, env_ids):
        if self._env.cfg.scene_backend == "robocasa" and self._env.cfg.task_backend == "robocasa":
            for env_id, success in zip(env_ids, (self._env.cfg.isaaclab_arena_env.task._success_cache >= self._env.cfg.isaaclab_arena_env.task._success_count)[env_ids]):
                if not success.item():
                    self._env.recorder_manager._episodes[env_id] = EpisodeData()
        return None, None


@configclass
class PrePhysicsStepJointTargetsRecorderCfg(RecorderTermCfg):
    """Configuration for the step IK Target recorder term."""

    class_type: type[RecorderTerm] = PrePhysicsStepJointTargetsRecorder


@configclass
class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_joint_targets = PrePhysicsStepJointTargetsRecorderCfg()


@configclass
class EmbodimentBaseSceneCfg:
    robot: ArticulationCfg = MISSING


@configclass
class EmbodimentGeneralObsCfg(ObsGroup):
    actions: ObsTerm = ObsTerm(func=lw_benchhub_mdp.last_action)
    joint_pos: ObsTerm = ObsTerm(func=lw_benchhub_mdp.joint_pos_rel)
    joint_vel: ObsTerm = ObsTerm(func=lw_benchhub_mdp.joint_vel_rel)
    eef_pos: ObsTerm = ObsTerm(func=lw_benchhub_mdp.ee_frame_pos)
    eef_quat: ObsTerm = ObsTerm(func=lw_benchhub_mdp.ee_frame_quat)

    def __post_init__(self):
        self.concatenate_terms = False


@configclass
class EmbodimentBaseObservationCfg:
    """Observations for policy group."""

    embodiment_general_obs: EmbodimentGeneralObsCfg = EmbodimentGeneralObsCfg()


@configclass
class EmbodimentBasePolicyObservationCfg(ObsGroup):
    """Observations for policy group."""
    pass


class LwEmbodimentBase(EmbodimentBase):
    observation_cameras: dict | None = {}
    active_observation_camera_names: list[str] = []
    robot_vis_helper_cfg: dict | None = None
    robot_base_link: str | None = None

    def __init__(self, enable_cameras: bool = False, initial_pose: Optional[Pose] = None):
        self.context = get_context()
        super().__init__(enable_cameras, initial_pose)
        self.enable_cameras = self.context.enable_cameras
        self.scene_config = EmbodimentBaseSceneCfg()
        self.action_config = EmbodimentBaseActionsCfg()
        self.observation_config = EmbodimentBaseObservationCfg()
        self.policy_observation_config = EmbodimentBasePolicyObservationCfg()
        self.event_config = None
        self.mimic_env = None
        self.offset_config = None
        self.robot_spawn_deviation_pos_x = 0.15
        self.robot_spawn_deviation_pos_y = 0.05
        self.robot_spawn_deviation_rot = 0.0
        self.robot_to_fixture_dist = 0.20
        self.reward_gripper_joint_names = []
        self.reward_arm_joint_names = []
        self.add_camera_to_observation = self.context.add_camera_to_observation

        self.set_default_offset_config()

    def modify_observation_cameras(self, task_type: str):
        if self.context.replay_cfgs and "render_resolution" in self.context.replay_cfgs:
            render_resolution = self.context.replay_cfgs["render_resolution"]
            if render_resolution is not None:
                for cam_info in self.observation_cameras.values():
                    if self.context.execute_mode in cam_info["execute_mode"]:
                        cam_info["camera_cfg"].width = render_resolution[0]
                        cam_info["camera_cfg"].height = render_resolution[1]

    def _setup_camera_config(self, task_type):
        for cam_name, cam_info in self.observation_cameras.items():
            if self.context.execute_mode in cam_info["execute_mode"]:
                cam_info = self.observation_cameras[cam_name]
                setattr(self.camera_config, cam_name, cam_info["camera_cfg"])
        if self.enable_cameras and self.camera_config and self.add_camera_to_observation:
            camera_observation_config = make_camera_observation_cfg(self.camera_config)
            camera_observations = camera_observation_config.camera_obs
            for field_name in camera_observations.__dataclass_fields__:
                field = camera_observations.__dataclass_fields__[field_name]
                if isinstance(field.type, type) and issubclass(field.type, ObsTerm):
                    self.active_observation_camera_names.append(field_name)
            self.policy_observation_config = combine_configclass_instances(
                "PolicyObservationCfg",
                camera_observations,
                self.policy_observation_config,
            )

    def get_recorder_term_cfg(self):
        if self.context.execute_mode not in [ExecuteMode.TRAIN, ExecuteMode.EVAL, ExecuteMode.REPLAY_STATE]:
            return RecorderManagerCfg()

    def get_observation_cfg(self):
        return self.observation_config

    def get_policy_observation_cfg(self):
        return self.policy_observation_config

    def preprocess_device_action(self, action: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def get_default_action(self, device: torch.device, num_envs: int) -> torch.Tensor:
        return None

    def filter_action(self, action: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Apply simple EMA smoothing on absolute-mode arm actions only.
        # This function does not modify gripper commands.
        # It preserves relative-mode actions untouched.
        left_arm_action = action.get("left_arm_action", None)
        right_arm_action = action.get("right_arm_action", None)

        if self.last_pos is None and left_arm_action is not None and right_arm_action is not None:
            self.last_pos = {}
            self.last_pos["left_arm_action"] = left_arm_action.clone()
            self.last_pos["right_arm_action"] = right_arm_action.clone()

        if self.last_pos is not None and left_arm_action is not None and right_arm_action is not None:
            last_left_arm_action = self.last_pos["left_arm_action"]
            last_right_arm_action = self.last_pos["right_arm_action"]

            max_vel = 0.05
            min_vel = 0.002

            for i in [0, 1, 2]:
                if left_arm_action[i] - last_left_arm_action[i] > max_vel:
                    left_arm_action[i] = last_left_arm_action[i] + max_vel
                elif left_arm_action[i] - last_left_arm_action[i] < -max_vel:
                    left_arm_action[i] = last_left_arm_action[i] - max_vel

                if abs(left_arm_action[i] - last_left_arm_action[i]) < min_vel:
                    left_arm_action[i] = last_left_arm_action[i]

            for i in [0, 1, 2]:
                if right_arm_action[i] - last_right_arm_action[i] > max_vel:
                    right_arm_action[i] = last_right_arm_action[i] + max_vel
                elif right_arm_action[i] - last_right_arm_action[i] < -max_vel:
                    right_arm_action[i] = last_right_arm_action[i] - max_vel
                if abs(right_arm_action[i] - last_right_arm_action[i]) < min_vel:
                    right_arm_action[i] = last_right_arm_action[i]

            self.last_pos["left_arm_action"] = left_arm_action.clone()
            self.last_pos["right_arm_action"] = right_arm_action.clone()

        return action

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> IsaacLabArenaManagerBasedRLEnvCfg:
        env_cfg = super().modify_env_cfg(env_cfg)
        if self.context.execute_mode == ExecuteMode.REPLAY_JOINT_TARGETS:
            self.set_replay_joint_targets_action(env_cfg)
        return env_cfg

    def set_replay_joint_targets_action(self, cfg):
        cfg.actions = EmbodimentBaseActionsCfg()
        cfg.actions.joint_targets = JointReplayPositionActionCfg()

    def get_ep_meta(self, env=None):
        ep_meta = {}
        ep_meta["robot_name"] = self.name
        ep_meta["init_robot_base_pos_anchor"] = self.init_robot_base_pos_anchor.tolist()
        ep_meta["init_robot_base_ori_anchor"] = self.init_robot_base_ori_anchor.tolist()
        ep_meta["camera_cfg"] = get_camera_metadata(self.observation_cameras)
        if self.context.execute_mode != ExecuteMode.REPLAY_JOINT_TARGETS:
            ep_meta["action_space_definition"] = get_action_space_def(env)
        return ep_meta

    def set_default_offset_config(self):
        self.offset_config = {
            "left_gripper_offset": np.zeros(3,),
            "right_gripper_offset": np.zeros(3,),
            "controller2gripper_l_arm": np.eye(4),
            "controller2gripper_r_arm": np.eye(4),
        }

    def setup_env_config(self, orchestrator):
        if orchestrator.task.context.task_backend == "robocasa":
            robot_pos, robot_ori \
                = csv_loader.load_robot_pose(orchestrator.embodiment.name,
                                             orchestrator.scene.scene_name,
                                             orchestrator.task.task_name)
            if robot_pos is not None and robot_ori is not None:
                orchestrator.task.resample_robot_placement_on_reset = False
                self.init_robot_base_pos_anchor = np.array(robot_pos, dtype=np.float32)
                self.init_robot_base_ori_anchor = np.array(robot_ori, dtype=np.float32)
            else:
                self.init_robot_base_pos_anchor, self.init_robot_base_ori_anchor = self.get_robot_anchor(orchestrator)
            self.scene_config.robot.init_state.pos = self.init_robot_base_pos_anchor
            self.scene_config.robot.init_state.rot = Tn.convert_quat(Tn.mat2quat(Tn.euler2mat(self.init_robot_base_ori_anchor)), to="wxyz")
            self.modify_observation_cameras(orchestrator.task.task_type)
            self._setup_camera_config(orchestrator.task.task_type)
        elif orchestrator.task.context.task_backend == "local":
            self.init_robot_base_pos_anchor = np.array(self.scene_config.robot.init_state.pos)
            self.init_robot_base_ori_anchor = Tn.mat2euler(Tn.quat2mat(Tn.convert_quat(np.array(self.scene_config.robot.init_state.rot), to="xyzw")))

    def get_robot_anchor(self, orchestrator):
        (
            robot_base_pos_anchor,
            robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(orchestrator)

        if hasattr(orchestrator.embodiment, "robot_base_offset"):
            try:
                robot_base_pos_anchor += np.array(orchestrator.embodiment.robot_base_offset["pos"])
                robot_base_ori_anchor += np.array(orchestrator.embodiment.robot_base_offset["rot"])
            except KeyError:
                raise ValueError("offset value is not correct !! please make sure offset has key pos and rot !!")

        # Intercept the unsafe anchor and make it safe
        safe_anchor_pos, safe_anchor_ori = get_safe_robot_anchor(
            cfg=orchestrator.embodiment,
            unsafe_anchor_pos=robot_base_pos_anchor,
            unsafe_anchor_ori=robot_base_ori_anchor
        )
        return safe_anchor_pos, safe_anchor_ori
