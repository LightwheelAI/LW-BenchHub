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

from re import M
import torch
import numpy as np
from dataclasses import MISSING
from typing import List, Optional

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import RecorderTerm, RecorderTermCfg, ActionStateRecorderManagerCfg
from isaaclab.utils.datasets.episode_data import EpisodeData

import lwlab.core.mdp as mdp
from lwlab.utils.env import ExecuteMode
from lwlab.utils.isaaclab_utils import get_robot_joint_target_from_scene
from lwlab.core.context import get_context
from lwlab.utils.place_utils.env_utils import get_safe_robot_anchor
import lwlab.utils.place_utils.env_utils as EnvUtils
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn


@configclass
class ActionsCfg:
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
    # joint_targets: mdp.ActionTermCfg = mdp.JointActionCfg(
    # )


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
        for env_id, success in zip(env_ids, (self._env.cfg.isaac_arena_env.scene.success_cache >= self._env.cfg.isaac_arena_env.scene.success_count)[env_ids]):
            if not success.item():
                self._env.recorder_manager._episodes[env_id] = EpisodeData()
        return None, None


@configclass
class PrePhysicsStepJointTargetsRecorderCfg(RecorderTermCfg):
    """Configuration for the step IK Target recorder term."""

    class_type: type[RecorderTerm] = PrePhysicsStepJointTargetsRecorder


class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_joint_targets = PrePhysicsStepJointTargetsRecorderCfg()


from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.utils.pose import Pose
from typing import Dict, Any, Callable


class LwLabEmbodimentBase(EmbodimentBase):
    observation_cameras: dict | None = None
    robot_vis_helper_cfg: dict | None = None

    def __init__(self, enable_cameras: bool = False, initial_pose: Optional[Pose] = None):
        self.context = get_context()
        super().__init__(enable_cameras, initial_pose)
        self.scene_config = None
        self.action_config = ActionsCfg()
        self.observation_config = None
        self.event_config = None
        self.mimic_env = None
        self.robot_spawn_deviation_pos_x = 0.15
        self.robot_spawn_deviation_pos_y = 0.05
        self.robot_spawn_deviation_rot = 0.0

        self.set_default_offset_config()

    def get_recorder_term_cfg(self):
        if self.context.execute_mode in [ExecuteMode.TELEOP, ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS]:
            return RecorderManagerCfg()

    def preprocess_device_action(self, action: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

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

    def set_replay_joint_targets_action(self, cfg):
        cfg.actions = ActionsCfg()
        cfg.actions.joint_targets = JointReplayPositionActionCfg()

    def get_ep_meta(self):
        ep_meta = {}
        ep_meta["robot_name"] = self.name
        return ep_meta

    def set_default_offset_config(self):
        self.offset_config = {
            "left_gripper_offset": np.zeros(3,),
            "right_gripper_offset": np.zeros(3,),
            "controller2gripper_l_arm": np.eye(4),
            "controller2gripper_r_arm": np.eye(4),
        }

    def setup_env_config(self, orchestrator):
        self.place_robot(orchestrator)

    def place_robot(self, orchestrator):
        robot_base_pos_anchor, robot_base_ori_anchor = self.get_robot_anchor(orchestrator)
        self.scene_config.robot.init_state.pos = robot_base_pos_anchor
        self.scene_config.robot.init_state.rot = Tn.convert_quat(Tn.mat2quat(Tn.euler2mat(robot_base_ori_anchor)), to="wxyz")

    def get_robot_anchor(self, orchestrator):
        (
            robot_base_pos_anchor,
            robot_base_ori_anchor,
        ) = EnvUtils.init_robot_base_pose(orchestrator)

        if hasattr(orchestrator, "robot_base_offset"):
            try:
                robot_base_pos_anchor += np.array(orchestrator.robot_base_offset["pos"])
                robot_base_ori_anchor += np.array(orchestrator.robot_base_offset["rot"])
            except KeyError:
                raise ValueError("offset value is not correct !! please make sure offset has key pos and rot !!")

        # Intercept the unsafe anchor and make it safe
        safe_anchor_pos, safe_anchor_ori = get_safe_robot_anchor(
            cfg=orchestrator.embodiment,
            unsafe_anchor_pos=robot_base_pos_anchor,
            unsafe_anchor_ori=robot_base_ori_anchor
        )
        return safe_anchor_pos, safe_anchor_ori
