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

import numpy as np
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets.episode_data import EpisodeData

import lw_benchhub.core.mdp as mdp
from lw_benchhub.core.cfg import LwBaseCfg
from lw_benchhub.utils.isaaclab_utils import get_robot_joint_target_from_scene


OFFSET_CONFIG = {
    "left_gripper_offset": np.zeros(3,),
    "right_gripper_offset": np.zeros(3,),
    "controller2gripper_l_arm": np.eye(4),
    "controller2gripper_r_arm": np.eye(4),
}


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    """Action specifications for the MDP."""

    # arm_action: Optional[mdp.ActionTermCfg] = None
    # gripper_action: Optional[mdp.ActionTermCfg] = None
    # base_action: Optional[mdp.ActionTermCfg] = None
    pass


class JointReplayPositionAction(mdp.JointPositionAction):
    def __init__(self, cfg: mdp.JointPositionActionCfg, env: LwBaseCfg):
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

    def __init__(self, cfg: RecorderTermCfg, env: LwBaseCfg):
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
        for env_id, success in zip(env_ids, (self._env.cfg.isaaclab_arena_env.scene.success_cache >= self._env.cfg.isaaclab_arena_env.scene.success_count)[env_ids]):
            if not success.item():
                self._env.recorder_manager._episodes[env_id] = EpisodeData()
        return None, None


@configclass
class PrePhysicsStepJointTargetsRecorderCfg(RecorderTermCfg):
    """Configuration for the step IK Target recorder term."""

    class_type: type[RecorderTerm] = PrePhysicsStepJointTargetsRecorder


class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_joint_targets = PrePhysicsStepJointTargetsRecorderCfg()
