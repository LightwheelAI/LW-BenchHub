# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import MISSING
from typing import Any, Optional

import numpy as np
import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import RecorderTerm, RecorderTermCfg, ActionStateRecorderManagerCfg
from isaaclab.utils import configclass

import isaac_arena.core.mdp as mdp
from isaac_arena.core import IsaacArenaBaseCfg
from isaac_arena.geometry.pose import Pose
from isaac_arena.isaaclab_utils import get_robot_joint_target_from_scene
from isaac_arena.utils.env import ExecuteMode


OFFSET_CONFIG = {
    "left_gripper_offset": np.zeros(3,),
    "right_gripper_offset": np.zeros(3,),
    "controller2gripper_l_arm": np.eye(4),
    "controller2gripper_r_arm": np.eye(4),
}


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Define the action terms here, e.g.:
    # base_action: Optional[mdp.ActionTermCfg] = None
    # arm_action: Optional[mdp.ActionTermCfg] = None
    # gripper_action: Optional[mdp.ActionTermCfg] = None
    pass


class JointReplayPositionAction(mdp.JointPositionAction):
    def __init__(self, cfg: mdp.JointPositionActionCfg, env: IsaacArenaBaseCfg):
        self.decimation = env.cfg.decimation
        self.apply_action_count = 0
        super().__init__(cfg, env)
        self._offset = 0

    @property
    def action_dim(self) -> int:
        return self._num_joints * self.decimation

    def apply_actions(self):
        action_to_apply = self.process_actions.reshape(self.decimation, -1, self._num_joints)[self.apply_action_count % self.decimation]

        self._asset.set_joint_position_target(action_to_apply, joint_ids=self._joint_ids)
        self.apply_action_count += 1


@configclass
class JointReplayPositionActionCfg(mdp.JointPositionActionCfg):
    """Joint targets for the MDP."""
    class_type = type[mdp.ActionTerm] = JointReplayPositionAction
    asset_name = str = "robot"
    joing_names = ".*"
    # joint_targets mdp.ActionTermCfg = mdp.JointActionCfg()


class PrePhysicsStepJointTargetsRecorder(RecorderTerm):
    """Recorder term that records the IK target pos at the beginning of each step."""

    def __init__(self, cfg: RecorderTermCfg, env: IsaacArenaBaseCfg):
        super().__init__(cfg, env)
        self.decimation = env.cfg.decimation
        self._record_count = 0
        self._buffer_joint_targets = []

    def stack_joint_targets(self):
        if not self._buffer_joint_targets:
            return RuntimeError("No joint targets to stack.")
        stacked_joint_targets = dict()
        joint_targets = self._buffer_joint_targets[0]
        for name in joint_targets:
            stacked = torch.stack([jt[name] for jt in self._buffer_joint_targets], dim=1)
            stacked_joint_targets[name] = stacked.reshape(stacked.shape[0], -1)
        self._buffer_joint_targets = []
        return stacked_joint_targets

    def record_pre_physics_step(self):
        self._record_count += 1
        self._buffer_joint_targets.append(get_robot_joint_target_from_scene(self._env.scene))
        if self._record_count % self.decimation == 0:
            return "joint_targets", self.stack_joint_targets()
        return None, None


@configclass
class PrePhysicsJointTargetRecorderCfg(RecorderTermCfg):
    """Configuration for the step IK target recorder term."""
    class_type: type[RecorderTerm] = PrePhysicsStepJointTargetsRecorder


class RecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_joint_targets = PrePhysicsJointTargetRecorderCfg()


class EmbodimentBaseCfg(IsaacArenaBaseCfg):

    name: str | None = None
    tags: list[str] = ["embodiment"]  # TODO: append lw tags
    robot_cfg: ArticulationCfg = MISSING
    actions: ActionsCfg = ActionsCfg()
    robot_pos: Optional[tuple[float, float, float]] = None
    robot_rot: Optional[tuple[float, float, float, float]] = None
    """robot_ori is in quaternion format (w, x, y, z)"""
    offset_config: Optional[dict] = OFFSET_CONFIG
    robot_to_fixture_dist: float = 0.20
    robot_base_link: str = None
    observation_cameras: dict = {}

    def __post_init__(self):
        super().__post_init__()
        self.robot_cfg = self.robot_cfg.copy().replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.robot_pos is not None:
            self.robot_cfg.init_state.pos = self.robot_pos
        if self.robot_rot is not None:
            self.robot_cfg.init_state.rot = self.robot_rot

        self.scene.robot = self.robot_cfg

        self.viewport_cfg = {
            "offset": [-0.4, 0.0, 0.8],
            "lookat": [1.0, 0.0, -0.85]
        }
        if self.execute_mode == ExecuteMode.TELEOP:
            self.recorders = RecorderManagerCfg()

    def set_replay_joint_targets_action(self):
        self.actions = ActionsCfg()
        self.actions.joint_targets = JointReplayPositionActionCfg()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["robot_name"] = self.name
        return ep_meta
