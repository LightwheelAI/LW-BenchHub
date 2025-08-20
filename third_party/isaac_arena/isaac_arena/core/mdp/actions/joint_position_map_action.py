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

from pathlib import Path
import yaml
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, List, Callable, Optional
import numpy as np
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.assets.articulation import Articulation


class JointPositionMapAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: "JointPositionMapActionCfg"

    @property
    def action_dim(self) -> int:
        return 1

    def apply_actions(self):
        processed_actions = self.cfg.post_process_fn(self.processed_actions, self._joint_names)
        self._asset.set_joint_position_target(processed_actions, joint_ids=self._joint_ids)


@configclass
class JointPositionMapActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = JointPositionMapAction
    joint_names: list[str] = MISSING
    post_process_fn: Callable = lambda x, y: x
