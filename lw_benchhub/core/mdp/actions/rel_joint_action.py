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

import torch

from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import RelativeJointPositionAction
from isaaclab.utils import configclass


class RelJointPositionAction(RelativeJointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: "RelJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._processed_actions = self._processed_actions * self._scale + self._offset


@configclass
class RelJointPositionActionCfg(RelativeJointPositionActionCfg):
    class_type = RelJointPositionAction
