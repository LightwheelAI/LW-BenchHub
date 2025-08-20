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
from pathlib import Path
import lwlab.core.mdp as mdp
from .base_gripper import BaseGripperCfg


class InspireHandsGripperCfg(BaseGripperCfg):
    def __init__(self, left_retageting_file_name: Path, right_retageting_file_name: Path):
        super().__init__(left_retageting_file_name, right_retageting_file_name)
        self.left_contact_body_name = "L_thumb_distal"
        self.right_contact_body_name = "R_thumb_distal"

    def left_hand_action_cfg(self):
        return {
            "tracking": mdp.DexRetargetingActionCfg(
                asset_name="robot",
                config_name=self.left_retageting_file_name,
                retargeting_index=[4, 6, 2, 0, 11, 8],  # pinky, ring, middle, index, thumbyaw, thumbpitch
                joint_names=["L_.*"],
                post_process_fn=self.post_process_left,
            )
        }

    def right_hand_action_cfg(self):
        return {
            "tracking": mdp.DexRetargetingActionCfg(
                asset_name="robot",
                config_name=self.right_retageting_file_name,
                retargeting_index=[4, 6, 2, 0, 11, 8],  # pinky, ring, middle, index, thumbyaw, thumbpitch
                joint_names=["R_.*"],
                post_process_fn=self.post_process_right,
            )
        }

    def post_process_left(self, retargeted_actions, num_joints):
        actions = np.zeros((retargeted_actions.shape[0], num_joints), dtype=np.float32)
        actions[:, 0] = actions[:, 5] = retargeted_actions[:, 3]  # index
        actions[:, 1] = actions[:, 6] = retargeted_actions[:, 2]  # middle
        actions[:, 2] = actions[:, 7] = retargeted_actions[:, 0]  # max(0,min(1.7,(retargeted_actions[:,0]-0.4)*1.7)) # pinky
        actions[:, 3] = actions[:, 8] = retargeted_actions[:, 1]  # ring
        actions[:, 4] = retargeted_actions[:, 5]  # thumbyaw
        actions[:, 9] = actions[:, 10] = actions[:, 11] = retargeted_actions[:, 4]  # thumbpitch
        return actions

    def post_process_right(self, retargeted_actions, num_joints):
        actions = np.zeros((retargeted_actions.shape[0], num_joints), dtype=np.float32)
        actions[:, 0] = actions[:, 5] = retargeted_actions[:, 3]  # index
        actions[:, 1] = actions[:, 6] = retargeted_actions[:, 2]  # middle
        actions[:, 2] = actions[:, 7] = retargeted_actions[:, 0]  # max(0,min(1.7,(retargeted_actions[:,0]-0.7)*1.7)) # pinky
        actions[:, 3] = actions[:, 8] = retargeted_actions[:, 1]  # ring
        actions[:, 4] = retargeted_actions[:, 5]  # thumbyaw
        actions[:, 9] = actions[:, 10] = actions[:, 11] = retargeted_actions[:, 4]  # thumbpitch
        return actions
