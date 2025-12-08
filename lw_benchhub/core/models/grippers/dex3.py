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

import lw_benchhub.core.mdp as mdp

from .base_gripper import BaseGripperCfg


class Dex3GripperCfg(BaseGripperCfg):
    def __init__(self, left_retageting_file_name: str, right_retageting_file_name: str):
        super().__init__(left_retageting_file_name, right_retageting_file_name)
        self.left_contact_body_name = "left_hand_thumb_2_link"
        self.right_contact_body_name = "right_hand_thumb_2_link"

    def left_hand_action_cfg(self):
        return {
            "tracking": mdp.DexRetargetingActionCfg(
                asset_name="robot",
                config_name=self.left_retageting_file_name,
                retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],# in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
                joint_names=["left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                             "left_hand_index_0_joint", "left_hand_index_1_joint",
                             "left_hand_middle_0_joint", "left_hand_middle_1_joint"],
                # joint_names=["left_hand_thumb_0_joint","left_hand_thumb_1_joint","left_hand_thumb_2_joint"]
                post_process_fn=self.post_process_left
            ),
            "handle": mdp.JointPositionMapActionCfg(
                asset_name="robot",
                joint_names=['left_hand_thumb_1_joint', 'left_hand_thumb_2_joint', 'left_hand_index.*', 'left_hand_middle.*'],
                post_process_fn=self.process_hand,
            ),
            "rl": mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["left_hand_.*"],
                open_command_expr={'left_hand_index.*': 0.0, 'left_hand_middle.*': 0.0, 'left_hand_thumb.*': 0.0},
                close_command_expr={'left_hand_thumb_0_joint': 0.0, 'left_hand_thumb_1_joint': -np.pi / 6,
                                    'left_hand_thumb_2_joint': -np.pi / 6, 'left_hand_index_1_joint': np.pi / 3, 'left_hand_middle_1_joint': np.pi / 3,
                                    'left_hand_index_0_joint': np.pi / 6, 'left_hand_middle_0_joint': np.pi / 6},
            )
        }

    def right_hand_action_cfg(self):
        return {
            "tracking": mdp.DexRetargetingActionCfg(
                asset_name="robot",
                config_name=self.right_retageting_file_name,
                retargeting_index=[0, 2, 4, 1, 3, 5, 6],  # [4,5,6],#in0,in1,mi0,mi1, th0,th1,th2 ==> in0,mi0,th0,th1,mi1,in1,th2
                joint_names=["right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                             "right_hand_index_0_joint", "right_hand_index_1_joint",
                             "right_hand_middle_0_joint", "right_hand_middle_1_joint"],
                # joint_names=["right_hand_thumb_0_joint","right_hand_thumb_1_joint","right_hand_thumb_2_joint"]
                post_process_fn=self.post_process_right
            ),
            "handle": mdp.JointPositionMapActionCfg(
                asset_name="robot",
                joint_names=['right_hand_thumb_1_joint', 'right_hand_thumb_2_joint', 'right_hand_index.*', 'right_hand_middle.*'],
                post_process_fn=self.process_hand,
            ),
            "rl": mdp.BinaryJointPositionActionCfg(
                asset_name="robot",
                joint_names=["right_hand_.*"],
                open_command_expr={'right_hand_index.*': 0.0, 'right_hand_middle.*': 0.0, 'right_hand_thumb.*': 0.0},
                close_command_expr={'right_hand_thumb_0_joint': 0.0, 'right_hand_thumb_1_joint': -np.pi / 6,
                                    'right_hand_thumb_2_joint': -np.pi / 6, 'right_hand_index_1_joint': np.pi / 3, 'right_hand_middle_1_joint': np.pi / 3,
                                    'right_hand_index_0_joint': np.pi / 6, 'right_hand_middle_0_joint': np.pi / 6},
            )
        }

    def post_process_left(self, retargeted_actions, num_joints):
        scale = 1.3
        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = (retargeted_actions[:, 0]) * np.pi / 2 * scale  # in 0
        actions[:, 1] = (retargeted_actions[:, 1]) * np.pi / 2 * scale  # mi 0
        actions[:, 2] = 0  # th 0
        actions[:, 3] = retargeted_actions[:, 0] * scale  # in 1
        actions[:, 4] = retargeted_actions[:, 1] * scale  # mi 1
        actions[:, 5] = np.pi / 3 + retargeted_actions[:, 2] * scale  # th 1
        actions[:, 6] = np.pi / 3 + retargeted_actions[:, 2] * scale  # th 2
        return actions

    def post_process_right(self, retargeted_actions, num_joints):
        scale = 1.3
        actions = np.zeros_like(retargeted_actions, dtype=np.float32)
        actions[:, 0] = (retargeted_actions[:, 0]) * np.pi / 2 * scale  # in 0
        actions[:, 1] = (retargeted_actions[:, 1]) * np.pi / 2 * scale  # mi 0
        actions[:, 2] = 0  # th 0
        actions[:, 3] = retargeted_actions[:, 0] * scale  # in 1
        actions[:, 4] = retargeted_actions[:, 1] * scale  # mi 1
        actions[:, 5] = -np.pi / 3 - retargeted_actions[:, 2] * scale  # th 1
        actions[:, 6] = -np.pi / 3 - retargeted_actions[:, 2] * scale  # th 2
        return actions

    def process_hand(self, target_pos, joint_names):
        if target_pos.shape[-1] == 1:
            target_pos = target_pos.repeat(1, len(joint_names))
        if joint_names[0].startswith('left'):
            target_pos = (target_pos + 1) / 2
        else:
            target_pos = -(target_pos + 1) / 2
        target_pos[:, 0:4] *= -1.4  # index 0 middle 0 index 1 middlrbe 1
        target_pos[:, 4] *= np.pi / 6  # thumb 1
        target_pos[:, 5] *= 1.3  # thumb 2
        return target_pos
