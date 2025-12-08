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

class BaseGripperCfg:
    """
    Base gripper configuration.
    This class is used to configure the gripper for the robot.
    It contains the configuration for the gripper, including the retageting file name, the contact body name, and the action configuration.
    The retageting file name is the name of the file that contains the retageting data for the gripper.
    The contact body name is the name of the body that is used to contact the object.
    The action configuration is the configuration for the action that is used to control the gripper.
    """

    def __init__(self, left_retageting_file_name: str = None, right_retageting_file_name: str = None):
        """
        Args:
            left_retageting_file_name: The name of the left retageting file.
            right_retageting_file_name: The name of the right retageting file.
        """
        self.left_retageting_file_name = left_retageting_file_name
        self.right_retageting_file_name = right_retageting_file_name
        self.left_contact_body_name = None
        self.right_contact_body_name = None

    def left_hand_action_cfg(self):
        return {
            "tracking": None,
            "handle": None,
            "rl": None
        }

    def right_hand_action_cfg(self):
        return {
            "tracking": None,
            "handle": None,
            "rl": None
        }
