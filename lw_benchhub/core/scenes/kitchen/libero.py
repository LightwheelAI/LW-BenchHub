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

from .kitchen import LwScene


LIBRO_SCENE_ROBOT_STATE = {
    "libero-1-1": {
        "LeRobot-AbsJointGripper-RL": {
            "init_robot_base_pos": [2.43, -1.82, 0.72],
            "init_robot_base_ori": [0.00, 0.00, 0.0],
        },

    },
    "libero-2-2": {
        "LeRobot-AbsJointGripper-RL": {
            "init_robot_base_pos": [2.5, -2.0, 0.628],
            "init_robot_base_ori": [0.00, 0.00, 0.0],
        }
    },
    "libero-4-4": {
        "LeRobot-AbsJointGripper-RL": {
            "init_robot_base_pos": [2.35, -2.0, 0.668],
            "init_robot_base_ori": [0.00, 0.00, 0.0],
        }
    },
    "libero-5-5": {
        "LeRobot-AbsJointGripper-RL": {
            "init_robot_base_pos": [2.26, -2.93, -0.03],
            "init_robot_base_ori": [0.00, 0.00, 3.14],
        }
    },
    "libero-8-8": {
        "LeRobot-AbsJointGripper-RL": {
            "init_robot_base_pos": [2.43, -1.82, 0.72],
            "init_robot_base_ori": [0.00, 0.00, 0.0],
        }
    },
}


class LiberoEnvCfg(LwScene):
    """Configuration for the libero task environment."""

    def sample_robot_base(self, env, env_ids=None):
        if LIBRO_SCENE_ROBOT_STATE.get(self.scene_name, {}).get(self.robot_name, {}):
            self.init_robot_base_pos = LIBRO_SCENE_ROBOT_STATE.get(self.scene_name, {}).get(self.robot_name, {}).get("init_robot_base_pos", None)
            self.init_robot_base_ori = LIBRO_SCENE_ROBOT_STATE.get(self.scene_name, {}).get(self.robot_name, {}).get("init_robot_base_ori", None)
        else:
            super().sample_robot_base(env, env_ids)
