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

from collections.abc import Callable

from isaaclab.devices.device_base import DeviceBase

from .so101_leader import SO101Leader


class BiSO101Leader(DeviceBase):
    def __init__(self, env, left_port: str = '/dev/ttyACM0', right_port: str = '/dev/ttyACM1', recalibrate: bool = False):
        super().__init__(env)
        self.env = env

        # use left so101 leader as the main device to store state
        print("Connecting to left_so101_leader...")
        self.left_so101_leader = SO101Leader(env, left_port, recalibrate, "left_so101_leader.json")
        print("Connecting to right_so101_leader...")
        self.right_so101_leader = SO101Leader(env, right_port, recalibrate, "right_so101_leader.json")

        self.right_so101_leader.listener.stop()

    def __str__(self) -> str:
        """Returns: A string containing the information of bi-so101 leader."""
        msg = "Bi-SO101-Leader device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove Bi-SO101-Leader to control Bi-SO101-Follower\n"
        msg += "\tIf SO101-Follower can't synchronize with Bi-SO101-Leader, please add --recalibrate and rerun to recalibrate Bi-SO101-Leader.\n"
        return msg

    def add_callback(self, key: str, func: Callable):
        self.left_so101_leader.add_callback(key, func)
        self.right_so101_leader.add_callback(key, lambda: None)

    def reset(self):
        self.left_so101_leader.reset()
        self.right_so101_leader.reset()

    def _get_raw_data(self):
        return {
            "left_arm": self.left_so101_leader._get_raw_data(),
            "right_arm": self.right_so101_leader._get_raw_data()
        }

    def advance(self):
        action_dict = dict()
        action_dict['joint_state'] = self._get_raw_data()
        action_dict['motor_limits'] = {
            'left_arm': self.left_so101_leader.motor_limits,
            'right_arm': self.right_so101_leader.motor_limits
        }
        action_dict['bi_so101_leader'] = True
        return self.env.cfg.isaaclab_arena_env.embodiment.preprocess_device_action(action_dict, self)
