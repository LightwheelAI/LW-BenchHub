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
import yaml


class Config:

    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            for key, value in config.items():
                setattr(self, key, value)
            self.simulation_dt = self.control_dt / self.control_decimation
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32) if "cmd_scale" in config else None

            self.max_cmd = np.array(config["max_cmd"], dtype=np.float32) if "max_cmd" in config else None
            self.cmd_debug = np.array(config["cmd_debug"], dtype=np.float32) if "cmd_debug" in config else None
            self.cmd_clip = np.array(config["cmd_clip"], dtype=np.float32) if "cmd_clip" in config else None
            # gait information
            if "gait_parameters" in config:
                self.gait_parameters = {}
                for key in config["gait_parameters"]:
                    self.gait_parameters[key] = config["gait_parameters"][key]
            else:
                self.gait_parameters = None

            self.transition_time = config.get('transition_time', 50)
