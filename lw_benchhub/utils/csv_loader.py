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

import ast
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from lw_benchhub import CONFIGS_PATH


class CSVLoader:
    def __init__(self):
        self.configs_root = CONFIGS_PATH
        self.csv_data = []
        self._collect_csv_files()
        self.robot_pos = None
        self.robot_ori = None
        self.obj_offset = np.zeros(2,)

    def _collect_csv_files(self):
        for csv_path in self.configs_root.rglob("*.csv"):
            self.csv_data.append(pd.read_csv(csv_path.resolve()))

        entry_points = importlib.metadata.entry_points()
        lw_benchhub_modules = entry_points.select(group="lw_benchhub_modules")
        for entry_point in lw_benchhub_modules:
            if entry_point.name == "config_search_path":
                additional_config_path = entry_point.load().__path__[0]
                for csv_path in Path(additional_config_path).rglob("*.csv"):
                    self.csv_data.append(pd.read_csv(csv_path.resolve()))
                break

    def load_robot_pose(self, robot_name=None, scene_name=None, task_name=None):
        if robot_name is not None and scene_name is not None and task_name is not None:
            for data in self.csv_data:
                row = data[(data['robot'] == robot_name) &
                           (data['layout'] == scene_name) &
                           (data['task'] == task_name)]
                if not row.empty:
                    self.robot_pos = ast.literal_eval(row.iloc[0]['init_robot_base_pos'])
                    self.robot_ori = ast.literal_eval(row.iloc[0]['init_robot_base_ori'])
                    print(
                        f"[CSV Match] {robot_name} | {scene_name} | {task_name} | "
                        f"Init Pos:{self.robot_pos} - Init Ori:{self.robot_ori}"
                    )
                    return self.robot_pos, self.robot_ori
        return None, None

    def load_object_offset(self, robot_name=None, scene_name=None, task_name=None):
        if robot_name is not None and scene_name is not None and task_name is not None:
            for data in self.csv_data:
                row = data[(data['robot'] == robot_name) &
                           (data['layout'] == scene_name) &
                           (data['task'] == task_name)]
                if not row.empty:
                    self.obj_offset = ast.literal_eval(row.iloc[0]['object_init_offset'])
                    print(
                        f"[CSV Match] Init Obj Offset:{self.obj_offset}"
                    )
                    return self.obj_offset
        return np.zeros(2,)


csv_loader = CSVLoader()
