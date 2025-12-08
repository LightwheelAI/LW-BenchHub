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

from lw_benchhub.core.checks.base_checker import BaseChecker


class StartObjectMoveChecker(BaseChecker):
    type = "start_object_move"

    def __init__(self, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self._init_state()

    def _init_state(self):
        self._start_obj_move_text = ""
        self._obj_move_info = {}
        self._pos_start = {}
        self.POS_DIFF_THRESHOLD = 0.03  # Threshold for position difference to consider as movement
        self.CHECK_STEP_NUM = 300  # currently step dt is 1/200 s

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_motion(env)

    def _check_motion(self, env):
        """
        Check if any object has moved from its initial position during the start period.
        Compares the current position of each object to its position at the start of the episode.
        If the position difference exceeds a defined threshold, it is considered as movement.
        Returns:
            dict: Dictionary containing movement violation information for each object
        """
        self.env = env

        # First frame, store initial positions of all objects
        if not self._pos_start:
            for obj in self.env.scene.articulations.keys():
                if hasattr(self.env.scene.articulations[obj], 'data') and obj != 'robot':
                    self._pos_start[obj] = self.env.scene.articulations[obj].data.body_com_pos_w[0].clone()
            for obj in self.env.scene.rigid_objects.keys():
                if hasattr(self.env.scene.rigid_objects[obj], 'data'):
                    self._pos_start[obj] = self.env.scene.rigid_objects[obj].data.body_com_pos_w[0].clone()
            return {"success": True, "warning_text": "", "metrics": {"success": True, "move_info": {}}}

        # Check for object movement within the defined step range
        if self._pos_start and self.env.common_step_counter <= self.CHECK_STEP_NUM:
            for obj in self.env.scene.articulations.keys():
                if hasattr(self.env.scene.articulations[obj], 'data') and obj != 'robot':
                    body_names = self.env.scene.articulations[obj].data.body_names
                    obj_pos = self.env.scene.articulations[obj].data.body_com_pos_w[0]
                    for i, body_name in enumerate(body_names):
                        if i < len(obj_pos) and i < len(self._pos_start[obj]):
                            start_pos = self._pos_start[obj][i]
                            current_pos = obj_pos[i]
                            pos_diff = current_pos - start_pos
                            pos_diff_norm = torch.norm(pos_diff)
                            if pos_diff_norm > self.POS_DIFF_THRESHOLD:
                                obj_key = f"{obj}_{body_name}"
                                if obj_key in self._obj_move_info:
                                    if pos_diff_norm > torch.norm(torch.tensor(self._obj_move_info[obj_key]['position_difference'])):
                                        self._obj_move_info[obj_key] = {
                                            'position_difference': pos_diff.tolist(),
                                            'current_pose': current_pos.tolist(),
                                            'start_pose': start_pos.tolist()
                                        }
                                else:
                                    self._obj_move_info[obj_key] = {
                                        'position_difference': pos_diff.tolist(),
                                        'current_pose': current_pos.tolist(),
                                        'start_pose': start_pos.tolist()
                                    }

            for obj in self.env.scene.rigid_objects.keys():
                if hasattr(self.env.scene.rigid_objects[obj], 'data'):
                    obj_pos = self.env.scene.rigid_objects[obj].data.body_com_pos_w[0]
                    for i in range(len(obj_pos)):
                        if i < len(self._pos_start[obj]):
                            start_pos = self._pos_start[obj][i]
                            current_pos = obj_pos[i]
                            pos_diff = current_pos - start_pos
                            pos_diff_norm = torch.norm(pos_diff)
                            obj_key = f"{obj}_{i}"
                            if pos_diff_norm > self.POS_DIFF_THRESHOLD:
                                if obj_key in self._obj_move_info:
                                    if pos_diff_norm > torch.norm(torch.tensor(self._obj_move_info[obj_key]['position_difference'])):
                                        self._obj_move_info[obj_key] = {
                                            'position_difference': pos_diff.tolist(),
                                            'current_pose': current_pos.tolist(),
                                            'start_pose': start_pos.tolist()
                                        }
                                else:
                                    self._obj_move_info[obj_key] = {
                                        'position_difference': pos_diff.tolist(),
                                        'current_pose': current_pos.tolist(),
                                        'start_pose': start_pos.tolist()
                                    }

        if len(self._obj_move_info) > 0:
            success = False
        else:
            success = True

        self._start_obj_move_text = f"Objects moved during start: {list(self._obj_move_info.keys())}"

        metrics = {}
        metrics["success"] = success
        metrics["move_info"] = self._obj_move_info

        result = {
            "success": success,
            "warning_text": self._start_obj_move_text,
            "metrics": metrics
        }
        return result
