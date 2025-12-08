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


class VelocityJumpChecker(BaseChecker):
    type = "velocity_jump"

    def __init__(self, jump_threshold=0.3, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self.jump_threshold = jump_threshold
        self._init_state()

    def _init_state(self):
        self._prev_body_poses = None
        self._prev_velocities = None
        self._velocity_jump_counts = {}
        self._velocity_jump_warning_frame_count = 0
        self._velocity_jump_warning_text = ""
        self._frame_count = 0
        self._frame_counter = 0
        self._last_result = {"success": True, "warning_text": "", "metrics": {}}

    def reset(self):
        self._init_state()

    def _check(self, env):
        self._frame_counter += 1
        if self._frame_counter % 2 != 0:
            return self._last_result

        result = self._check_velocity_jump(env)
        self._last_result = result
        return result

    def _check_velocity_jump(self, env):
        """
        Check if velocity jump happens for bodies containing 'arm' in their name.
        """
        self.env = env
        self._frame_count += 1
        self._jump_violations = {}

        robot = env.scene.articulations['robot']
        current_body_poses = robot.data.body_com_pose_w[0]
        body_names = robot.data.body_names

        if not hasattr(self, "_arm_indices") or not self._arm_indices:
            self._arm_indices = [i for i, name in enumerate(body_names) if "arm" in name.lower()]
            if not self._arm_indices:
                return {"success": True, "warning_text": "", "metrics": {}}

        arm_indices = self._arm_indices

        if self._prev_body_poses is None:
            self._prev_body_poses = current_body_poses.clone()
            return {"success": True, "warning_text": "", "metrics": {}}

        prev_pos = self._prev_body_poses[arm_indices, :3]
        curr_pos = current_body_poses[arm_indices, :3]

        pos_diff = curr_pos - prev_pos
        velocities = torch.norm(pos_diff, dim=1) / (2 * self.env.step_dt)

        if self._prev_velocities is None:
            self._prev_velocities = velocities.clone()
            self._prev_body_poses = current_body_poses.clone()
            return {"success": True, "warning_text": "", "metrics": {}}

        v_diff = torch.abs(velocities - self._prev_velocities)
        over_threshold = v_diff > self.jump_threshold

        if over_threshold.any():
            for i, flag in enumerate(over_threshold):
                if flag:
                    body_name = body_names[arm_indices[i]]
                    self._jump_violations[body_name] = {
                        "velocity_jump": v_diff[i].item(),
                        "prev_velocity": self._prev_velocities[i].item(),
                        "curr_velocity": velocities[i].item()
                    }
                    self._velocity_jump_counts[body_name] = self._velocity_jump_counts.get(body_name, 0) + 1

        if self._frame_count > 60:
            fast_bodies = list(self._jump_violations.keys())[:3]
            if fast_bodies and self._velocity_jump_warning_frame_count == 0:
                joined = ", ".join(f"<<{b}>>" for b in fast_bodies)
                self._velocity_jump_warning_text = f"velocity_jump Warning: Bodies {joined} velocity jump too large"
                self._velocity_jump_warning_frame_count = 1
            elif self._velocity_jump_warning_frame_count > 0:
                self._velocity_jump_warning_frame_count += 1
                if self._velocity_jump_warning_frame_count > 50:
                    self._velocity_jump_warning_text = ""
                    self._velocity_jump_warning_frame_count = 0

        self._prev_velocities = velocities.clone()
        self._prev_body_poses = current_body_poses.clone()

        success = len(self._velocity_jump_counts) == 0
        metrics = self.get_velocity_jump_metrics()
        metrics["success"] = success

        return {
            "success": success,
            "warning_text": self._velocity_jump_warning_text,
            "metrics": metrics
        }

    def get_velocity_jump_metrics(self):
        """Calculate the metrics for velocity jump."""
        metrics = {}
        if self._velocity_jump_counts:
            robot_metrics = {k: v for k, v in self._velocity_jump_counts.items() if v > 0}
            if robot_metrics:
                metrics["robot"] = robot_metrics
        return metrics
