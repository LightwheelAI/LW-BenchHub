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


class MotionChecker(BaseChecker):
    type = "motion"

    def __init__(self, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self._init_state()

    def _init_state(self):
        self._motion_warning_frame_count = 0
        self._motion_warning_text = ""
        self._motion_violation_counts = {}
        self._prev_body_poses = None

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_motion(env)

    def _check_motion(self, env):
        """
        Check if any robot body has excessive motion between frames.
        Calculates velocity by dividing position difference by physics_dt.
        Threshold is set to 1.

        Returns:
            dict: Dictionary containing motion violation information for each body
        """
        self.env = env
        motion_violations = {}

        # Get current frame robot body poses and names
        current_body_poses = self.env.scene.articulations['robot'].data.body_com_pose_w[0]
        body_names = self.env.scene.articulations['robot'].data.body_names

        # Check if we have previous frame data
        if self._prev_body_poses is None:
            # First frame, just store current poses for next frame comparison
            self._prev_body_poses = current_body_poses.clone()
            return motion_violations

        # Skip first 10 frames from motion calculation
        if self.env.common_step_counter <= 10:
            # Store current poses for next frame comparison
            self._prev_body_poses = current_body_poses.clone()
            return motion_violations

        # Initialize motion violation counts if not exists
        if self._motion_violation_counts is None:
            self._motion_violation_counts = {}

        # Check if we need to clear old warnings (after 50 frames)
        if self._motion_warning_frame_count is not None and 50 > self._motion_warning_frame_count > 0:
            self._motion_warning_frame_count += 1
        else:
            self._motion_warning_frame_count = 0
            self._motion_warning_text = ""

        # Calculate velocity for each body
        fast_bodies = []  # Track fast body names for UI display

        for i, body_name in enumerate(body_names):
            if i < len(current_body_poses) and i < len(self._prev_body_poses):
                current_pose = current_body_poses[i]
                prev_pose = self._prev_body_poses[i]

                # Calculate position difference (assuming first 3 elements are x, y, z)
                if len(current_pose) >= 3 and len(prev_pose) >= 3:
                    pos_diff = current_pose[:3] - prev_pose[:3]
                    pos_diff_norm = torch.norm(pos_diff)

                    # Calculate velocity magnitude
                    velocity_magnitude = pos_diff_norm / self.env.step_dt

                    # Check if velocity exceeds threshold
                    if velocity_magnitude > 1.0:
                        motion_violations[body_name] = {
                            'velocity_magnitude': velocity_magnitude.item(),
                            'position_difference': pos_diff.tolist(),
                            'current_pose': current_pose.tolist(),
                            'previous_pose': prev_pose.tolist()
                        }

                        # Count motion violations for metrics
                        if body_name not in self._motion_violation_counts:
                            self._motion_violation_counts[body_name] = 0
                        self._motion_violation_counts[body_name] += 1

                        # Add to fast bodies list for UI (max 3)
                        if len(fast_bodies) < 3:
                            fast_bodies.append(body_name)

        # Create simplified warning text for UI (only body names)
        if fast_bodies and self._motion_warning_frame_count == 0:
            if len(fast_bodies) == 1:
                self._motion_warning_text = f"motion Warning: Body <<{fast_bodies[0]}>> too fast"
            elif len(fast_bodies) == 2:
                self._motion_warning_text = f"motion Warning: Bodies <<{fast_bodies[0]}>>, <<{fast_bodies[1]}>> too fast"
            else:
                self._motion_warning_text = f"motion Warning: Bodies <<{fast_bodies[0]}>>, <<{fast_bodies[1]}>>, <<{fast_bodies[2]}>> too fast"
            self._motion_warning_frame_count = 1

        # Store current poses for next frame comparison
        self._prev_body_poses = current_body_poses.clone()

        if len(motion_violations) > 0:
            success = False
        else:
            success = True

        metrics = self.get_motion_metrics()

        metrics["success"] = success

        result = {
            "success": success,
            "warning_text": self._motion_warning_text,
            "metrics": metrics
        }
        return result

    def get_motion_metrics(self):
        """
        Get formatted motion metrics data for JSON export.

        Returns:
            dict: Formatted motion metrics data
        """
        motion_metrics = {}

        if self._motion_violation_counts:
            robot_metrics = {}

            # Process motion violation counts for each body
            for body_name, count in self._motion_violation_counts.items():
                if count > 0:  # Only record bodies that had violations
                    robot_metrics[body_name] = count

            if robot_metrics:  # Only add if there are violations
                motion_metrics['robot'] = robot_metrics

        return motion_metrics
