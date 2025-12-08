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


class ActionStateInconsistencyChecker(BaseChecker):
    type = "action_state_inconsistency"

    def __init__(self, warning_on_screen=False):
        """
        Checker to identify inconsistencies between the robot's action commands and its actual joint states.
        Supports per-joint thresholds.
        """
        super().__init__(warning_on_screen)
        self._init_state()
        self._init_joint_thresholds()
        self._frame_count = 0
        self._last_result = {"success": True, "warning_text": "", "metrics": {}}

    def _init_state(self):
        self._warning_frame_count = 0
        self._consistency_warning_text = ""
        self._consistency_violation_counts = {}
        self.name = "robot"

    def reset(self):
        self._init_state()
        self._frame_count = 0
        self._last_result = {"success": True, "warning_text": "", "metrics": {}}

    def _init_joint_thresholds(self):
        """Define per-joint thresholds: {joint_name: (minor, major)}"""
        self.joint_thresholds = {
            "body_updown_joint": (0.05, 0.1),
            "body_yaw_joint": (0.05, 0.1),
            "body_roll_joint": (0.05, 0.1),
            "body_pitch_joint": (0.15, 0.2),
            "arm_left_shoulder_pitch_joint": (0.05, 0.1),
            "arm_right_shoulder_pitch_joint": (0.05, 0.1),
            "head_pitch_joint": (1.0, 1.5),
            "arm_left_shoulder_roll_joint": (0.05, 0.1),
            "arm_right_shoulder_roll_joint": (0.05, 0.1),
            "head_roll_joint": (0.05, 0.1),
            "arm_left_shoulder_yaw_joint": (0.05, 0.1),
            "arm_right_shoulder_yaw_joint": (0.05, 0.1),
            "head_yaw_joint": (0.05, 0.1),
            "arm_left_elbow_pitch_joint": (1.65, 1.7),
            "arm_right_elbow_pitch_joint": (1.65, 1.7),
            "arm_left_wrist_yaw_joint": (1.65, 1.7),
            "arm_right_wrist_yaw_joint": (1.65, 1.7),
            "arm_left_wrist_pitch_joint": (0.05, 0.1),
            "arm_right_wrist_pitch_joint": (0.05, 0.1),
            "arm_left_wrist_roll_joint": (0.05, 0.1),
            "arm_right_wrist_roll_joint": (0.05, 0.1),
            "left_index_1_joint": (0.65, 0.7),
            "left_little_1_joint": (0.65, 0.7),
            "left_middle_1_joint": (0.65, 0.7),
            "left_ring_1_joint": (0.65, 0.7),
            "left_thumb_1_joint": (0.65, 0.7),
            "right_index_1_joint": (0.65, 0.7),
            "right_little_1_joint": (0.65, 0.7),
            "right_middle_1_joint": (0.65, 0.7),
            "right_ring_1_joint": (0.65, 0.7),
            "right_thumb_1_joint": (0.65, 0.7),
            "left_index_2_joint": (1.4, 1.5),
            "left_little_2_joint": (1.4, 1.5),
            "left_middle_2_joint": (1.4, 1.5),
            "left_ring_2_joint": (1.4, 1.5),
            "left_thumb_2_joint": (0.3, 0.4),
            "right_index_2_joint": (1.4, 1.5),
            "right_little_2_joint": (1.4, 1.5),
            "right_middle_2_joint": (1.4, 1.5),
            "right_ring_2_joint": (1.4, 1.5),
            "right_thumb_2_joint": (0.3, 0.4),
            "left_thumb_3_joint": (0.2, 0.25),
            "right_thumb_3_joint": (0.2, 0.25),
            "left_thumb_4_joint": (0.3, 0.35),
            "right_thumb_4_joint": (0.3, 0.35),
        }

    def _check(self, env):
        self._frame_count += 1
        if self._frame_count % 2 != 0:
            return self._last_result

        result = self._check_action_consistency(env)
        self._last_result = result
        return result

    def _check_action_consistency(self, env):
        self.env = env
        scene = env.scene
        robot = scene.articulations[self.name]

        if self._warning_frame_count != 0 and 50 > self._warning_frame_count > 0:
            self._warning_frame_count += 1
        else:
            self._warning_frame_count = 0
            self._consistency_warning_text = ""

        if not hasattr(env, "latest_action") or env.latest_action is None:
            return self._last_result

        actions = env.latest_action
        joint_pos = robot.data.joint_pos
        num_envs, num_joints = joint_pos.shape
        joint_names = robot.joint_names

        ignore_indices = torch.arange(13, 17)
        mask = torch.ones(num_joints, dtype=torch.bool, device=joint_pos.device)
        mask[ignore_indices] = False

        device = joint_pos.device
        if actions is not None:
            actions = actions.to(device)

        diff = torch.abs(actions - joint_pos)

        minor_envs, major_envs = [], []

        for j_idx, joint_name in enumerate(joint_names):
            if not mask[j_idx]:
                continue
            minor_th, major_th = self.joint_thresholds.get(joint_name, (0.07, 0.2))
            joint_diff = diff[:, j_idx]
            minor_idx = torch.where((joint_diff > minor_th) & (joint_diff <= major_th))[0]
            major_idx = torch.where(joint_diff > major_th)[0]

            if len(minor_idx) > 0:
                minor_envs.extend([(i.item(), joint_name) for i in minor_idx])
            if len(major_idx) > 0:
                major_envs.extend([(i.item(), joint_name) for i in major_idx])

        num_minor = len(minor_envs)
        num_major = len(major_envs)
        self._consistency_violation_counts["minor"] = self._consistency_violation_counts.get("minor", 0) + num_minor
        self._consistency_violation_counts["major"] = self._consistency_violation_counts.get("major", 0) + num_major

        warning_entries_major, warning_entries_minor = [], []

        if num_major > 0:
            env_to_joints = {}
            for env_i, joint_name in major_envs:
                env_to_joints.setdefault(env_i, []).append(joint_name)
            for env_i, joints in env_to_joints.items():
                joint_info = ", ".join([f"<<{j}>>" for j in joints[:3]])
                warning_entries_major.append(f"env_{env_i} large inconsistency in {joint_info}")

        if num_minor > 0:
            env_to_joints = {}
            for env_i, joint_name in minor_envs:
                env_to_joints.setdefault(env_i, []).append(joint_name)
            for env_i, joints in env_to_joints.items():
                joint_info = ", ".join([f"<<{j}>>" for j in joints[:3]])
                warning_entries_minor.append(f"env_{env_i} minor inconsistency in {joint_info}")

        if warning_entries_major:
            self._consistency_warning_text = (
                f"action_state_inconsistency Warning: {', '.join(warning_entries_major[:2])}"
            )
            self._warning_frame_count = 1
        elif warning_entries_minor:
            self._consistency_warning_text = (
                f"action_state_inconsistency_1 Warning: {', '.join(warning_entries_minor[:2])}"
            )
            self._warning_frame_count = 1

        metrics = self.get_consistency_metrics()
        success = (metrics.get("major", 0) == 0)
        metrics["success"] = success

        result = {
            "success": success,
            "warning_text": self._consistency_warning_text,
            "metrics": metrics,
        }

        return result

    def get_consistency_metrics(self):
        metrics = {}
        if self._consistency_violation_counts:
            for k, v in self._consistency_violation_counts.items():
                if v > 0:
                    metrics[k] = v
        return metrics
