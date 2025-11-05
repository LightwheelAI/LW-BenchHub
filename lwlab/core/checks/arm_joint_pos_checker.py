from lwlab.core.checks.base_checker import BaseChecker
import torch
import numpy as np


class ArmJointAngleChecker(BaseChecker):
    type = "arm_joint_angle"

    def __init__(self, warning_on_screen=False, lower_deg=80, upper_deg=100):
        super().__init__(warning_on_screen)
        self.lower_bound = np.deg2rad(lower_deg)
        self.upper_bound = np.deg2rad(upper_deg)
        self._init_state()

    def _init_state(self):
        self._arm_joint_angle_warning_frame_count = 0
        self._arm_joint_angle_warning_text = ""
        self._arm_joint_angle_violation_counts = {}
        self.name = "robot"
        self.joint_names = [
            "arm_right_elbow_pitch_joint",
            "arm_left_elbow_pitch_joint"
        ]

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_arm_joint_angle(env)

    def _check_arm_joint_angle(self, env):
        """
        Check if arm joint angles are within (80°~100°) range
        """
        self.env = env
        scene = self.env.scene
        out_of_bound_joints = []

        for j_name in self.joint_names:
            joint_idx = scene.articulations[self.name].data.joint_names.index(j_name)
            joint_qpos = scene.articulations[self.name].data.joint_pos[:, joint_idx]

            out_of_bounds = (joint_qpos < self.lower_bound) | (joint_qpos > self.upper_bound)

            count = int(torch.sum(out_of_bounds).item())
            self._arm_joint_angle_violation_counts[j_name] = \
                self._arm_joint_angle_violation_counts.get(j_name, 0) + count

            if count > 0:
                out_of_bound_joints.append(j_name)

        if self._arm_joint_angle_warning_frame_count is not None and 50 > self._arm_joint_angle_warning_frame_count > 0:
            self._arm_joint_angle_warning_frame_count += 1
        else:
            self._arm_joint_angle_warning_frame_count = 0
            self._arm_joint_angle_warning_text = ""

        if out_of_bound_joints and self._arm_joint_angle_warning_frame_count == 0:
            if len(out_of_bound_joints) == 1:
                self._arm_joint_angle_warning_text = f"arm_joint_angle Warning: <<{out_of_bound_joints[0]}>> angle out of range"
            else:
                joined = ", ".join(out_of_bound_joints[:3])
                self._arm_joint_angle_warning_text = f"arm_joint_angle Warning: {joined} angles out of range"
            self._arm_joint_angle_warning_frame_count += 1

        success = not any(count > 0 for count in self._arm_joint_angle_violation_counts.values())

        metrics = self.get_arm_joint_angle_metrics()
        metrics["success"] = success

        return {
            "success": success,
            "warning_text": self._arm_joint_angle_warning_text,
            "metrics": metrics,
        }

    def get_arm_joint_angle_metrics(self):
        """
        Get the arm joint angle violation counts.
        """
        arm_joint_angle_metrics = {}
        if self._arm_joint_angle_violation_counts:
            for j_name, count in self._arm_joint_angle_violation_counts.items():
                if count > 0:
                    arm_joint_angle_metrics[j_name] = count
        return arm_joint_angle_metrics
