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

from lw_benchhub.core.checks.base_checker import BaseChecker
from lw_benchhub.core.models.fixtures.fixture_types import FixtureType
from lw_benchhub.utils.object_utils import check_contact


class GripperCollisionChecker(BaseChecker):
    type = "gripper_collision"

    def __init__(self, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self._init_state()

    def _init_state(self):
        self._left_gripper_collision_warning_text = ""
        self._right_gripper_collision_warning_text = ""
        self._gripper_collision_warning_frame_count = 0
        self._gripper_collision_counts = 0

    def reset(self):
        self._init_state()

    def _check(self, env):
        return self._check_collision(env)

    def _check_collision(self, env):
        """
        Check if one gripper collides with the other.
        """

        if self._gripper_collision_counts is None:
            self._gripper_collision_counts = 0

        if self._gripper_collision_warning_frame_count is not None and 50 > self._gripper_collision_warning_frame_count > 0:
            self._gripper_collision_warning_frame_count += 1
        else:
            self._gripper_collision_warning_frame_count = 0
            self._left_gripper_collision_warning_text = ""
            self._right_gripper_collision_warning_text = ""

        # Get robot body names and their collision geometries
        left_gripper = "left_gripper"
        right_gripper = "right_gripper"

        left_gripper_collision = False
        right_gripper_collision = False

        for fixture in FixtureType:
            if fixture in [FixtureType.COFFEE_MACHINE]:
                self.object = env.cfg.isaaclab_arena_env.task.get_fixture(FixtureType.COFFEE_MACHINE)

                # Handle both scalar and multi-environment tensors
                left_contact_tensor = check_contact(env, left_gripper, self.object)
                if left_contact_tensor.dim() == 0:
                    left_contact = left_contact_tensor.item()
                else:
                    left_contact = left_contact_tensor[0].item()

                if left_contact:
                    left_gripper_collision = True
                    self.collision_object_left = self.object

                right_contact_tensor = check_contact(env, right_gripper, self.object)
                if right_contact_tensor.dim() == 0:
                    right_contact = right_contact_tensor.item()
                else:
                    right_contact = right_contact_tensor[0].item()

                if right_contact:
                    right_gripper_collision = True
                    self.collision_object_right = self.object

        if left_gripper_collision and self._gripper_collision_warning_frame_count == 0:
            self._left_gripper_collision_warning_text = f"gripper_collision Warning: Collision between <<Left Gripper>> and Object <<{self.collision_object_left}>> happens"
            self._gripper_collision_counts += 1
            self._gripper_collision_warning_frame_count += 1

        if right_gripper_collision and self._gripper_collision_warning_frame_count == 0:
            self._right_gripper_collision_warning_text = f"gripper_collision Warning: Collision between <<Right Gripper>> and Object <<{self.collision_object_right}>> happens"
            self._gripper_collision_counts += 1
            self._gripper_collision_warning_frame_count += 1

        if self._gripper_collision_counts > 0:
            success = False
        else:
            success = True

        left_gripper_collision = False
        right_gripper_collision = False

        metrics = {}

        metrics["gripper_collision_times"] = self._gripper_collision_counts
        metrics["success"] = success

        result = {
            "success": success,
            "warning_text": self._left_gripper_collision_warning_text + self._right_gripper_collision_warning_text,
            "metrics": metrics
        }

        return result
