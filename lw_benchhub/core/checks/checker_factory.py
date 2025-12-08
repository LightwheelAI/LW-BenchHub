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

from lw_benchhub.core.checks.action_state_inconsistency_checker import ActionStateInconsistencyChecker
from lw_benchhub.core.checks.actuator_velocity_jump_checker import VelocityJumpChecker
from lw_benchhub.core.checks.arm_joint_pos_checker import ArmJointAngleChecker
from lw_benchhub.core.checks.clipping_checker import ClippingChecker
from lw_benchhub.core.checks.gripper_collision_checker import GripperCollisionChecker
from lw_benchhub.core.checks.kitchen_coffee_collision_checker import KitchenCoffeeCollisionChecker
from lw_benchhub.core.checks.motion_checker import MotionChecker
from lw_benchhub.core.checks.obj_drop_checker import ObjDropChecker
from lw_benchhub.core.checks.start_object_move_checker import StartObjectMoveChecker

CHECKER_REGISTRY = {
    "motion": MotionChecker,
    "kitchen_coffee_collision": KitchenCoffeeCollisionChecker,
    "gripper_collision": GripperCollisionChecker,
    "clipping": ClippingChecker,
    "velocity_jump": VelocityJumpChecker,
    "start_object_move": StartObjectMoveChecker,
    "obj_drop": ObjDropChecker,
    "arm_joint_angle": ArmJointAngleChecker,
    "action_state_inconsistency": ActionStateInconsistencyChecker,
}


def get_checker(checker_type):
    if checker_type not in CHECKER_REGISTRY:
        raise ValueError(f"Checker type {checker_type} not found")
    return CHECKER_REGISTRY[checker_type]


def get_checkers_from_cfg(checkers_cfg):
    checkers = []
    for checker_type in checkers_cfg.keys():
        checker_cfg = checkers_cfg[checker_type]
        checker = get_checker(checker_type)
        checkers.append(checker(warning_on_screen=checker_cfg.get("warning_on_screen", False)))
    return checkers


def form_checker_result(checkers_cfg):
    checkers_results = {}
    for check_type in checkers_cfg.keys():
        checkers_results[check_type] = {}
    return checkers_results
