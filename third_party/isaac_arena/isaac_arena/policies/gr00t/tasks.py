# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

class EvalTaskConfig(Enum):
    NUTPOURING = (
        "Isaac-NutPour-GR1T2-ClosedLoop-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Nut-Pouring-task",
        (
            "Pick up the beaker and tilt it to pour out 1 metallic nut into the bowl. Pick up the bowl and place it on"
            " the metallic measuring scale."
        ),
        "nut_pouring_task.hdf5",
        0   # 1 is reserved for data validity check, following GR00T-N1 guidelines.
    )
    PIPESORTING = (
        "Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Exhaust-Pipe-Sorting-task",
        "Pick up the blue pipe and place it into the blue bin.",
        "exhaust_pipe_sorting_task.hdf5",
        2   # 1 is reserved for data validity check, following GR00T-N1 guidelines.
    )
    SQUATPNPBOX = (
        "Isaac-Squatting-PickPlace-Box-G1-Abs-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Squatting-Pnp-Box-task",
        "Squat down, pick up the box and stand up.",
        "squatting_pickplace_box_mimic_generated_500_v5_action_noise_005.hdf5",
        3
    )
    WALKPNPBOX = (
        "Isaac-Navigate-PickPlace-G1-Scene-2-Abs-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Walk-Pnp-Box-task",
        "Walk toward the table, pick up the box from the table.",
        "navigate_pickplace_scene_2_mimic_generated_500_v6_action_noise_003.hdf5",
        4
    )
    WALKSQUATPNPBOX = (
        "Isaac-Navigate-Squat-Pick-G1-Abs-v0",
        "/home/gr00t/GR00T-N1-2B-tuned-Walk-Squat-Pnp-Box-task",
        "Walk toward the table, squat down, pick up the box from the table and stand up.",
        "navigate_squat_pick_mimic_generated_500_v1_action_noise_003.hdf5",
        5
    )

    def __init__(self, task: str, model_path: str, language_instruction: str, hdf5_name: str, task_index: int):
        self.task = task
        self.model_path = model_path
        self.language_instruction = language_instruction
        self.hdf5_name = hdf5_name
        assert task_index != 1, "task_index must not be 1. (Use 0 for nutpouring, 2 for exhaustpipe, etc.)"
        self.task_index = task_index