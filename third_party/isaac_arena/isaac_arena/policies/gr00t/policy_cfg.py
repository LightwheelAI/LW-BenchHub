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

import os
from dataclasses import dataclass, field
from pathlib import Path

from isaac_arena.isaac_arena.policies.gr00t.tasks import EvalTaskConfig

# Pulling args from Gr00tN1ClosedLoopArguments into a data class
@dataclass
class GR00TN15Config():
    # model specific parameters
    task_name: str = field(
        default="nutpouring", metadata={"description": "Short name of the task to run (e.g., nutpouring, exhaustpipe)."}
    )
    task: str = field(default="", metadata={"description": "Full task name for the gym-registered environment."})
    language_instruction: str = field(
        default="", metadata={"description": "Instruction given to the policy in natural language."}
    )
    model_path: str = field(default="", metadata={"description": "Full path to the tuned model checkpoint directory."})
    action_horizon: int = field(
        default=16, metadata={"description": "Number of actions in the policy's predictionhorizon."}
    )
    embodiment_tag: str = field(
        default="new_embodiment",
        metadata={
            "description": (
                "Identifier for the robot embodiment used in the policy inference (e.g., 'gr1' or 'new_embodiment')."
            )
        },
    )
    denoising_steps: int = field(
        default=4, metadata={"description": "Number of denoising steps used in the policy inference."}
    )
    gr00t_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "gr00t_43dof_joint_space.yaml",
        metadata={"description": "Path to the YAML file specifying the joint ordering configuration for GR00T policy."},
    )
    action_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 action space in Lab."
            )
        },
    )
    state_joints_config_path: Path = field(
        default=Path(__file__).parent.resolve() / "g1" / "43dof_joint_space.yaml",
        metadata={
            "description": (
                "Path to the YAML file specifying the joint ordering configuration for GR1 state space in Lab."
            )
        },
    )
    # Closed loop specific parameters
    num_feedback_actions: int = field(
        default=16,
        metadata={
            "description": "Number of feedback actions to execute per rollout (can be less than action_horizon)."
        },
    )

    def __post_init__(self):
        # Populate fields from enum based on task_name
        if self.task_name.upper() not in EvalTaskConfig.__members__:
            raise ValueError(f"task_name must be one of: {', '.join(EvalTaskConfig.__members__.keys())}")
        config = EvalTaskConfig[self.task_name.upper()]
        if self.task == "":
            self.task = config.task

        import os
        if self.dataset_path == "":
            if self.model_path == "":
                self.model_path = config.model_path
            assert Path(self.model_path).exists(), "model_path does not exist."
            # If model path is relative, return error
            if not os.path.isabs(self.model_path):
                raise ValueError("model_path must be an absolute path. Do not use relative paths.")
        else:
            assert Path(self.dataset_path).exists(), "dataset_path does not exist."

        if self.language_instruction == "":
            self.language_instruction = config.language_instruction

        assert (
            self.num_feedback_actions <= self.action_horizon
        ), "num_feedback_actions must be less than or equal to action_horizon"
        # assert all paths exist
        assert Path(self.gr00t_joints_config_path).exists(), "gr00t_joints_config_path does not exist"
        assert Path(self.action_joints_config_path).exists(), "action_joints_config_path does not exist"
        assert Path(self.state_joints_config_path).exists(), "state_joints_config_path does not exist"
        assert Path(self.model_path).exists(), "model_path does not exist. Do not use relative paths."
        # embodiment_tag
        assert self.embodiment_tag in [
            "gr1",
            "new_embodiment",
        ], "embodiment_tag must be one of the following: " + ", ".join(["gr1", "new_embodiment"])
