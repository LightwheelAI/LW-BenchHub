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

import isaaclab.sim as sim_utils

from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase
from lw_benchhub.utils.env import ExecuteMode

##
# Scene definition
##
# Increase PhysX GPU aggregate pairs capacity to avoid simulation errors
sim_utils.simulation_context.gpu_total_aggregate_pairs_capacity = 160000


class LiftObj(LwTaskBase):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """
    counter_id: FixtureType = FixtureType.COUNTER
    task_name: str = "LiftObj"

    def __init__(self):
        super().__init__()
        self.fix_object_pose_cfg: dict = {"object": {"pos": (2.94, -4.08, 0.95)}}  # y- near to robot
        self.resample_robot_placement_on_reset = False

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references(scene)

        self.counter = self.register_fixture_ref("counter", dict(id=self.counter_id, fix_id=2))
        # self.useful_fixture_names = [self.counter.name]
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):

        cfgs = []

        cfgs.append(
            dict(
                name="object",
                obj_groups="cube",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.20, 0.20),
                    pos=(0, -0.60),
                    offset=(-0.75, 0.25)
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the cube is lifted.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        if self.context.execute_mode == ExecuteMode.TRAIN:
            return torch.tensor([False], device=env.device).repeat(env.num_envs)

        object_height = env.scene['object'].data.root_pos_w[:, 2]
        is_height_sufficient = (object_height >= 0.965)

        success = is_height_sufficient
        return success
