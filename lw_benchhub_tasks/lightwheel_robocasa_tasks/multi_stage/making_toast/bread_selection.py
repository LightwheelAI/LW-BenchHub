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

import numpy as np

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class BreadSelection(LwTaskBase):
    """
    Bread Selection: composite task for Making Toast activity.

    Simulates the task of setting up ingredients for making a bread snack.

    Steps:
        Place a croissant and a jar of jam on the cutting board.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to specify the cabinet where the jam is placed.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "BreadSelection"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.6))
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "From the different types of pastries on the counter, select a croissant and place it on the cutting board. "
            "Then retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board."
        )

        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    rotation=[np.pi / 2, np.pi / 2],
                ),
            )
        )

        # Three types of pastries
        cfgs.append(
            dict(
                name="distr_pastry",
                obj_groups=("baguette", "cupcake"),
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="croissant",
                obj_groups="croissant",
                placement=dict(
                    fixture=self.counter,
                    reuse_region_from="cutting_board",
                    size=(1.0, 0.6),
                    pos=(0.0, -1.0),
                    try_to_place_in="plate",
                ),
            )
        )

        # Jar of jam in the cabinet
        cfgs.append(
            dict(
                name="jam",
                obj_groups="jam",
                placement=dict(fixture=self.cab, size=(1.0, 0.10), pos=(0, -1.0)),
            )
        )

        # Additional distractor on the counter
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(1.0, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        # Check that both the croissant and jam are on the cutting board
        return (
            OU.check_obj_in_receptacle(env, "croissant", "cutting_board")
            & OU.gripper_obj_far(env, obj_name="croissant")
            & OU.check_obj_in_receptacle(env, "jam", "cutting_board")
        )
