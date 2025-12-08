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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class OrganizeVegetables(LwTaskBase):
    """
    Organize Vegetables: composite task for Chopping Food activity.

    Simulates the task of organizing vegetables on cutting boards.

    Steps:
        Place the vegetables on separate cutting boards
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.DINING_COUNTER]
    task_name: str = "OrganizeVegetables"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1.0, 0.4))
        )

        # self.counter = self.get_fixture(FixtureType.DINING_COUNTER, ref=self.sink)
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("vegetable1")
        obj_name_2 = self.get_obj_lang("vegetable2")

        ep_meta[
            "lang"
        ] = f"Place the {obj_name_1} and the {obj_name_2} on separate cutting boards."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board1",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.3, 0.3),
                    rotation=torch.pi / 2,
                    pos=(-0.6, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="cutting_board2",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.3, 0.3),
                    rotation=0,
                    pos=(0.5, -0.4),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable1",
                obj_groups=["vegetable", "fruit"],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.40, 0.40),
                    pos=(0, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable2",
                obj_groups=["vegetable", "fruit"],
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(0.40, 0.40),
                    pos=(0, -0.5),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Make sure vegetables are on different cutting boards
        """
        vegetable1_cutting_board_contact1 = OU.check_obj_in_receptacle(
            env, "vegetable1", "cutting_board1"
        )
        vegetable2_cutting_board_contact1 = OU.check_obj_in_receptacle(
            env, "vegetable2", "cutting_board1"
        )
        vegetable1_cutting_board_contact2 = OU.check_obj_in_receptacle(
            env, "vegetable1", "cutting_board2"
        )
        vegetable2_cutting_board_contact2 = OU.check_obj_in_receptacle(
            env, "vegetable2", "cutting_board2"
        )

        gipper_success = OU.gripper_obj_far(env, "vegetable1") & OU.gripper_obj_far(env, "vegetable2")

        return ((
            vegetable1_cutting_board_contact1 & vegetable2_cutting_board_contact2
        ) | (vegetable2_cutting_board_contact1 & vegetable1_cutting_board_contact2)) & gipper_success
