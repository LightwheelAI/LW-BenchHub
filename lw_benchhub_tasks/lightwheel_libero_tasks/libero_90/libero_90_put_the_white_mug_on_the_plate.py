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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.tasks.base import LwTaskBase
from lw_benchhub.core.models.fixtures import FixtureType


class L90L6PutTheWhiteMugOnThePlate(LwTaskBase):
    task_name: str = 'L90L6PutTheWhiteMugOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the white mug on the plate."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.chocolate_pudding = "chocolate_pudding"
        self.plate = "plate"
        self.porcelain_mug = "porcelain_mug"
        self.red_coffee_mug = "red_coffee_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                object_scale=0.5,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(0.5, -0.6),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="ChocolatePudding001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(0.25, -0.8),
                    ensure_valid_placement=True,
                ),
                asset_name="Plate012.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups=self.porcelain_mug,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, -0.8),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Cup012.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups=self.red_coffee_mug,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.5, -0.8),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Cup030.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the bowl is placed on the plate.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.porcelain_mug)
        object_on_plate = OU.check_obj_in_receptacle(env, self.porcelain_mug, self.plate)
        return object_on_plate & is_gripper_obj_far
