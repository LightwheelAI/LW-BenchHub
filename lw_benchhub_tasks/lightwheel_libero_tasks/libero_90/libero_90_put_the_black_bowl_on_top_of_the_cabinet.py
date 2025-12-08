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
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class L90K1PutTheBlackBowlOnTopOfTheCabinet(LwTaskBase):
    task_name: str = 'L90K1PutTheBlackBowlOnTopOfTheCabinet'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["storage_furniture"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the black bowl on top of the cabinet."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.storage_furniture = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE))
        self.init_robot_base_ref = self.dining_table
        self.plate = "plate"
        self.bowl = "bowl"

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.0, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Bowl008.usd",
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
                    size=(0.25, 0.25),
                    pos=(0.5, -0.15),
                    ensure_valid_placement=True,
                ),
                asset_name="Plate012.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(env, self.bowl, self.storage_furniture)
