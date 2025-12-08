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

from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class PutBlackBowlOnPlate(LwTaskBase):
    task_name: str = f"PutBlackBowlOnPlate"
    enable_fixtures: list[str] = ["storage_furniture", "stovetop"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.storage_furniture = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE))
        self.init_robot_base_ref = self.dining_table
        self.plate = "plate"
        self.ramekin = "ramekin"
        self.cookies = "cookies"
        # akita black bowl
        self.bowl_asset_name = "Bowl008.usd"
        self.bowl_placement = {
            "between_plate_and_ramekin": dict(
                fixture=self.dining_table,
                size=(0.25, 0.25),
                pos=(0.4, -0.4),
                ensure_valid_placement=True,
            ),
            'center': dict(
                fixture=self.dining_table,
                size=(0.25, 0.25),
                pos=(0.1, -0.3),
                ensure_valid_placement=True,
            ),
            'near_ramekin': dict(
                fixture=self.dining_table,
                size=(0.25, 0.25),
                pos=(0.3, -0.8),
                ensure_valid_placement=True,
            ),
            'near_cookies': dict(
                fixture=self.dining_table,
                size=(0.25, 0.25),
                pos=(1.0, -0.5),
                ensure_valid_placement=True,
            ),
            'near_plate': dict(
                fixture=self.dining_table,
                size=(0.25, 0.25),
                pos=(0.65, -0.1),
                ensure_valid_placement=True,
            ),
        }

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.top_joint_name = list(self.storage_furniture._joint_infos.keys())[0]
        self.storage_furniture.set_joint_state(0.1, 0.2, env, [self.top_joint_name])

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.cookies,
                obj_groups="cookies",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(1, 1),
                    ensure_valid_placement=True,
                ),
                asset_name="Cookies002.usd",
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
                    size=(0.4, 0.4),
                    pos=(1, -0.1),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Plate012.usd",
            )
        )
        # ramekin
        cfgs.append(
            dict(
                name=self.ramekin,
                obj_groups="ramekin",
                graspable=True,
                object_scale=0.5,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.4),
                    pos=(1, 0),
                    ensure_valid_placement=True,
                ),
                asset_name="Bowl009.usd",
            )
        )

        return cfgs
