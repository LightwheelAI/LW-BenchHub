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


class L90S3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddy(LwTaskBase):
    task_name: str = 'L90S3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddy'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the white mug and place it to the right compartment of the caddy."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.desk_caddy = "desk_caddy"
        self.book = "book"
        self.white_mug = "white_mug"
        self.red_mug = "red_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.desk_caddy,
                obj_groups=self.desk_caddy,
                graspable=True,
                object_scale=2.0,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.6, 0.6),
                    pos=(0.0, -0.3),
                    rotation=0.0,
                ),
                asset_name="DeskCaddy001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.book,
                obj_groups="book",
                graspable=True,
                object_scale=0.4,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.5, -0.5),
                    ensure_valid_placement=True,
                ),
                asset_name="Book042.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.white_mug,
                obj_groups="mug",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.2, -0.5),
                    rotation=np.pi / 4,
                    ensure_valid_placement=True,
                ),
                asset_name="Cup012.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.red_mug,
                obj_groups="mug",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(0.0, -0.5),
                    ensure_valid_placement=True,
                ),
                asset_name="Cup030.usd",
            )
        )
        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_side_by_obj2(
            env,
            self.white_mug,
            self.desk_caddy,
            {
                "gripper_far": True,
                "contact": False,
                "side": "right",
                "side_threshold": 1.0,
                "margin_threshold": [0.001, 0.2],
                "stable_threshold": 0.5,
            }
        )
