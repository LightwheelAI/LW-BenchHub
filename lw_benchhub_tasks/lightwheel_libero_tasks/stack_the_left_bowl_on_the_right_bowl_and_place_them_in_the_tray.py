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


class L90L4StackTheLeftBowlOnTheRightBowlAndPlaceThemInTheTray(LwTaskBase):
    task_name: str = "L90L4StackTheLeftBowlOnTheRightBowlAndPlaceThemInTheTray"
    enable_fixtures = ["salad_dressing"]
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.6)),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"stack the left bowl on the right bowl and place them in the tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"wooden_tray",
                obj_groups=["tray"],
                graspable=True,
                washable=True,
                asset_name="Tray016.usd",
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.6, 0.6),
                    rotation=np.pi / 2,
                    margin=0.02,
                    pos=(-0.5, -0.6)
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    margin=0.02,
                    pos=(-0.1, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl_right",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    margin=0.02,
                    pos=(0.4, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="chocolate_pudding",
                obj_groups="chocolate_pudding",
                graspable=True,
                asset_name="ChocolatePudding001.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.80, 0.50),
                    pos=(0.2, 0.2),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        ret1 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl", "akita_black_bowl_right")
        ret2 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl_right", "wooden_tray")
        return ret1 & ret2
