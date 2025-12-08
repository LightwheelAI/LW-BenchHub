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


class L90L3PickUpTheCreamCheeseAndPutItInTheTray(LwTaskBase):
    task_name: str = "L90L3PickUpTheCreamCheeseAndPutItInTheTray"
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.TABLE, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the cream cheese and put it in the tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cream_cheese",
                obj_groups="cream_cheese_stick",
                init_robot_here=True,
                graspable=True,
                asset_name="CreamCheeseStick013.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.3),
                    pos=(0.0, -0.9),
                    ensure_valid_placement=True,
                ),
            )
        )

        cfgs.append(
            dict(
                name="wooden_tray",
                obj_groups="tray",
                graspable=True,
                object_scale=0.8,
                asset_name="Tray016.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.7),
                    pos=(-0.8, 0.0),
                    rotation=np.pi / 2,
                ),
            )
        )
        cfgs.append(
            dict(
                name="butter",
                obj_groups="butter",
                graspable=True,
                asset_name="Butter001.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.3),
                    pos=(0.1, -0.2),
                    ensure_valid_placement=True,
                ),
            )
        )
        cfgs.append(
            dict(
                name="alphabet_soup",
                obj_groups="alphabet_soup",
                graspable=True,
                asset_name="AlphabetSoup001.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.3),
                    pos=(0.25, -0.3),
                    ensure_valid_placement=True,
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        return OU.check_obj_in_receptacle(env, "cream_cheese", "wooden_tray") & OU.gripper_obj_far(env, "cream_cheese")
