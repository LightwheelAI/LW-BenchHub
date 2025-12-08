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


class L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray(LwTaskBase):
    task_name: str = 'L90L4StackTheRightBowlOnTheLeftBowlAndPlaceThemInTheTray'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["saladdressing"]
    removable_fixtures: list[str] = ["saladdressing"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Stack the right bowl on the left bowl and place them in the tray."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.chocolate_pudding = "chocolate_pudding"
        self.tray = "tray"
        self.bowl_left = "bowl_left"
        self.bowl_right = "bowl_right"

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.tray,
                obj_groups=self.tray,
                object_scale=0.6,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.25, 0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Tray016.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                object_scale=0.5,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(-0.25, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="ChocolatePudding001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.bowl_left,
                obj_groups="bowl",
                asset_name="Bowl008.usd",
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.5, -0.5),
                    ensure_valid_placement=True,
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.bowl_right,
                obj_groups="bowl",
                asset_name="Bowl008.usd",
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.0, -0.5),
                    ensure_valid_placement=True,
                ),
            )
        )

        return cfgs

    def _check_success(self, env):

        is_gripper_obj_far = OU.gripper_obj_far(env, self.bowl_right)
        bowl_on_bowl = OU.check_obj_in_receptacle_no_contact(env, self.bowl_right, self.bowl_left)
        bowl_on_plate = OU.check_obj_in_receptacle_no_contact(env, self.bowl_left, self.tray)
        return bowl_on_plate & bowl_on_bowl & is_gripper_obj_far
