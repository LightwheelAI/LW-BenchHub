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


class L90L2PickUpTheMilkAndPutItInTheBasket(LwTaskBase):
    task_name: str = f"L90L2PickUpTheMilkAndPutItInTheBasket"
    enable_fixtures: list[str] = ["ketchup"]
    removable_fixtures = enable_fixtures
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.basket = "basket"
        self.alphabet_soup = "alphabet_soup"
        self.orange_juice = "orange_juice"
        self.cream_cheese_stick = "cream_cheese_stick"
        self.ketchup = "ketchup"
        self.milk_drink = "milk_drink"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the milk and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.basket,
                obj_groups=self.basket,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.5, -0.4),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Basket058.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.milk_drink,
                obj_groups=self.milk_drink,
                object_scale=0.8,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.5, -0.7),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="MilkDrink009.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.75, 0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="AlphabetSoup001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups=self.ketchup,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.25, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Ketchup003.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.orange_juice,
                obj_groups=self.orange_juice,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.8, -0.6),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="OrangeJuice001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.cream_cheese_stick,
                obj_groups=self.cream_cheese_stick,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.8, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the milk is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.milk_drink)
        object_in_basket = OU.check_obj_in_receptacle(env, self.milk_drink, self.basket)
        return object_in_basket & is_gripper_obj_far
