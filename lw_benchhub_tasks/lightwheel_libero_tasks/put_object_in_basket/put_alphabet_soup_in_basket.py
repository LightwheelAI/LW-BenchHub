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

from .put_object_in_basket import PutObjectInBasket


class LOPickUpTheAlphabetSoupAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheAlphabetSoupAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["saladdressing"]
    removable_fixtures = enable_fixtures
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.salad_dressing = self.register_fixture_ref("salad_dressing", dict(id=FixtureType.SALAD_DRESSING))
        self.butter = "butter"
        self.cream_cheese_stick = "cream_cheese_stick"
        self.milk_drink = "milk_drink"
        self.ketchup = "ketchup"
        self.alphabet_soup = "alphabet_soup"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the alphabet soup and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    # ensure_object_boundary_in_range=False,
                ),
                asset_name="AlphabetSoup001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Butter001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups=self.ketchup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Ketchup003.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.cream_cheese_stick,
                obj_groups=self.cream_cheese_stick,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="CreamCheeseStick013.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.milk_drink,
                obj_groups=self.milk_drink,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="MilkDrink009.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the alphabetsoup is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.alphabet_soup)
        object_in_basket = OU.check_obj_in_receptacle(env, self.alphabet_soup, self.basket)
        return object_in_basket & is_gripper_obj_far
