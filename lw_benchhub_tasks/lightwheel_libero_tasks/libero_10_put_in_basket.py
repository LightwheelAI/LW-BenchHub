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


class Libero10PutInBasket(LwTaskBase):
    """
    Libero10PutInBasket: base class for all libero 10 put in basket tasks
    """

    task_name: str = "Libero10PutInBasket"

    enable_fixtures = ['ketchup']
    removable_fixtures = enable_fixtures

    def __post_init__(self):
        self.activate_contact_sensors = False
        super().__post_init__()

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP, ref=self.counter))

        self.init_robot_base_ref = self.counter

        # Check if ketchup fixture was found
        if self.ketchup is None:
            print(f"Warning: Layout {self.layout_id} does not have a ketchup fixture. Task may not work correctly.")

        self.place_success = {}
        self.alphabet_soup = "alphabet_soup"
        self.basket = "basket"
        self.butter = "butter"
        self.cream_cheese = "cream_cheese"
        self.milk = "milk"
        self.orange_juice = "orange_juice"
        self.tomato_sauce = "tomato_sauce"

    def _get_obj_cfgs(self):
        cfgs = []

        basket_placement = dict(
            fixture=self.counter,
            size=(0.6, 0.6),
            pos=(0.7, -0.6),
            ensure_valid_placement=True,
        )
        alphabet_soup_placement = dict(
            fixture=self.counter,
            size=(0.2, 0.2),
            pos=(0.5, -0.8),
            ensure_valid_placement=True,
        )
        butter_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.7),
            ensure_valid_placement=True,
        )
        cream_cheese_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.7),
            ensure_valid_placement=True,
        )
        milk_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.7),
            ensure_valid_placement=True,
        )
        orange_juice_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.7),
            ensure_valid_placement=True,
        )
        tomato_sauce_placement = dict(
            fixture=self.counter,
            size=(0.4, 0.4),
            pos=(0.3, -0.8),
            ensure_valid_placement=True,
        )
        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                placement=alphabet_soup_placement,
                asset_name="AlphabetSoup001.usd",
                object_scale=0.8,
            )
        )
        cfgs.append(
            dict(
                name=self.basket,
                obj_groups=self.basket,
                graspable=True,
                placement=basket_placement,
                asset_name="Basket058.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=butter_placement,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups=self.cream_cheese,
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.milk,
                obj_groups=self.milk,
                graspable=True,
                placement=milk_placement,
                asset_name="MilkDrink009.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.orange_juice,
                obj_groups=self.orange_juice,
                graspable=True,
                placement=orange_juice_placement,
                asset_name="OrangeJuice001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.tomato_sauce,
                obj_groups=self.tomato_sauce,
                graspable=True,
                placement=tomato_sauce_placement,
                asset_name="Ketchup003.usd",
                object_scale=0.8,
            )
        )
        return cfgs


class L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket(Libero10PutInBasket):
    """
    L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket: put both the cream cheese box and the butter in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket
        pick up the butter
        put the butter in the basket

    """

    task_name: str = "L10L2PutBothTheCreamCheeseBoxAndTheButterInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the cream cheese box and the butter, and put them in the basket."
        return ep_meta

    def _check_success(self, env):
        success_cream_cheese = OU.check_place_obj1_on_obj2(
            env,
            self.cream_cheese,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success_butter = OU.check_place_obj1_on_obj2(
            env,
            self.butter,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        return success_cream_cheese & success_butter


class L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket(Libero10PutInBasket):
    """
    L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket: put both the alphabet soup and the tomato sauce in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket
        pick up the butter
        put the butter in the basket

    """

    task_name: str = "L10L2PutBothTheAlphabetSoupAndTheTomatoSauceInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the alphabet soup and the tomato sauce, and put them in the basket."
        return ep_meta

    def _check_success(self, env):
        success_alphabet_soup = OU.check_place_obj1_on_obj2(
            env,
            self.alphabet_soup,
            self.basket,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success_tomato_sauce = OU.check_place_obj1_on_obj2(
            env,
            self.tomato_sauce,
            self.basket,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        return success_alphabet_soup & success_tomato_sauce


class L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket(Libero10PutInBasket):
    """
    L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket: put both the alphabet soup and the cream cheese box in the basket

    Steps:
        pick up the cream cheese box
        put the cream cheese box in the basket
        pick up the butter
        put the butter in the basket

    """

    task_name: str = "L10L1PutBothTheAlphabetSoupAndTheCreamCheeseBoxInTheBasket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the alphabet soup and the cream cheese box, and put them in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = [self.butter, self.milk, self.orange_juice]
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        success_cream_cheese = OU.check_place_obj1_on_obj2(
            env,
            self.cream_cheese,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success_alphabet_soup = OU.check_place_obj1_on_obj2(
            env,
            self.alphabet_soup,
            self.basket,
            th_z_axis_cos=0.0,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        return success_cream_cheese & success_alphabet_soup
