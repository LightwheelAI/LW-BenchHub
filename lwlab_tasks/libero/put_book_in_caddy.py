import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import copy
from isaaclab.utils.math import matrix_from_quat
from time import time

import numpy as np


class L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy(LwLabTaskBase):
    task_name: str = "L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the book and place it in the right compartment of the caddy."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        placement = dict(
            fixture=self.counter,
            size=(0.8, 0.4),
            pos=(0.0, -0.6),
            offset=(-0.05, 0),
            ensure_valid_placement=True,
        )
        cfgs.append(
            dict(
                name="desk_caddy",
                obj_groups="desk_caddy",
                graspable=True,
                object_scale=2.0,
                init_robot_here=True,
                asset_name="DeskCaddy001.usd",
                placement=placement,
            )
        )
        cfgs.append(
            dict(
                name="black_book",
                object_scale=0.4,
                obj_groups="book",
                graspable=True,
                asset_name="Book042.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.4, 0.3),
                    pos=(0.2, -0.9),
                    offset=(-0.05, 0),
                    ensure_valid_placement=True,
                ),
            )
        )
        cfgs.append(
            dict(
                name="porcelain_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup012.usd",
                placement=placement,
            )
        )
        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup030.usd",
                placement=placement,
            )
        )
        return cfgs

    def _check_success(self, env):
        book_success = OU.check_obj_in_receptacle(env, "black_book", "desk_caddy")
        gipper_far_success = OU.gripper_obj_far(env, "black_book", 0.35)
        return book_success & gipper_far_success


class L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy(L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy):

    task_name: str = "L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the book and place it in the right compartment of the caddy."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="black_book",
                obj_groups="book",
                object_scale=0.4,
                graspable=True,
                asset_name="Book042.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.3, -0.8)
                ),
            )
        )
        cfgs.append(
            dict(
                name="desk_caddy",
                obj_groups="desk_caddy",
                graspable=True,
                object_scale=2.0,
                init_robot_here=True,
                asset_name="DeskCaddy001.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.8, 0.45),
                    margin=0.02,
                    pos=(0.6, -0.5)
                ),
            )
        )
        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup030.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.1)
                ),
            )
        )
        return cfgs


class L90S3PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy(L90S2PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy):

    task_name: str = "L90S3PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the book and place it in the left compartment of the caddy."
        return ep_meta


class _BaseBookInCaddy(LwLabTaskBase):
    task_name: str = "_BaseBookInCaddy"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        self.black_book = "black_book"
        self.desk_caddy = "desk_caddy"
        self.red_coffee_mug = "red_coffee_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        caddy_pl = dict(
            fixture=self.counter,
            size=(0.6, 0.4),
            pos=(0, -0.4),
            rotation=np.pi / 8,
            margin=0.02,
            ensure_valid_placement=True,
        )
        book_pl = dict(
            fixture=self.counter,
            size=(0.35, 0.35),
            pos=(0.2, -0.5),
            rotation=0,
            margin=0.02,
            ensure_valid_placement=True,
        )
        mug_pl = dict(
            fixture=self.counter,
            size=(0.35, 0.3),
            pos=(-0.3, -0.6),
            rotation=0,
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.desk_caddy,
                obj_groups="desk_caddy",
                graspable=True,
                placement=caddy_pl,
                asset_name="DeskCaddy001.usd",
                object_scale=2.0,
            )
        )
        cfgs.append(
            dict(
                name=self.black_book,
                obj_groups="book",
                graspable=True,
                placement=book_pl,
                asset_name="Book042.usd",
                object_scale=0.4,
            )
        )
        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups="cup",
                graspable=True,
                placement=mug_pl,
                asset_name="Cup030.usd",
            )
        )

        return cfgs

    def _success_common(self, env):
        in_caddy = OU.check_obj_in_receptacle(env, self.black_book, self.desk_caddy)
        gripper_far_success = OU.gripper_obj_far(env, self.black_book, 0.35)
        return in_caddy & gripper_far_success


class L90S2PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy(_BaseBookInCaddy):
    task_name: str = "L90S2PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book and place it in the left compartment of the caddy."
        return ep_meta

    def _check_success(self, env):
        # TODO: add compartment-level check (left half of caddy) when utility is available
        return self._success_common(env)


class L90S2PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy(_BaseBookInCaddy):
    task_name: str = "L90S2PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book and place it in the back compartment of the caddy."
        return ep_meta

    def _check_success(self, env):
        # TODO: add compartment-level check (back half of caddy) when utility is available
        return self._success_common(env)


class L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy(L90S3PickUpTheBookAndPlaceItInTheRightCompartmentOfTheCaddy):

    task_name: str = "L10S1PickUpTheBookAndPlaceItInTheBackCompartmentOfTheCaddy"

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="black_book",
                obj_groups="book",
                object_scale=0.4,
                graspable=True,
                asset_name="Book042.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.3, -0.5)
                ),
            )
        )
        cfgs.append(
            dict(
                name="desk_caddy",
                obj_groups="desk_caddy",
                graspable=True,
                object_scale=2.0,
                init_robot_here=True,
                asset_name="DeskCaddy001.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.8, 0.45),
                    margin=0.02,
                    pos=(0.3, -0.5)
                ),
            )
        )
        cfgs.append(
            dict(
                name="white_yellow_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup014.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs


# --- Study Scene 4 Book Tasks ---

class _BaseStudyScene4(LwLabTaskBase):
    """Base class for Study Scene 4 tasks with books and shelves"""
    task_name: str = "_BaseStudyScene4"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        self.black_book = "black_book"
        self.yellow_book = "yellow_book"
        self.middle_book = "midlle_book"
        self.shelf = "shelf"

    def _get_obj_cfgs(self):
        cfgs = []

        left_book_pl = dict(
            fixture=self.counter,
            size=(0.35, 0.35),
            pos=(-0.35, 0.8),
            margin=0.02,
            ensure_valid_placement=True,
        )
        right_book_pl = dict(
            fixture=self.counter,
            size=(0.35, 0.35),
            pos=(0.35, -0.6),
            margin=0.02,
            ensure_valid_placement=True,
        )
        middle_book_pl = dict(
            fixture=self.counter,
            size=(0.35, 0.35),
            pos=(0, -0.6),
            margin=0.02,
            ensure_valid_placement=True,
        )
        shelf_pl = dict(
            fixture=self.counter,
            size=(0.5, 0.75),
            pos=(0.7, -0.5),
            rotation=-np.pi / 2,
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.shelf,
                obj_groups="shelf",
                graspable=True,
                placement=shelf_pl,
                asset_name="Shelf073.usd",
                object_scale=1.0,
            )
        )
        cfgs.append(
            dict(
                name=self.black_book,
                obj_groups="book",
                graspable=True,
                placement=left_book_pl,
                asset_name="Book042.usd",
                object_scale=0.4,
            )
        )
        cfgs.append(
            dict(
                name=self.yellow_book,
                obj_groups="book",
                graspable=True,
                placement=right_book_pl,
                asset_name="Book043.usd",
                object_scale=0.4,
            )
        )
        cfgs.append(
            dict(
                name=self.middle_book,
                obj_groups="book",
                graspable=True,
                placement=middle_book_pl,
                asset_name="Book043.usd",
                object_scale=0.4,
            )
        )

        return cfgs


class L90S4PickUpTheBookOnTheLeftAndPlaceItOnTopOfTheShelf(_BaseStudyScene4):
    task_name: str = "L90S4PickUpTheBookOnTheLeftAndPlaceItOnTopOfTheShelf"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book on the left and place it on top of the shelf."
        return ep_meta

    def _check_success(self, env):
        # Check if black book (left book) is placed on top of the desk_caddy (shelf)
        book_on_shelf_result = OU.check_place_obj1_on_obj2(
            env,
            self.black_book,
            self.shelf,
            th_z_axis_cos=0.0,
            th_xy_dist=0.25,
            th_xyz_vel=0.5
        )
        return book_on_shelf_result


class L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf(_BaseStudyScene4):
    task_name: str = "L90S4PickUpTheBookOnTheRightAndPlaceItOnTheCabinetShelf"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick up the book on the right and place it on the cabinet shelf."
        return ep_meta

    def _check_success(self, env):
        # Check if yellow book (right book) is placed on top of the desk_caddy (cabinet shelf)
        book_on_shelf_result = OU.check_place_obj1_on_obj2(
            env,
            self.yellow_book,
            self.shelf,
            th_z_axis_cos=0.0,
            th_xy_dist=0.25,
            th_xyz_vel=0.5
        )
        return book_on_shelf_result
