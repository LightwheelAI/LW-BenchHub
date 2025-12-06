import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class PubBookInCaddy(LwLabTaskBase):
    task_name: str = "PubBookInCaddy"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.6)),
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
                graspable=True,
                object_scale=0.4,
                asset_name="Book042.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(-0.1, -0.5)
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
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        return False


class L90S3PickUpTheRedMugAndPlaceItToTheRightOfTheCaddy(PubBookInCaddy):
    task_name = "L90S3PickUpTheRedMugAndPlaceItToTheRightOfTheCaddy"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the red mug and place it to the right compartment of the caddy."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
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
                    size=(0.8, 0.5),
                    pos=(-0.2, -0.1),
                ),
            )
        )
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
                    pos=(-0.4, -0.5)
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
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(0.3, -0.6)
                ),
            )
        )
        cfgs.append(
            dict(
                name="porcelain_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup012.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.3),
                    margin=0.02,
                    pos=(-0.1, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        success = OU.check_place_obj1_side_by_obj2(
            env,
            "red_coffee_mug",
            "desk_caddy",
            {
                "gripper_far": True,
                "side": "right",
                "side_threshold": 0.5,
                "margin_threshold": [0.001, 0.3],
                "stable_threshold": 0.5,
            }
        )

        return success
