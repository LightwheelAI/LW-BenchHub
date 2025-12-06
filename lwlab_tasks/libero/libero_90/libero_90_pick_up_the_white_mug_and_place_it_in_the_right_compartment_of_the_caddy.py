import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L90S3PickUpTheWhiteMugAndPlaceItToTheRightOfTheCaddy(LwLabTaskBase):
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

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

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
