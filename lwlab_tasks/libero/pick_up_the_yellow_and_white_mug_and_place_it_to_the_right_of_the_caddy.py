import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU

import numpy as np


class L90S1PickUpTheYellowAndWhiteMugAndPlaceItToTheRightOfTheCaddy(LwLabTaskBase):
    task_name: str = "L90S1PickUpTheYellowAndWhiteMugAndPlaceItToTheRightOfTheCaddy"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.init_robot_base_ref = self.table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "pick up the yellow and white mug and place it to the right of the caddy."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

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
                    fixture=self.table,
                    rotation=0.0,
                    size=(0.6, 0.4),
                    pos=(0.0, -0.1),
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
                    fixture=self.table,
                    size=(0.25, 0.25),
                    pos=(-0.5, -0.5),
                    ensure_valid_placement=True,
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
                    fixture=self.table,
                    rotation=-np.pi / 4.0,
                    size=(0.4, 0.4),
                    pos=(-0.3, -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        success_mug_caddy = OU.check_place_obj1_side_by_obj2(
            env,
            "white_yellow_mug",
            "desk_caddy",
            check_states={
                "side": "right",
                "side_threshold": 1.0,
                "margin_threshold": [0.001, 0.2],
                "gripper_far": True,
                "contact": False,
            }
        )
        return success_mug_caddy
