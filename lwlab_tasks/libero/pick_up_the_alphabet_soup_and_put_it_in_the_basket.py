import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90L1PickUpTheAlphabetSoupAndPutItInTheBasket(LwLabTaskBase):
    task_name: str = "L90L1PickUpTheAlphabetSoupAndPutItInTheBasket"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the black bowl on the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"basket",
                obj_groups=["basket"],
                graspable=True,
                washable=True,
                asset_name="Basket058.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    margin=0.02,
                    pos=(-0.2, -0.8)
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"alphabet_soup",
                obj_groups=["alphabet_soup"],
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="AlphabetSoup001.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.3),
                    margin=0.02,
                    pos=(0.1, -0.8),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"butter",
                obj_groups=["butter"],
                graspable=True,
                washable=True,
                asset_name="Butter001.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.35),
                    margin=0.02,
                    pos=(0.1, -0.1)
                ),
            )
        )
        cfgs.append(
            dict(
                name="cream_cheese",
                obj_groups="cream_cheese_stick",
                init_robot_here=True,
                graspable=True,
                asset_name="CreamCheeseStick013.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.35),
                    margin=0.02,
                    pos=(0.3, -0.2)
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"tomato_sauce",
                obj_groups=["ketchup"],
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="Ketchup003.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.35),
                    margin=0.02,
                    pos=(-0.1, -0.2)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        th = env.cfg.isaaclab_arena_env.task.objects["basket"].horizontal_radius
        soup_in_tray = OU.check_obj_in_receptacle(env, "alphabet_soup", "basket", th)
        return soup_in_tray & OU.gripper_obj_far(env, "alphabet_soup")
