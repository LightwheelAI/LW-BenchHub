import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle(LwLabTaskBase):

    task_name: str = "L90K2StackTheBlackBowlAtTheFrontOnTheBlackBowlInTheMiddle"
    enable_fixtures = ["storage_furniture"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.6)),
        )
        self.drawer = self.register_fixture_ref("singlecabinet", dict(id=FixtureType.STORAGE_FURNITURE))
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"stack the black bowl at the front on the black bowl in the middle."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                object_scale=0.7,  # Scale down bowls to fit better
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),  # Reduce sampling area
                    margin=0.02,
                    pos=(0.0, -0.5),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl_front",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                object_scale=0.7,  # Scale down bowls to fit better
                asset_name="Bowl008.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),  # Reduce sampling area
                    margin=0.02,
                    pos=(0.0, -0.9),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl_back",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                object_scale=0.7,  # Scale down bowls to fit better
                asset_name="Bowl008.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.30),  # Reduce sampling area
                    margin=0.02,
                    pos=(0.3, -0.2),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"plate",
                obj_groups=["plate"],
                graspable=True,
                washable=True,
                object_scale=0.6,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.35, 0.30),  # Reduce sampling area
                    margin=0.02,
                    pos=(-0.4, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        ret = OU.check_place_obj1_on_obj2(env, "akita_black_bowl_front", "akita_black_bowl")
        ret1 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl_front", "akita_black_bowl_back")
        ret2 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl_back", "akita_black_bowl_front")
        ret3 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl_back", "akita_black_bowl")
        ret4 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl", "akita_black_bowl_front")
        ret5 = OU.check_place_obj1_on_obj2(env, "akita_black_bowl", "akita_black_bowl_back")
        return ret | ret1 | ret2 | ret3 | ret4 | ret5
