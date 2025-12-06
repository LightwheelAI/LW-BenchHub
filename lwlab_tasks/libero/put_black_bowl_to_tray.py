import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L90L4PickUpTheBlackBowlOnTheLeftAndPutItInTheTray(LwLabTaskBase):
    task_name: str = "L90L4PickUpTheBlackBowlOnTheLeftAndPutItInTheTray"
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures = ["salad_dressing"]
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.TABLE, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the black bowl on the left and put it in the tray."
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
                name="chocolate_pudding",
                obj_groups="chocolate_pudding",
                graspable=True,
                asset_name="ChocolatePudding001.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.50),
                    pos=(-0.1, -0.1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="akita_black_bowl",
                obj_groups="bowl",
                graspable=True,
                asset_name="Bowl008.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.50),
                    pos=(-0.3, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"wooden_tray",
                obj_groups=["tray"],
                graspable=True,
                washable=True,
                object_scale=0.6,
                asset_name="Tray016.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.6, 0.5),
                    rotation=np.pi / 2,
                    margin=0.02,
                    pos=(0.2, -0.7)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        bowl_success = OU.check_obj_in_receptacle(env, "akita_black_bowl", "wooden_tray")
        gipper_success = OU.gripper_obj_far(env, "akita_black_bowl")
        return bowl_success & gipper_success
