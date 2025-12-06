import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K9PutTheWhiteBowlOnTopOfTheCabinet(LwLabTaskBase):
    task_name: str = "L90K9PutTheWhiteBowlOnTopOfTheCabinet"
    enable_fixtures = ["stove"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the white bowl on top of the cabinet"
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.dining_table
        self.shelf = "shelf"
        self.frying_pan = "frying_pan"
        self.bowl = "bowl"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.shelf,
                obj_groups="shelf",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.75, 0.25),
                    rotation=np.pi / 2,
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Shelf073.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.35),
                    pos=(0.1, -0.1),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Bowl009.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.5, -0.25),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Pot086.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        ret = OU.check_place_obj1_on_obj2(env, self.bowl, self.shelf)
        return ret
