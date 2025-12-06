import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K1OpenTheBottomDrawerOfTheCabinet(LwLabTaskBase):
    task_name: str = "L90K1OpenTheBottomDrawerOfTheCabinet"
    enable_fixtures = ["storage_furniture"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.dining_table))
        self.init_robot_base_ref = self.dining_table

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.botton_joint_name = list(self.drawer._joint_infos.keys())[-1]
        self.top_joint_name = list(self.drawer._joint_infos.keys())[0]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the bottom drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups="bowl",
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.5),
                    margin=0.02,
                    pos=(0.1, -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"plate",
                obj_groups="plate",
                graspable=True,
                washable=True,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(0.2, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        return self.drawer.is_open(env, [self.botton_joint_name], th=0.5) & OU.gripper_obj_far(env, self.drawer.name, th=0.5)


class L90K1OpenTheTopDrawerOfTheCabinet(L90K1OpenTheBottomDrawerOfTheCabinet):
    task_name: str = "L90K1OpenTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"open the top drawer of the cabinet."
        return ep_meta

    def _check_success(self, env):
        return self.drawer.is_open(env, [self.top_joint_name], th=0.5) & OU.gripper_obj_far(env, self.drawer.name, th=0.5)
