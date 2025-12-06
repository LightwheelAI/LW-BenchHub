from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np
import torch


class _BasePutOnStove(LwLabTaskBase):
    task_name: str = "_BasePutOnStove"

    enable_fixtures = ['mokapot_1', 'stovetop']
    removable_fixtures = ['mokapot_1']


class _BasePutRightMokaPotOnStove(LwLabTaskBase):
    task_name: str = "_BasePutRightMokaPotOnStove"

    enable_fixtures = ['mokapot_1', 'mokapot_2', 'stovetop']
    removable_fixtures = ['mokapot_1', 'mokapot_2']

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.stove = self.register_fixture_ref("stovetop", dict(id=FixtureType.STOVE))
        self.mokapot_1 = self.register_fixture_ref("mokapot_1", dict(id="mokapot_1"))
        self.mokapot_2 = self.register_fixture_ref("mokapot_2", dict(id="mokapot_2"))
        self.init_robot_base_ref = self.counter
        self.frying_pan = "chefmate_8_frypan"

    def _get_obj_cfgs(self):
        cfgs = []

        pan_pl = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(1.0, -0.75),
            rotation=-np.pi / 2,
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                placement=pan_pl,
                asset_name="Pot086.usd",
            )
        )

        return cfgs


class L90K3PutTheFryingPanOnTheStove(_BasePutOnStove):
    task_name: str = "L90K3PutTheFryingPanOnTheStove"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the frying pan on the stove."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.mokapot = self.register_fixture_ref("mokapot_1", dict(id=FixtureType.MOKA_POT))
        self.stove = self.register_fixture_ref("stovetop", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.counter
        self.frying_pan = "chefmate_8_frypan"

    def _get_obj_cfgs(self):
        cfgs = []

        pan_pl = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(1.0, -0.75),
            rotation=-np.pi / 2,
            margin=0.02,
            ensure_valid_placement=False,
        )

        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                placement=pan_pl,
                asset_name="Pot086.usd",
            )
        )
        # add_cfg(self.moka_pot, "moka_pot", True, moka_pl, mjcf_path="/objects/lightwheel/moka_pot/MokaPot001/model.xml")

        return cfgs

    def _check_success(self, env):
        pan_on_stove = OU.check_obj_fixture_contact(env, self.frying_pan, self.stove)
        gripper_far = OU.gripper_obj_far(env, self.frying_pan, th=0.4)
        return pan_on_stove & gripper_far


class L90K3PutTheMokaPotOnTheStove(_BasePutOnStove):
    task_name: str = "L90K3PutTheMokaPotOnTheStove"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.mokapot = self.register_fixture_ref("mokapot_1", dict(id=FixtureType.MOKA_POT))
        self.stove = self.register_fixture_ref("stovetop", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.counter
        self.frying_pan = "chefmate_8_frypan"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the moka pot on the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        pan_pl = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(1.0, -0.75),
            rotation=-np.pi / 2,
            margin=0.02,
            ensure_valid_placement=False,
        )

        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                placement=pan_pl,
                asset_name="Pot086.usd",
            )
        )
        # add_cfg(self.moka_pot, "moka_pot", True, moka_pl, mjcf_path="/objects/lightwheel/moka_pot/MokaPot001/model.xml")

        return cfgs

    def _check_success(self, env):
        # Check if moka pot (fixture) is placed on stove
        success = OU.check_place_obj1_on_obj2(
            env,
            self.mokapot,
            self.stove,
            th_xy_dist=0.5,  # Allow larger distance since stove is big
            gipper_th=0.4    # Gripper distance threshold
        )
        return success


class L90K8PutTheRightMokaPotOnTheStove(_BasePutRightMokaPotOnStove):
    task_name: str = "L90K8PutTheRightMokaPotOnTheStove"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the right moka pot on the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        return []

    def _check_success(self, env):
        # Check if at least one moka pot is on the stove
        mokapot_2_on_stove = OU.check_place_obj1_on_obj2(env, self.mokapot_2, self.stove)
        return mokapot_2_on_stove
