import copy
from lwlab.core.tasks.base import LwLabTaskBase
import re
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90L4PickUpTheSaladDressingAndPutItInTheTray(LwLabTaskBase):
    """
    L90L4PickUpTheSaladDressingAndPutItInTheTray: pick up the salad dressing and put it in the tray
    """

    task_name: str = "L90L4PickUpTheSaladDressingAndPutItInTheTray"

    enable_fixtures = ["saladdressing"]
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.salad_dressing = self.register_fixture_ref("saladdressing", dict(id=FixtureType.SALAD_DRESSING))
        self.init_robot_base_ref = self.counter
        self.akita_black_bowl = "akita_black_bowl"
        self.chocolate_pudding = "chocolate_pudding"
        self.wooden_tray = "wooden_tray"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _reset_internal(self, env_ids):
        super()._reset_internal(env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the salad dressing and put it in the tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        tray_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )
        akita_black_bowl_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )
        chocolate_pudding_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups=self.akita_black_bowl,
                graspable=True,
                placement=akita_black_bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wooden_tray,
                obj_groups=self.wooden_tray,
                graspable=True,
                placement=tray_placement,
                asset_name="Tray016.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.salad_dressing,
            self.wooden_tray,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
