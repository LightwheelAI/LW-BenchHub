import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90L3PickUpTheButterAndPutItInTheTray(LwLabTaskBase):
    task_name: str = 'L90L3PickUpTheButterAndPutItInTheTray'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["ketchup"]
    removable_fixtures = enable_fixtures

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the butter and put it in the tray."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP))

        self.init_robot_base_ref = self.dining_table
        self.tray = "tray"
        self.butter = "butter"
        self.cream_cheese_stick = "cream_cheese_stick"
        self.ketchup = "ketchup"
        self.alphabet_soup = "alphabet_soup"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.tray,
                obj_groups=self.tray,
                object_scale=0.6,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.25, -0.2),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Tray016.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.5, -0.1),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="AlphabetSoup001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.4, -0.3),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Butter001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups=self.ketchup,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, -0.1),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Ketchup003.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.cream_cheese_stick,
                obj_groups=self.cream_cheese_stick,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.5, -0.1),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):

        is_gripper_obj_far = OU.gripper_obj_far(env, self.butter)
        object_on_tray = OU.check_obj_in_receptacle(env, self.butter, self.tray)
        return object_on_tray & is_gripper_obj_far
