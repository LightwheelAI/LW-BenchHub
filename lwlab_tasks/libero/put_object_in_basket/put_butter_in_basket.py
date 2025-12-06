from .put_object_in_basket import PutObjectInBasket
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import torch


class LOPickUpTheButterAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheButterAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["ketchup", "bbq_sauce"]
    removable_fixtures = enable_fixtures
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP))
        self.bbq_sauce = self.register_fixture_ref("bbq_sauce", dict(id=FixtureType.BBQ_SOURCE))
        self.chocolate_pudding = "chocolate_pudding"
        self.orange_juice = "orange_juice"
        self.ketchup = "ketchup"
        self.butter = "butter"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the butter and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Butter001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="ChocolatePudding001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.orange_juice,
                obj_groups=self.orange_juice,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="OrangeJuice001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups=self.ketchup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Ketchup003.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the butter is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.butter)
        object_in_basket = OU.check_obj_in_receptacle(env, self.butter, self.basket)
        return object_in_basket & is_gripper_obj_far


class L90L2PickUpTheButterAndPutItInTheBasket(LwLabTaskBase):
    task_name: str = f"L90L2PickUpTheButterAndPutItInTheBasket"
    enable_fixtures = ["ketchup"]
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.obj_name = []
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )

        self.init_robot_base_ref = self.dining_table

    def _load_model(self):
        super()._load_model()
        for cfg in self.object_cfgs:
            self.obj_name.append(cfg["name"])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the butter and put it in the basket."
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
                asset_name="AlphabetSoup001.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.3),
                    margin=0.02,
                    pos=(0.1, 0.1),
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
                    pos=(0.1, -0.8)
                ),
            )
        )
        cfgs.append(
            dict(
                name="milk_drink",
                obj_groups="milk_drink",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, -0.5),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="MilkDrink009.usd",
            )
        )
        cfgs.append(
            dict(
                name="orange_juice",
                obj_groups="orange_juice",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    pos=(-0.25, 0.2),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="OrangeJuice001.usd",
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
        '''
        Check if the butter is placed in the basket.
        '''

        far_from_objects = self._gripper_obj_farfrom_objects(env)

        obj_pos = torch.mean(env.scene.rigid_objects["butter"].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        basket_pos = torch.mean(env.scene.rigid_objects["basket"].data.body_com_pos_w, dim=1)  # (num_envs, 3)

        xy_dist = torch.norm(obj_pos[:, :2] - basket_pos[:, :2], dim=-1)  # (num_envs,)
        object_in_basket_xy = xy_dist < 0.5

        object_stable = OU.check_object_stable(env, "butter", threshold=0.01)

        z_diff = obj_pos[:, 2] - basket_pos[:, 2]
        height_check = (z_diff > -0.05) & (z_diff < 0.02)

        return object_in_basket_xy & far_from_objects & object_stable & height_check

    def _gripper_obj_farfrom_objects(self, env):
        gripper_far_tensor = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for obj_name in self.obj_name:
            gripper_far_tensor = gripper_far_tensor & OU.gripper_obj_far(env, obj_name)
        return gripper_far_tensor
