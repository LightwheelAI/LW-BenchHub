from .put_object_in_basket import PutObjectInBasket
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LOPutCreamCheeseInBasket(PutObjectInBasket):

    task_name: str = f"LOPutCreamCheeseInBasket"
    # enable_fixtures: list[str] = [""]
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.alphabet_soup = "alphabet_soup"
        self.butter = "butter"
        self.milk_drink = "milk_drink"
        self.orange_juice = "orange_juice"
        self.ketchup = "ketchup"
        self.cream_cheese_stick = "cream_cheese_stick"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the cream cheese and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.3, 0.3),
                    pos=(-0.1, -0.1),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/alphabet_soup/AlphabetSoup001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/butter/Butter001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.milk_drink,
                obj_groups=self.milk_drink,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/milk_drink/MilkDrink009/model.xml",
                ),
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
                    pos=(0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/orange_juice/OrangeJuice001/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups=self.ketchup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/ketchup/Ketchup003/model.xml",
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.cream_cheese_stick,
                obj_groups=self.cream_cheese_stick,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.3),
                    pos=(0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                info=dict(
                    mjcf_path="/objects/lightwheel/cream_cheese_stick/CreamCheeseStick013/model.xml",
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the cream cheese is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.cream_cheese_stick)
        object_in_basket = OU.check_obj_in_receptacle(env, self.cream_cheese_stick, self.basket)
        return object_in_basket & is_gripper_obj_far
