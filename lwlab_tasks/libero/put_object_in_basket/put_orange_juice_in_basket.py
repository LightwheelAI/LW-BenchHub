from .put_object_in_basket import PutObjectInBasket
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LOPickUpTheOrangeJuiceAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheOrangeJuiceAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["saladdressing", "ketchup", "bbq_sauce"]
    removable_fixtures = enable_fixtures
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.chocolate_pudding = "chocolate_pudding"
        self.butter = "butter"
        self.orange_juice = "orange_juice"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the orange juice and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.orange_juice,
                obj_groups=self.orange_juice,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.25),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="OrangeJuice001.usd",
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
                name=self.butter,
                obj_groups=self.butter,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Butter001.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the orange juice is placed in the basket.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.orange_juice)
        object_in_basket = OU.check_obj_in_receptacle(env, self.orange_juice, self.basket)
        return object_in_basket & is_gripper_obj_far
