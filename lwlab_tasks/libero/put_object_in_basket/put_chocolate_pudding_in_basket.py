from .put_object_in_basket import PutObjectInBasket
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import torch


class LOPickUpTheChocolatePuddingAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheChocolatePuddingAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["saladdressing", "ketchup", "bbq_sauce"]
    removable_fixtures = enable_fixtures
    EXCLUDE_LAYOUTS: list = [63, 64]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.orange_juice = "orange_juice"
        self.alphabet_soup = "alphabet_soup"
        self.chocolate_pudding = "chocolate_pudding"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Pick the chocolate pudding and place it in the basket."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                object_scale=0.5,
                placement=dict(
                    fixture=self.floor,
                    size=(0.2, 0.25),
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
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="OrangeJuice001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.alphabet_soup,
                obj_groups=self.alphabet_soup,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="AlphabetSoup001.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the chocolate pudding is placed in the basket.
        '''

        is_gripper_obj_far = OU.gripper_obj_far(env, self.chocolate_pudding)

        obj_pos = torch.mean(env.scene.rigid_objects[self.chocolate_pudding].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        basket_pos = torch.mean(env.scene.rigid_objects[self.basket].data.body_com_pos_w, dim=1)  # (num_envs, 3)

        xy_dist = torch.norm(obj_pos[:, :2] - basket_pos[:, :2], dim=-1)  # (num_envs,)

        object_in_basket_xy = xy_dist < 0.10

        object_stable = OU.check_object_stable(env, self.chocolate_pudding, threshold=0.5)

        z_diff = obj_pos[:, 2] - basket_pos[:, 2]

        height_check = (z_diff > -0.5) & (z_diff < 0.5)

        return object_in_basket_xy & is_gripper_obj_far & object_stable & height_check
