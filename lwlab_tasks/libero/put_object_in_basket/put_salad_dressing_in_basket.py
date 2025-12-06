from .put_object_in_basket import PutObjectInBasket
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LOPickUpTheSaladDressingAndPlaceItInTheBasket(PutObjectInBasket):

    task_name: str = f"LOPickUpTheSaladDressingAndPlaceItInTheBasket"
    enable_fixtures: list[str] = ["ketchup", "saladdressing"]
    EXCLUDE_LAYOUTS: list = [63, 64]
    removable_fixtures: list[str] = ["saladdressing", "ketchup"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.salad_dressing = self.register_fixture_ref("saladdressing", dict(id=FixtureType.SALAD_DRESSING))
        self.ketchup = self.register_fixture_ref("ketchup", dict(id=FixtureType.KETCHUP))
        self.alphabet_soup = "alphabet_soup"
        self.cream_cheese_stick = "cream_cheese_stick"
        self.milk_drink = "milk_drink"
        self.ketchup = "ketchup"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the salad dressing and place it in the basket."
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
                    size=(0.2, 0.3),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="AlphabetSoup001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.cream_cheese_stick,
                obj_groups=self.cream_cheese_stick,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.25),
                    pos=(-0.1, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="CreamCheeseStick013.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.milk_drink,
                obj_groups=self.milk_drink,
                graspable=True,
                placement=dict(
                    fixture=self.floor,
                    size=(0.4, 0.3),
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="MilkDrink009.usd",
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
                    pos=(-0.2, 0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Ketchup003.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the salad dressing is placed in the basket.
        Only use position-based checks, not contact detection.
        '''
        import torch

        # Check 1: Gripper must be far from salad dressing
        is_gripper_obj_far = OU.gripper_obj_far(env, self.salad_dressing.name, th=0.35)

        # Get positions
        salad_pos = torch.tensor(
            env.scene.articulations[self.salad_dressing.name].data.root_link_pos_w,
            device=env.device
        )
        basket_pos = torch.mean(
            env.scene.rigid_objects[self.basket].data.body_com_pos_w,
            dim=1
        )

        # Check 2: Horizontal (XY) position check - must be within basket's horizontal radius
        basket_radius = env.cfg.isaaclab_arena_env.task.objects[self.basket].horizontal_radius
        xy_distance = torch.norm(salad_pos[:, :2] - basket_pos[:, :2], dim=-1)
        is_inside_horizontally = xy_distance < basket_radius * 0.7

        # Check 3: Vertical (Z) height check - must be INSIDE basket, not above or below
        z_diff = salad_pos[:, 2] - basket_pos[:, 2]

        is_inside_vertically = (z_diff > -0.10) & (z_diff < 0.05)

        # Check 4: Salad dressing should be stable (not moving)
        salad_vel = env.scene.articulations[self.salad_dressing.name].data.root_lin_vel_w
        salad_vel_norm = torch.norm(salad_vel, dim=-1)
        is_stable = salad_vel_norm < 0.15

        fixture_in_basket = is_inside_horizontally & is_inside_vertically & is_stable
        return fixture_in_basket & is_gripper_obj_far
