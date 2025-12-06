import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class L90S1PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy(LwLabTaskBase):
    task_name: str = 'L90S1PickUpTheBookAndPlaceItInTheLeftCompartmentOfTheCaddy'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the book and place it in the left compartment of the caddy."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.init_robot_base_ref = self.dining_table
        self.desk_caddy = "desk_caddy"
        self.book = "book"
        self.mug = "mug"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.desk_caddy,
                obj_groups=self.desk_caddy,
                graspable=True,
                object_scale=2.0,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.6, 0.6),
                    pos=(-0.5, -0.3),
                ),
                asset_name="DeskCaddy001.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.book,
                obj_groups="book",
                graspable=True,
                object_scale=0.3,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.3, -0.7),
                    ensure_valid_placement=True,
                ),
                asset_name="Book042.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.mug,
                obj_groups="mug",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.5, 0.5),
                    ensure_valid_placement=True,
                ),
                asset_name="Cup014.usd",
            )
        )

        return cfgs

    def _check_success(self, env):

        # Get full 3D positions for all environments
        book_pos_full = torch.mean(env.scene.rigid_objects[self.book].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        caddy_pos_full = torch.mean(env.scene.rigid_objects[self.desk_caddy].data.body_com_pos_w, dim=1)  # (num_envs, 3)

        # Check 1: gripper far from book
        is_gripper_obj_far = OU.gripper_obj_far(env, self.book)  # tensor (num_envs,)

        # Check 2: book is near caddy (xy distance check)
        xy_dist = torch.norm(book_pos_full[:, :2] - caddy_pos_full[:, :2], dim=-1)  # (num_envs,)
        caddy_obj = env.cfg.isaaclab_arena_env.task.objects[self.desk_caddy]
        th = float(caddy_obj.horizontal_radius * 0.7)  # convert to float scalar
        object_on_caddy = xy_dist < th  # tensor (num_envs,) - comparison with scalar is valid

        # Check 3: book is in left compartment (x coordinate check)
        pos_success = (book_pos_full[:, 0] - caddy_pos_full[:, 0]) > 0.1  # tensor (num_envs,)

        # Check 4: book is actually inside the caddy (height check)
        # Book should be lower than caddy top, ensuring it's inside not just on top
        z_diff = book_pos_full[:, 2] - caddy_pos_full[:, 2]
        book_inside_caddy = (z_diff > -0.05) & (z_diff < 0.1)  # tensor (num_envs,)

        # Check 5: book is stable (not moving)
        book_stable = OU.check_object_stable(env, self.book, threshold=0.5)  # tensor (num_envs,)

        return pos_success & is_gripper_obj_far & object_on_caddy & book_inside_caddy & book_stable
