import torch
from lwlab.core.tasks.base import LwLabTaskBase
import copy
from .put_black_bowl_on_plate import PutBlackBowlOnPlate
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LSPickUpTheBlackBowlNextToTheRamekinAndPlaceItOnThePlate(PutBlackBowlOnPlate):

    task_name: str = 'LSPickUpTheBlackBowlNextToTheRamekinAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl next to the ramekin and place it on the plate."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.bowl_target = "bowl_target"
        self.bowl = "bowl"

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        cfgs.append(
            dict(
                name=self.bowl_target,
                obj_groups="bowl",
                asset_name=self.bowl_asset_name,
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.3, 0.3),
                    pos=(0.2, -0.8),
                    ensure_valid_placement=True,
                ),
            )
        )
        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                asset_name=self.bowl_asset_name,
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.2, -0.6),
                    ensure_valid_placement=True,
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        '''
        Check if the bowl is placed on the plate.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.bowl_target)

        bowl_pos = torch.mean(env.scene.rigid_objects[self.bowl_target].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        plate_pos = torch.mean(env.scene.rigid_objects[self.plate].data.body_com_pos_w, dim=1)  # (num_envs, 3)

        xy_distance = torch.norm(bowl_pos[:, :2] - plate_pos[:, :2], dim=1)
        bowl_centered = xy_distance < 0.08

        z_diff = bowl_pos[:, 2] - plate_pos[:, 2]
        bowl_on_plate_height = (z_diff > 0.01) & (z_diff < 0.15)

        bowl_vel = torch.mean(env.scene.rigid_objects[self.bowl_target].data.body_com_vel_w, dim=1)  # (num_envs, 3)
        bowl_speed = torch.norm(bowl_vel, dim=1)

        bowl_stable = bowl_speed < 0.05

        success = is_gripper_obj_far & bowl_centered & bowl_on_plate_height & bowl_stable
        return success
