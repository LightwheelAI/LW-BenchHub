import torch
from lwlab.core.tasks.base import LwLabTaskBase
from .put_black_bowl_on_plate import PutBlackBowlOnPlate
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LSPickUpTheBlackBowlFromTableCenterAndPlaceItOnThePlate(PutBlackBowlOnPlate):

    task_name: str = 'LSPickUpTheBlackBowlFromTableCenterAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl from table center and place it on the plate."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.bowl = "bowl"
        self.bowl_target = "bowl_target"

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()

        # Shift parent objects to the right to open up the table center.
        for cfg in cfgs:
            placement = cfg.get("placement", {})
            if cfg["name"] == self.cookies:
                placement["pos"] = (0.85, 0.8)
            elif cfg["name"] == self.plate:
                placement["pos"] = (0.9, -0.9)
            elif cfg["name"] == self.ramekin:
                placement["pos"] = (0.8, 0.9)

        cfgs.append(
            dict(
                name=self.bowl_target,
                obj_groups="bowl",
                asset_name=self.bowl_asset_name,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.4),
                    pos=(0.0, -0.1),
                    ensure_valid_placement=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                asset_name=self.bowl_asset_name,
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(0.4, -1),
                    ensure_valid_placement=False,
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
        bowl_centered = xy_distance < 0.1

        z_diff = bowl_pos[:, 2] - plate_pos[:, 2]
        bowl_on_plate_height = (z_diff > 0.01) & (z_diff < 0.15)

        bowl_vel = torch.mean(env.scene.rigid_objects[self.bowl_target].data.body_com_vel_w, dim=1)  # (num_envs, 3)
        bowl_speed = torch.norm(bowl_vel, dim=1)

        bowl_stable = bowl_speed < 0.05

        success = is_gripper_obj_far & bowl_centered & bowl_on_plate_height & bowl_stable
        return success
