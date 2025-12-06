import torch
from lwlab.core.tasks.base import LwLabTaskBase
from .put_black_bowl_on_plate import PutBlackBowlOnPlate
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import copy


class LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate(PutBlackBowlOnPlate):

    task_name: str = 'LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the cookies box and place it on the plate."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.bowl_target = "bowl_target"
        self.bowl = "bowl"
        self.storage_furniture.set_target_reg_int(("int1",))

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
                    object=self.cookies,
                    size=(1.0, 1.0),
                    # ensure_valid_placement=True,
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


class LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate(LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate):
    task_name: str = 'LSPickUpTheBlackBowlOnTheRamekinAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the ramekin and place it on the plate."
        return ep_meta

    def _load_model(self):
        super()._load_model()
        if self.fix_object_pose_cfg is None:
            self.fix_object_pose_cfg = {}
        ramekin_pos = list(self.object_placements[self.ramekin][0])
        bowl_obj = copy.deepcopy(self.object_placements[self.bowl_target])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = ramekin_pos[0]
        bowl_pos[1] = ramekin_pos[1]
        bowl_pos[2] = bowl_pos[2] + self.object_placements[self.ramekin][2].size[2] - 0.05
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.fix_object_pose_cfg[self.bowl_target] = {"pos": bowl_pos}
        self.object_placements[self.bowl_target] = tuple(bowl_obj)


class LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate(LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate):
    task_name: str = 'LSPickUpTheBlackBowlOnTheStoveAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the stove and place it on the plate."
        return ep_meta

    def _load_model(self):
        super()._load_model()
        if self.fix_object_pose_cfg is None:
            self.fix_object_pose_cfg = {}
        stove_pos = self.stove.pos
        bowl_obj = copy.deepcopy(self.object_placements[self.bowl_target])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = stove_pos[0]
        bowl_pos[1] = stove_pos[1]
        bowl_pos[2] = bowl_pos[2] + self.stove.size[2]
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.fix_object_pose_cfg[self.bowl_target] = {"pos": bowl_pos}
        self.object_placements[self.bowl_target] = tuple(bowl_obj)


class LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate(LSPickUpTheBlackBowlOnTheCookieBoxAndPlaceItOnThePlate):
    task_name: str = 'LSPickUpTheBlackBowlOnTheWoodenCabinetAndPlaceItOnThePlate'
    EXCLUDE_LAYOUTS: list = [63, 64]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick the akita black bowl on the wooden cabinet and place it on the plate."
        return ep_meta

    def _load_model(self):

        if self.fix_object_pose_cfg is None:
            self.fix_object_pose_cfg = {}

        PutBlackBowlOnPlate._load_model(self)

        cabinet_pos = self.storage_furniture.pos
        bowl_obj = self.object_placements[self.bowl_target]
        bowl_height = bowl_obj[2].size[2]

        bowl_target_pos = (
            cabinet_pos[0],
            cabinet_pos[1],
            cabinet_pos[2] + self.storage_furniture.size[2] / 2 + bowl_height / 2.0 + 0.01
        )

        self.fix_object_pose_cfg[self.bowl_target] = {"pos": bowl_target_pos}

        bowl_obj_list = list(bowl_obj)
        bowl_obj_list[0] = bowl_target_pos
        self.object_placements[self.bowl_target] = tuple(bowl_obj_list)

    def _get_obj_cfgs(self):
        cfgs = PutBlackBowlOnPlate._get_obj_cfgs(self)

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

        cfgs.append(
            dict(
                name=self.bowl_target,
                obj_groups="bowl",
                asset_name=self.bowl_asset_name,
                graspable=True,
                object_scale=0.6,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.4, 0.4),
                    pos=(-0.4, -0.4),
                    ensure_valid_placement=True,
                ),
            )
        )

        return cfgs
