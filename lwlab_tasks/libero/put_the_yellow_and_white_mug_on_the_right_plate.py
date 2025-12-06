import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class L90L5PutTheYellowAndWhiteMugOnTheRightPlate(LwLabTaskBase):
    task_name: str = "L90L5PutTheYellowAndWhiteMugOnTheRightPlate"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.init_robot_base_ref = self.table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "pick up the yellow and white mug and place it to the right of the caddy."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="porcelain_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup012.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.30),
                    pos=(0.1, -0.2),
                ),
            )
        )
        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup030.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.30),
                    pos=(-0.1, -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name="white_yellow_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup014.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.50),
                    pos=(-0.3, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"plate",
                obj_groups="plate",
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.3),
                    margin=0.02,
                    pos=(0.2, -0.6),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"plate1",
                obj_groups="plate",
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.5),
                    margin=0.02,
                    pos=(-0.4, -0.5),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        success_ret = OU.check_place_obj1_on_obj2(env, "white_yellow_mug", "plate")
        return success_ret


class L90L5PutTheWhiteMugOnTheLeftPlate(LwLabTaskBase):
    task_name: str = "L90L5PutTheWhiteMugOnTheLeftPlate"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.init_robot_base_ref = self.table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "put the white mug on the left plate."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"plate",
                obj_groups="plate",
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.5),
                    margin=0.02,
                    pos=(-0.3, -0.3),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"plate_left",
                obj_groups="plate",
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.5),
                    margin=0.02,
                    pos=(0.4, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="porcelain_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup012.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.30),
                    pos=(0.2, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="red_coffee_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup030.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.50),
                    pos=(0.0, -0.6),
                ),
            )
        )

        cfgs.append(
            dict(
                name="white_yellow_mug",
                obj_groups="cup",
                graspable=True,
                asset_name="Cup014.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.50),
                    pos=(-0.2, -0.7),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        ret = OU.check_place_obj1_on_obj2(env, "porcelain_mug", "plate_left")
        ret2 = OU.check_place_obj1_on_obj2(env, "porcelain_mug", "plate")
        return ret | ret2


class L10L5PutWhiteMugOnLeftPlateAndPutYellowAndWhiteMugOnRightPlate(L90L5PutTheWhiteMugOnTheLeftPlate):
    task_name: str = "L10L5PutWhiteMugOnLeftPlateAndPutYellowAndWhiteMugOnRightPlate"

    def _check_success(self, env):
        ret = OU.check_place_obj1_on_obj2(env, "porcelain_mug", "plate_left")
        ret1 = OU.check_place_obj1_on_obj2(env, "porcelain_mug", "plate")
        porcelain_mug_success = ret | ret1

        ret_right = OU.check_place_obj1_on_obj2(env, "white_yellow_mug", "plate")
        ret_right2 = OU.check_place_obj1_on_obj2(env, "white_yellow_mug", "plate_left")
        white_yellow_mug_success = ret_right | ret_right2
        return porcelain_mug_success & white_yellow_mug_success
