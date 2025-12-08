# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class LiberoGoalTasksBase(LwTaskBase):
    """
    LiberoGoalTasksBase: base class for all libero goal tasks
    """

    task_name: str = "LiberoGoalTasksBase"
    enable_fixtures = ["storage_furniture", "stovetop", "winerack"]
    # removable_fixtures = ["winerack"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )

        self.init_robot_base_ref = self.table
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE))
        self.stove = self.register_fixture_ref("stovetop", dict(id=FixtureType.STOVE))
        self.winerack = self.register_fixture_ref("winerack", dict(id=FixtureType.WINE_RACK))

        # Define object names for drawer tasks
        self.akita_black_bowl = "akita_black_bowl"
        self.cream_cheese = "cream_cheese"
        self.plate = "plate"
        self.wine_bottle = "wine_bottle"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the top drawer joint name (first joint)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
        else:
            # Use default joint name if joint info is not available
            self.top_drawer_joint_name = "drawer_joint_1"

    def _get_obj_cfgs(self):
        cfgs = []
        return cfgs

    def _check_success(self, env):
        return torch.tensor([False], device=env.device)


class LGOpenTheMiddleDrawerOfTheCabinet(LiberoGoalTasksBase):
    """
    LGOpenTheMiddleDrawerOfTheCabinet: open the middle layer of the drawer

    Steps:
        1. open the middle drawer of the cabinet

    """

    task_name: str = "LGOpenTheMiddleDrawerOfTheCabinet"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the middle layer of the drawer."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.7),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.1, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the middle drawer joint name (second joint if available, else first)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            joint_names = list(self.drawer._joint_infos.keys())
            self.middle_drawer_joint_name = joint_names[1]  # Second joint is middle drawer
            self.drawer.set_joint_state(0.0, 0.0, env, [self.middle_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.middle_drawer_joint_name = "drawer_joint_2"

    def _check_success(self, env):
        # Check if the middle drawer is open
        drawer_open = self.drawer.is_open(env, [self.middle_drawer_joint_name], th=0.6)
        return drawer_open & OU.gripper_obj_far(env, self.drawer.name, th=0.4)


class LGPutTheBowlOnTopOfTheCabinet(LiberoGoalTasksBase):
    """
    LGPutTheBowlOnTopOfTheCabinet: put the bowl on the top of the drawer

    Steps:
        1. put the bowl on the top of the drawer

    """

    task_name: str = "LGPutTheBowlOnTopOfTheCabinet"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the bowl on the top of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.7),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.1, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if the bowl is on top of the drawer
        # Get bowl position and check if it's on the drawer
        bowl_pos = env.scene.rigid_objects[self.akita_black_bowl].data.root_pos_w[0, :].cpu().numpy()
        bowl_on_drawer = OU.point_in_fixture(bowl_pos, self.drawer, only_2d=True)
        bowl_on_drawer_tensor = torch.tensor(bowl_on_drawer, dtype=torch.bool, device=env.device).repeat(env.num_envs)
        # Check if gripper is far from the bowl
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)

        return bowl_on_drawer_tensor & gripper_far


class LGOpenTopDrawerOfCabinet(LiberoGoalTasksBase):
    """
    LGOpenTopDrawerOfCabinet: open the top drawer of the cabinet

    Steps:
        1. open the top drawer of the cabinet

    """

    task_name: str = "LGOpenTopDrawerOfCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the top drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.7),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.1, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the top drawer joint name (first joint)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
            # Set the top drawer to closed state initially
            self.drawer.set_joint_state(0.0, 0.0, env, [self.top_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.top_drawer_joint_name = "drawer_joint_1"

    def _check_success(self, env):
        # Check if the top drawer is open
        drawer_open = self.drawer.is_open(env, [self.top_drawer_joint_name], th=0.3)
        return drawer_open


class LGPutTheWineBottleOnTopOfTheCabinet(LiberoGoalTasksBase):
    """
    LGPutTheWineBottleOnTopOfTheCabinet: put the wine bottle on the top of the drawer

    Steps:
        1. put the wine bottle on the top of the drawer

    """

    task_name: str = "LGPutTheWineBottleOnTopOfTheCabinet"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the wine bottle on the top of the drawer."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.7),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.1, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if the wine bottle is on top of the drawer
        # Get wine bottle position and check if it's on the drawer
        wine_bottle_pos = env.scene.rigid_objects[self.wine_bottle].data.root_pos_w[0, :].cpu().numpy()
        bottle_on_drawer = OU.point_in_fixture(wine_bottle_pos, self.drawer, only_2d=True)
        bottle_on_drawer_tensor = torch.tensor(bottle_on_drawer, dtype=torch.bool, device=env.device).repeat(env.num_envs)
        # Check if gripper is far from the wine bottle
        gripper_far = OU.gripper_obj_far(env, self.wine_bottle)

        return bottle_on_drawer_tensor & gripper_far


class LGOpenTheTopDrawerAndPutTheBowlInside(LiberoGoalTasksBase):
    """
    LGOpenTheTopDrawerAndPutTheBowlInside: open the top layer of the drawer and put the bowl inside

    Steps:
        1. open the top drawer of the cabinet
        2. put the bowl inside the drawer

    """

    task_name: str = "LGOpenTheTopDrawerAndPutTheBowlInside"
    #  EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the top layer of the drawer and put the bowl inside."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.7),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.1, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                object_scale=0.8,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the top drawer joint name (first joint)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
            # Set the top drawer to closed state initially
            self.drawer.set_joint_state(0.0, 0.0, env, [self.top_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.top_drawer_joint_name = "drawer_joint_1"

    def _check_success(self, env):
        # Check if the top drawer is open
        drawer_open = self.drawer.is_open(env, [self.top_drawer_joint_name], th=0.3)

        # Check if the black bowl is inside the drawer
        bowl_in_drawer = OU.obj_inside_of(env, self.akita_black_bowl, "storage_furniture")

        # Check if gripper is far from the bowl
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)

        # Convert to boolean and combine results
        return drawer_open & bowl_in_drawer & gripper_far


class LGPutTheBowlOnThePlate(LiberoGoalTasksBase):
    """
    LGPutTheBowlOnThePlate: put the akita black bowl on the plate

    Steps:
        pick up the akita black bowl
        put the akita black bowl on the plate

    """

    task_name: str = "LGPutTheBowlOnThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the akita black bowl and put it on the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.8),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.1, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0.2, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.25,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        return success


class LGPutTheCreamCheeseInTheBowl(LiberoGoalTasksBase):
    """
    LGPutTheCreamCheeseInTheBowl: put the cream cheese in the bowl

    Steps:
        pick up the cream cheese
        put the cream cheese in the bowl

    """

    task_name: str = "LGPutTheCreamCheeseInTheBowl"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the cream cheese and put it in the bowl."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.3, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(-0.9, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.cream_cheese,
            self.akita_black_bowl,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.25,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        return success


class LGPushThePlateToTheFrontOfTheStove(LiberoGoalTasksBase):
    """
    LGPushThePlateToTheFrontOfTheStove: push the plate to the front of the stove

    Steps:
        push the plate to the front of the stove

    """
    task_name: str = "LGPushThePlateToTheFrontOfTheStove"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Push the plate to the front of the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.8, 0.20),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.5),
        )
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(0.8, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        stove_pos = self.stove.pos
        plate_poses = OU.get_object_pos(env, self.plate)
        plate_success_tensor = torch.tensor([False] * env.num_envs, device=env.device)
        for i, plate_pos in enumerate(plate_poses):
            x_dist = plate_pos[0] - stove_pos[0]
            success = stove_pos[1] - plate_pos[1] > 0.3 and x_dist < self.stove.size[0] / 2.0
            plate_success_tensor[i] = success
        return plate_success_tensor & OU.gripper_obj_far(env, self.plate, th=0.35)


class LGPutTheBowlOnTheStove(LiberoGoalTasksBase):
    """
    LGPutTheBowlOnTheStove: put the bowl on the stove

    Steps:
        put the bowl on the stove

    """
    task_name: str = "LGPutTheBowlOnTheStove"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the bowl on the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(0.5, 0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0.5, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(1.0, 0.5),
        )
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.6),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(0.3, 0.50),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.5),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
                object_scale=0.8,
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
                object_scale=0.8,
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if the bowl is on the stove
        # Get bowl position and check if it's on the stove
        bowl_pos = env.scene.rigid_objects[self.akita_black_bowl].data.root_pos_w[0, :].cpu().numpy()
        bowl_on_stove = OU.point_in_fixture(bowl_pos, self.stove, only_2d=True)
        bowl_on_stove_tensor = torch.tensor(bowl_on_stove, dtype=torch.bool, device=env.device).repeat(env.num_envs)
        # Check if gripper is far from the bowl
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)

        # Convert to boolean and combine results
        return bowl_on_stove_tensor & gripper_far


class LGPutTheWineBottleOnTheRack(LiberoGoalTasksBase):
    """
    LGPutTheWineBottleOnTheRack: put the wine bottle on the rack

    Steps:
        put the wine bottle on the rack

    """
    task_name: str = "LGPutTheWineBottleOnTheRack"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the wine bottle on the rack."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0.2, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -1.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(0.6, -1.50),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.7),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if the wine bottle is on the rack
        # Get wine bottle position and check if it's on the rack
        wine_bottle_pos = env.scene.rigid_objects[self.wine_bottle].data.root_pos_w[0, :].cpu().numpy()
        bottle_on_rack = OU.point_in_fixture(wine_bottle_pos, self.winerack, only_2d=True)
        bottle_on_rack_tensor = torch.tensor(bottle_on_rack, dtype=torch.bool, device=env.device).repeat(env.num_envs)
        # Check if gripper is far from the wine bottle
        gripper_far = OU.gripper_obj_far(env, self.wine_bottle)

        # Convert to boolean and combine results

        return bottle_on_rack_tensor & gripper_far


class LGTurnOnTheStove(LiberoGoalTasksBase):
    """
    LGTurnOnTheStove: turn on the stove

    Steps:
        turn on the stove

    """
    task_name: str = "LGTurnOnTheStove"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn on the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, cream_cheese, plate, wine_bottle
        wine_bottle_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        cream_cheese_placement = dict(
            fixture=self.table,
            pos=(0.2, -0.70),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )
        plate_placement = dict(
            fixture=self.table,
            pos=(0.4, -1.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )
        bowl_placement = dict(
            fixture=self.table,
            pos=(0.6, 0.00),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.7),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.wine_bottle,
                obj_groups="bottle",
                graspable=True,
                placement=wine_bottle_placement,
                object_scale=0.8,
                asset_name="Bottle054.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.cream_cheese,
                obj_groups="cream_cheese",
                graspable=True,
                placement=cream_cheese_placement,
                asset_name="CreamCheeseStick013.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if the stove is turned on
        # Get the stove fixture and check if any knob is turned on
        knobs_state = self.stove.get_knobs_state(env)

        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_on = (abs_knob >= lower) & (abs_knob <= upper)
            knob_success = knob_success | knob_on
        # Check if gripper is far from the stove (not interacting with it)
        # Use the utility function to check gripper distance from stove
        gripper_far_from_stove = OU.gripper_obj_far(env, self.stove.name, th=0.3)

        return knob_success & gripper_far_from_stove
