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


class LiberoMugPlacementBase(LwTaskBase):
    """
    LiberoMugPlacementBase: base class for all libero mug placement tasks
    """

    task_name: str = "LiberoMugPlacementBase"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):
        cfgs = []

        return cfgs


class L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug(LiberoMugPlacementBase):
    """
    L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug: put the yellow and white mug to the front of the white mug

    Steps:
        pick up the yellow and white mug
        put the yellow and white mug to the front of the white mug

    """

    task_name: str = "L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS
    enable_fixtures = ['microwave']

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.white_yellow_mug = "white_yellow_mug"
        self.porcelain_mug = "porcelain_mug"
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.microwave.set_joint_state(0.9, 1.0, env, self.microwave.door_joint_names)

    def _get_obj_cfgs(self):
        cfgs = []

        white_mug_placement = dict(
            fixture=self.counter,
            size=(0.3, 0.3),
            pos=(0.2, -0.6),
            margin=0.02,
            ensure_valid_placement=True,
        )
        yellow_white_mug_placement = dict(
            fixture=self.counter,
            size=(0.3, 0.3),
            pos=(-0.2, -0.2),
            rotation=-np.pi / 2.0,
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups="cup",
                graspable=False,
                placement=white_mug_placement,
                asset_name="Cup012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.white_yellow_mug,
                obj_groups="cup",
                graspable=True,
                placement=yellow_white_mug_placement,
                asset_name="Cup014.usd",
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the yellow and white mug to the front of the white mug."
        return ep_meta

    def _check_success(self, env):
        return OU.check_place_obj1_side_by_obj2(
            env,
            self.white_yellow_mug,
            self.porcelain_mug,
            {
                "gripper_far": True,
                "contact": False,
                "side": "front",
                "side_threshold": 0.7,
                "margin_threshold": [0.001, 0.2],
                "stable_threshold": 0.5,
            }
        )


class L90K6CloseTheMicrowave(L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug):
    task_name: str = "L90K6CloseTheMicrowave"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the microwave."
        return ep_meta

    def _check_success(self, env):
        return self.microwave.is_closed(env) & OU.gripper_obj_far(env, self.microwave.name, th=0.4)


class L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt(L90K6PutTheYellowAndWhiteMugToTheFrontOfTheWhiteMug):
    task_name: str = "L10K6PutTheYellowAndWhiteMugInTheMicrowaveAndCloseIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the yellow and white mug in the microwave and close it."
        return ep_meta

    def _check_success(self, env):
        mug_poses = OU.get_object_pos(env, self.white_yellow_mug)
        mug_success_tensor = torch.tensor([False] * env.num_envs, device=env.device)
        for i, mug_pos in enumerate(mug_poses):
            mug_success = OU.point_in_fixture(mug_pos, self.microwave)
            mug_success_tensor[i] = torch.as_tensor(mug_success, dtype=torch.bool, device=env.device)
        return mug_success_tensor & self.microwave.is_closed(env) & OU.gripper_obj_far(env, self.microwave.name, th=0.4)


class L90L5PutTheRedMugOnTheLeftPlate(LiberoMugPlacementBase):
    """
    L90L5PutTheRedMugOnTheLeftPlate: put the red mug on the left plate

    Steps:
        pick up the red mug
        put the red mug on the left plate

    """

    task_name: str = "L90L5PutTheRedMugOnTheLeftPlate"
    # EXCLUDE_LAYOUTS: list = LiberoEnvCfg.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.red_coffee_mug = "red_coffee_mug"
        self.plate_left = "plate_left"
        self.plate_right = "plate_right"
        self.porcelain_mug = "porcelain_mug"
        self.white_yellow_mug = "white_yellow_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        plate_left_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(-0.6, -0.4),
            margin=0.02,
            ensure_valid_placement=True,
        )
        plate_right_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.6, -0.4),
            margin=0.02,
            ensure_valid_placement=True,
        )
        red_mug_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -1.2),
            margin=0.02,
            ensure_valid_placement=True,
        )
        white_yellow_mug_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(-0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
        )
        porcelain_mug_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.plate_left,
                obj_groups="plate",
                graspable=False,
                placement=plate_left_placement,
                asset_name="Plate012.usd",
                init_robot_here=True,
            )
        )
        cfgs.append(
            dict(
                name=self.plate_right,
                obj_groups="plate",
                graspable=False,
                placement=plate_right_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups="cup",
                graspable=True,
                placement=red_mug_placement,
                asset_name="Cup030.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups="cup",
                graspable=True,
                placement=porcelain_mug_placement,
                asset_name="Cup012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.white_yellow_mug,
                obj_groups="cup",
                graspable=True,
                placement=white_yellow_mug_placement,
                asset_name="Cup014.usd",
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the red mug on the left plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            'red_coffee_mug',
            'plate_left'
        )
        return success
