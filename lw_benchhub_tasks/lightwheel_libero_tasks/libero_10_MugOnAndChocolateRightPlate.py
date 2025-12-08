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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate(LwTaskBase):
    """
    L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate: put the white mug on the plate and put the chocolate pudding to the right of the plate
    """

    task_name: str = "L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        self.place_success = {}
        self.chocolate_pudding = "chocolate_pudding"
        self.plate = "plate"
        self.porcelain_mug = "porcelain_mug"
        self.red_coffee_mug = "red_coffee_mug"
        self.white_yellow_mug = "white_yellow_mug"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the white mug and put it on the plate, and put the chocolate pudding to the right of the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        plate_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.6),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )
        chocolate_pudding_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.7),
            margin=0.02,
            rotation=np.pi / 2.0,
            ensure_valid_placement=True,
        )
        porcelain_mug_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.6),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )
        red_coffee_mug_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.6),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups=self.chocolate_pudding,
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.plate,
                obj_groups=self.plate,
                graspable=True,
                placement=plate_placement,
                asset_name="Plate012.usd",
                init_robot_here=True,
            )
        )
        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups=self.porcelain_mug,
                graspable=True,
                placement=porcelain_mug_placement,
                asset_name="Cup012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups=self.red_coffee_mug,
                graspable=True,
                placement=red_coffee_mug_placement,
                asset_name="Cup030.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        success_porcelain_mug = OU.check_place_obj1_on_obj2(
            env,
            self.porcelain_mug,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5,     # velocity vector length less than 0.5
            gipper_th=0.35
        )
        success_chocolate_pudding = OU.check_place_obj1_side_by_obj2(
            env,
            self.chocolate_pudding,
            self.plate,
            check_states={
                "side": "right",    # right side of obj2
                "side_threshold": 0.4,    # threshold for distance between obj1 and obj2 in other directions, 0.25*(min(obj2_obj.size[:2]) + min(obj1_obj.size[:2]))/ 2
                "margin_threshold": [0.001, 0.1],     # threshold for distance between obj1 and obj2, 0.001
                "parallel": [0, 0, 1],    # parallel to y and z axis
                "parallel_threshold": 0.95,
                "gripper_far": True,
                "contact": False,   # not allowed to contact with obj2
            },
            gipper_th=0.35
        )
        return success_porcelain_mug & success_chocolate_pudding
