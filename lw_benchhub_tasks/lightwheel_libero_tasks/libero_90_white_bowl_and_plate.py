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

from copy import deepcopy

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.tasks.base import LwTaskBase
from lw_benchhub.core.models.fixtures import FixtureType


class L90K7PutTheWhiteBowlOnThePlate(LwTaskBase):
    """
    L90K7PutTheWhiteBowlOnThePlate: put the white bowl on the plate
    """

    task_name: str = "L90K7PutTheWhiteBowlOnThePlate"
    enable_fixtures = ['microwave']

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.microwave = self.register_fixture_ref("microwave", dict(id=FixtureType.MICROWAVE))
        self.init_robot_base_ref = self.counter
        self.plate = "plate"
        self.white_bowl = "white_bowl"

    def _load_model(self):
        super()._load_model()
        mircowave_pos = self.microwave.pos
        mircowave_size = self.microwave.size
        bowl_obj = deepcopy(self.object_placements[self.white_bowl])
        bowl_pos = list(bowl_obj[0])
        bowl_pos[0] = mircowave_pos[0]
        bowl_pos[1] = mircowave_pos[1]
        bowl_pos[2] = bowl_pos[2] + mircowave_size[2]
        bowl_obj = list(bowl_obj)
        bowl_obj[0] = bowl_pos
        self.object_placements[self.white_bowl] = tuple(bowl_obj)

    def _get_obj_cfgs(self):
        cfgs = []

        plate_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(0.0, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
        )
        white_bowl_placement = dict(
            fixture=self.counter,
            size=(0.5, 0.5),
            pos=(-0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups=self.plate,
                graspable=True,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.white_bowl,
                obj_groups=self.white_bowl,
                graspable=True,
                placement=white_bowl_placement,
                asset_name="Bowl011.usd",
            )
        )
        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the white bowl and put it on the plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.white_bowl,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.25,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        return success


class L90K7PutTheWhiteBowlToTheRightOfThePlate(L90K7PutTheWhiteBowlOnThePlate):
    task_name: str = "L90K7PutTheWhiteBowlToTheRightOfThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the white bowl to the right of the plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_side_by_obj2(
            env,
            self.white_bowl,
            self.plate,
            {
                "gripper_far": True,
                "contact": False,
                "side": "right",
                "side_threshold": 0.7,
                "margin_threshold": [0.001, 0.2],
                "stable_threshold": 0.5,
            }
        )
        return success


class L90K7OpenTheMicrowave(L90K7PutTheWhiteBowlToTheRightOfThePlate):
    task_name: str = "L90K7OpenTheMicrowave"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the microwave."
        return ep_meta

    def _check_success(self, env):
        return self.microwave.is_open(env, th=0.6) & OU.gripper_obj_far(env, self.microwave.name)
