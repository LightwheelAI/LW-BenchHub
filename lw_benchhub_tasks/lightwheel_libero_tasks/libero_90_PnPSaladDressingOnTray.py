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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class L90L4PickUpTheSaladDressingAndPutItInTheTray(LwTaskBase):
    """
    L90L4PickUpTheSaladDressingAndPutItInTheTray: pick up the salad dressing and put it in the tray
    """

    task_name: str = "L90L4PickUpTheSaladDressingAndPutItInTheTray"

    enable_fixtures = ["saladdressing"]
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.salad_dressing = self.register_fixture_ref("saladdressing", dict(id=FixtureType.SALAD_DRESSING))
        self.init_robot_base_ref = self.counter
        self.akita_black_bowl = "akita_black_bowl"
        self.chocolate_pudding = "chocolate_pudding"
        self.wooden_tray = "wooden_tray"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the salad dressing and put it in the tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        tray_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )
        akita_black_bowl_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )
        chocolate_pudding_placement = dict(
            fixture=self.counter,
            size=(0.8, 0.8),
            pos=(0.0, -1),
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups=self.akita_black_bowl,
                graspable=True,
                placement=akita_black_bowl_placement,
                asset_name="Bowl008.usd",
            )
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
                name=self.wooden_tray,
                obj_groups=self.wooden_tray,
                graspable=True,
                placement=tray_placement,
                asset_name="Tray016.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.salad_dressing,
            self.wooden_tray,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
