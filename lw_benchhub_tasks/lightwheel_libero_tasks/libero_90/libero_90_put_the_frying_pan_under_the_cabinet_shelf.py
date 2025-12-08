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


class L90K9PutTheFryingPanUnderTheCabinetShelf(LwTaskBase):
    task_name: str = 'L90K9PutTheFryingPanUnderTheCabinetShelf'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["stovetop"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the frying pan under the cabinet shelf."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.dining_table
        self.shelf = "shelf"
        self.frying_pan = "frying_pan"
        self.bowl = "bowl"

    def _get_obj_cfgs(self):
        cfgs = []

        # Place shelf first - reduce scale and adjust position to avoid blocking pan
        cfgs.append(
            dict(
                name=self.shelf,
                obj_groups="shelf",
                graspable=True,
                object_scale=1.0,  # Reduced from 1.2 to 1.0 for more clearance
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.2, 0.2),
                    pos=(-1, 0.4),
                    ensure_object_boundary_in_range=False,
                    rotation=np.pi / 2,
                ),
                asset_name="Shelf073.usd",
            )
        )

        # Place frying pan second - it's the main object
        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.4, -0.05),
                    rotation=-np.pi / 8,
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Pot086.usd",
            )
        )

        # Place bowl last in a separate area to avoid conflicts
        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),  # Even larger sampling area
                    pos=(0.2, 0.5),   # Move to a different area (upper right)
                    ensure_valid_placement=True,
                    margin=0.01,      # Reduce margin requirement for faster placement
                ),
                asset_name="Bowl008.usd",
            )
        )

        return cfgs

    def _check_success(self, env):

        is_gripper_obj_far = OU.gripper_obj_far(env, self.frying_pan, th=0.4)
        pot_on_shelf = OU.check_obj_in_receptacle_no_contact(env, self.frying_pan, self.shelf)
        pot_is_stable = OU.check_object_stable(env, self.frying_pan)
        pan_z = env.scene.rigid_objects[self.frying_pan].data.body_com_pos_w[0, 0, 2]
        shelf_z = env.scene.rigid_objects[self.shelf].data.body_com_pos_w[0, 0, 2]
        return is_gripper_obj_far & pot_on_shelf & pot_is_stable & (pan_z < shelf_z)
