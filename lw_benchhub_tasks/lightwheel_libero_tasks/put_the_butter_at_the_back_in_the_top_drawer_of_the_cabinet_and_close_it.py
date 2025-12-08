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


class L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt(LwTaskBase):
    task_name: str = "L90K10PutTheButterAtTheBackInTheTopDrawerOfTheCabinetAndCloseIt"
    enable_fixtures = ["storage_furniture"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.table))

        self.init_robot_base_ref = self.table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "put the butter at the back in the top drawer of the cabinet and close it."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
        self.drawer.set_joint_state(0.1, 0.2, env, [self.top_drawer_joint_name])

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="butter",
                obj_groups="butter",
                graspable=True,
                object_scale=0.8,
                asset_name="Butter001.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.80, 0.3),
                    pos=(0.0, -0.6),
                ),
            )
        )

        cfgs.append(
            dict(
                name="chocolate_pudding",
                obj_groups="chocolate_pudding",
                graspable=True,
                asset_name="ChocolatePudding001.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.80, 0.3),
                    pos=(-0.2, -0.8),
                ),
            )
        )
        cfgs.append(
            dict(
                name="akita_black_bowl",
                obj_groups="bowl",
                graspable=True,
                asset_name="Bowl008.usd",
                placement=dict(
                    fixture=self.table,
                    size=(0.80, 0.3),
                    pos=(0.2, -0.6),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        gipper_success = OU.gripper_obj_far(env, "butter")
        butter_in_drawer = OU.obj_inside_of(env, "butter", self.drawer, partial_check=True)
        cabinet_closed = self.drawer.is_closed(env, [self.top_drawer_joint_name])
        return cabinet_closed & gipper_success & butter_in_drawer
