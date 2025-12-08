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


class L90L4PickUpTheChocolatePuddingAndPutItInTheTray(LwTaskBase):
    task_name: str = "L90L4PickUpTheChocolatePuddingAndPutItInTheTray"
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures = ["salad_dressing"]
    removable_fixtures = enable_fixtures

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.TABLE, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"pick up the chocolate pudding and put it in the tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="chocolate_pudding",
                obj_groups="chocolate_pudding",
                graspable=True,
                asset_name="ChocolatePudding001.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.50, 0.50),
                    pos=(-0.1, -0.8),
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
                    fixture=self.counter,
                    size=(0.50, 0.50),
                    pos=(-1.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"wooden_tray",
                obj_groups=["tray"],
                graspable=True,
                washable=True,
                object_scale=0.6,
                asset_name="Tray016.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.55),
                    margin=0.02,
                    pos=(0.4, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        bowl_success = OU.check_obj_in_receptacle(env, "chocolate_pudding", "wooden_tray")
        gipper_success = OU.gripper_obj_far(env, "chocolate_pudding")
        return bowl_success & gipper_success
