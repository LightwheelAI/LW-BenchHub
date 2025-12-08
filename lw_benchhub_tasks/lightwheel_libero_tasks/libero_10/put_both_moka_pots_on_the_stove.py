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


class L10K8PutBothMokaPotsOnTheStove(LwTaskBase):
    task_name: str = 'L10K8PutBothMokaPotsOnTheStove'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["stovetop", "mokapot_1", "mokapot_2"]
    removable_fixtures: list[str] = ["mokapot_1", "mokapot_2"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put both moka pots on the stove."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.mokapot_1 = self.register_fixture_ref("mokapot_1", dict(id="mokapot_1"))
        self.mokapot_2 = self.register_fixture_ref("mokapot_2", dict(id="mokapot_2"))
        self.init_robot_base_ref = self.dining_table

    def _check_success(self, env):
        '''
        Check if the moka pot is placed on the plate.
        '''
        is_gripper_obj_far = OU.gripper_obj_far(env, self.mokapot_1.name) & OU.gripper_obj_far(env, self.mokapot_2.name)
        pot_on_stove = OU.check_place_obj1_on_obj2(env, self.mokapot_1, self.stove, th_xy_dist=0.5) & OU.check_place_obj1_on_obj2(env, self.mokapot_2, self.stove, th_xy_dist=0.5)
        return is_gripper_obj_far & pot_on_stove
