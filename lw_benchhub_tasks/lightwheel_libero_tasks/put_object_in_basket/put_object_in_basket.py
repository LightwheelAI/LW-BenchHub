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

from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class PutObjectInBasket(LwTaskBase):

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.floor = self.register_fixture_ref("floor", dict(id=FixtureType.FLOOR_LAYOUT))
        self.init_robot_base_ref = self.floor
        self.basket = "basket"

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.basket,
                obj_groups=self.basket,
                placement=dict(
                    fixture=self.floor,
                    size=(0.3, 0.25),
                    pos=(0.1, 0.0),
                    # ensure_object_boundary_in_range=False,
                ),
                asset_name="Basket058.usd",
            )
        )

        return cfgs
