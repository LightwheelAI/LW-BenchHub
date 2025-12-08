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


class ArrangeBreadBasket(LwTaskBase):
    """
    Arrange Bread Basket: composite task for Setting The Table activity.

    Simulates the task of arranging the bread basket.

    Steps:
        Pick the bread from the cabinet and place it in the bowl. Then move the bowl
        to the dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bread is
            picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOOL]
    task_name: str = "ArrangeBreadBasket"
    EXCLUDE_LAYOUTS = LwTaskBase.STOOL_EXCLUDED_LAYOUT
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter_small = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        bread_name = OU.get_obj_lang(self, "bread")
        ep_meta["lang"] = (
            f"Open the cabinet, pick up the {bread_name} from the cabinet and place it in the bowl. "
            "Then move the bowl to the dining counter."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.set_door_state(min=0.0, max=0.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter_small,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining2",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="bowl")
        bread_in_bowl = OU.check_obj_in_receptacle(env, "bread", "bowl")
        bowl_on_counter = OU.check_obj_fixture_contact(env, "bowl", self.dining_table)

        return gripper_obj_far & bread_in_bowl & bowl_on_counter
