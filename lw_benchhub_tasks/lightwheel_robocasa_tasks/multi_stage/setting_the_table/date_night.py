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


class DateNight(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOOL]

    """
    Date Night: composite task for Setting The Table activity.

    Simulates the task of setting the table for a date night.

    Steps:
        Pick up the decoration and the alcohol from the cabinet and move them to the
        dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the decoration
            and alcohol are picked.
    """

    task_name: str = "DateNight"
    EXCLUDE_LAYOUTS = LwTaskBase.STOOL_EXCLUDED_LAYOUT
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        decoration_name = self.get_obj_lang("decoration")
        alcohol_name = self.get_obj_lang("alcohol")
        ep_meta[
            "lang"
        ] = f"Pick up the {decoration_name} and the {alcohol_name} from the cabinet and move them to the dining counter."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="decoration",
                obj_groups="decoration",
                object_scale=0.8,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(1.0, -1.0),
                    offset=(0.0, -0.04),
                ),
            )
        )

        cfgs.append(
            dict(
                name="alcohol",
                obj_groups="alcohol",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-1.0, -1.0),
                    offset=(0.0, -0.04),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="decoration")
        alcohol_on_dining_table = OU.check_obj_fixture_contact(
            env, "alcohol", self.dining_table
        )
        decoration_on_dining_table = OU.check_obj_fixture_contact(
            env, "decoration", self.dining_table
        )

        return gripper_obj_far & decoration_on_dining_table & alcohol_on_dining_table
