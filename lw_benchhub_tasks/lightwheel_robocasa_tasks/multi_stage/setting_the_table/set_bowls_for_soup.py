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


class SetBowlsForSoup(LwTaskBase):
    """
    Set Bowls For Soup: composite task for Setting The Table activity.

    Simulates the task of setting the table with bowls for soup.

    Steps:
        Move the bowls from the cabinet to the plates on the dining table.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bowls are
            picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER, FixtureType.STOOL]
    task_name: str = "SetBowlsForSoup"
    EXCLUDE_LAYOUTS = LwTaskBase.DINING_COUNTER_EXCLUDED_LAYOUTS + LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter_large = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Move the bowls from the cabinet to the plates on the dining table."
        return ep_meta

    def _setup_scene(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.set_door_state(min=0.0, max=0.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        # set both plates' ref as self.cab to put the plates in a similar area since dining table can be large
        cfgs.append(
            dict(
                name="plate1",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter_large,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.80, 0.50),
                    pos=(-0.3, -1.0),
                    offset=(-0.05, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="plate2",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter_large,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.80, 0.50),
                    pos=(0.3, -1.0),
                    offset=(0.05, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl1",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl2",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_bowl1_far = OU.gripper_obj_far(env, obj_name="bowl1")
        gripper_bowl2_far = OU.gripper_obj_far(env, obj_name="bowl2")
        bowl1_on_plate1 = OU.check_obj_in_receptacle(env, "bowl1", "plate1")
        bowl1_on_plate2 = OU.check_obj_in_receptacle(env, "bowl1", "plate2")
        bowl2_on_plate1 = OU.check_obj_in_receptacle(env, "bowl2", "plate1")
        bowl2_on_plate2 = OU.check_obj_in_receptacle(env, "bowl2", "plate2")

        bowls_set = (bowl1_on_plate1 & bowl2_on_plate2) | (
            bowl1_on_plate2 & bowl2_on_plate1
        )

        return gripper_bowl1_far & gripper_bowl2_far & bowls_set
