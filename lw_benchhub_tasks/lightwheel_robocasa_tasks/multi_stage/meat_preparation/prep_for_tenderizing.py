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


class PrepForTenderizing(LwTaskBase):
    """
    Prep For Tenderizing: composite task for Meat Preparation activity.

    Simulates the task of preparing meat for tenderizing.

    Steps:
        Retrieve a rolling pin from the cabinet and place it next to the meat on
        the cutting board to prepare for tenderizing.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the rolling pin
            is picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "PrepForTenderizing"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.5, 0.5))
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Retrieve a rolling pin from the cabinet and place it next to the "
            "meat on the cutting board to prepare for tenderizing."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="meat",
                graspable=True,
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    size=(0.1, 0.1),
                    ensure_object_boundary_in_range=False,
                    pos=(0, -0.3),
                    try_to_place_in="cutting_board",
                ),
            )
        )

        cfgs.append(
            dict(
                name="rolling_pin",
                obj_groups="rolling_pin",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.02),
                    pos=(0, 0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        return (
            OU.check_obj_in_receptacle(env, "rolling_pin", "meat_container")
            & OU.gripper_obj_far(env, obj_name="meat_container")
            & OU.check_obj_in_receptacle(env, "meat", "meat_container")
        )
