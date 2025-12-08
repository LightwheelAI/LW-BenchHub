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

import torch

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class CandleCleanup(LwTaskBase):
    """
    Candle Cleanup: composite task for Clearing Table activity.

    Simulates the process of efficiently clearing the dining table decorations.

    Steps:
        Pick the decorations from the dining table and place it in the open cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the decorations
            are picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOOL]
    task_name: str = "CandleCleanup"
    EXCLUDE_LAYOUTS: list = LwTaskBase.DINING_COUNTER_EXCLUDED_LAYOUTS
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        # dining table is a sufficiently large counter where there are chairs nearby
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_name_1} and {obj_name_2} from the dining table and place them in the open cabinet,\nthen close the cabinet door"
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
                name="obj1",
                obj_groups="candle",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.30),
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    pos=(0, -1),
                    offset=(-0.05, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="candle",
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0, -1),
                    offset=(0.05, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1.0, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        obj1_inside_cab = OU.obj_inside_of(env, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(env, "obj2", self.cab)

        door_state = self.cab.get_door_state(env=env)

        # Collect tensor of all joint positions
        joint_positions = torch.stack(list(door_state.values()), dim=0)  # shape: (num_joints, num_envs)

        # Check if all joint positions are less than or equal to 0.01 (door closed)
        door_closed = (joint_positions <= 0.01).all(dim=0)

        return door_closed & obj1_inside_cab & obj2_inside_cab
