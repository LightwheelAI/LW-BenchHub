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


class SetupJuicing(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]

    """
    Setup Juicing: composite task for Mixing And Blending activity.

    Simulates the task of setting up juicing.

    Steps:
        Open the cabinet, pick all fruits from the cabinet and place them on the
        counter.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to specify the cabinet to pick the fruits from.
    """

    task_name: str = "SetupJuicing"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the cabinet, pick all {self.num_fruits} fruits from the cabinet and place them on the counter close to the cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.close_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        self.num_fruits = self.rng.choice([2, 3, 4])
        cfgs = []
        for i in range(self.num_fruits):

            cfgs.append(
                dict(
                    name=f"obj{i}",
                    obj_groups="fruit",
                    graspable=True,
                    placement=dict(
                        fixture=self.cab,
                        size=(0.60, 0.10),
                        pos=(0, -1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                exclude_obj_groups="fruit",
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
        fruit_on_counter = torch.stack([
            OU.check_obj_fixture_contact(env, f"obj{i}", self.counter)
            for i in range(self.num_fruits)
        ], dim=0).all(dim=0)
        return fruit_on_counter & OU.gripper_obj_far(env, "obj1")
