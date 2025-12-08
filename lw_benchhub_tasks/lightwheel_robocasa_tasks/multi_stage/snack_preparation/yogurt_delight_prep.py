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


class YogurtDelightPrep(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]

    """
    Yogurt Delight Prep: composite task for Snack Preparation activity.

    Simulates the preparation of a yogurt delight snack.

    Steps:
        Place the yogurt and fruit onto the counter.
    """

    task_name: str = "YogurtDelightPrep"
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        # want space for all the objects
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )

        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the yogurt and fruit onto the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="yogurt",
                obj_groups="yogurt",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.2),
                    pos=(0, -1),
                    offset=(0, -0.02),
                ),
            )
        )

        self.num_fruits = self.rng.choice([1, 2])
        for i in range(self.num_fruits):
            cfgs.append(
                dict(
                    name=f"fruit_{i}",
                    obj_groups="fruit",
                    placement=dict(
                        fixture=self.cab,
                        size=(0.5, 0.15),
                        pos=(0, -1),
                        offset=(0.05 * i, -0.02),
                    ),
                )
            )

        return cfgs

    def _check_success(self, env):
        fruits_on_counter = torch.stack([
            OU.check_obj_fixture_contact(env, f"fruit_{i}", self.counter)
            for i in range(self.num_fruits)
        ], dim=0).all(dim=0)

        yogurt_on_counter = OU.check_obj_fixture_contact(env, "yogurt", self.counter)
        items_on_counter = fruits_on_counter & yogurt_on_counter

        gripper_far_fruits = torch.stack([
            OU.gripper_obj_far(env, f"fruit_{i}")
            for i in range(self.num_fruits)
        ], dim=0).all(dim=0)

        gripper_far_yogurt = OU.gripper_obj_far(env, "yogurt")
        objs_far = gripper_far_fruits & gripper_far_yogurt
        return items_on_counter & objs_far
