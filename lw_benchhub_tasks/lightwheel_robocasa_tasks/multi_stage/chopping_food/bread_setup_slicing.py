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

import re
import torch

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class BreadSetupSlicing(LwTaskBase):
    """
    Bread Setup Slicing: composite task for Chopping Food activity.

    Simulates the task of setting up bread for slicing.

    Steps:
        Place all breads on the cutting board.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER]
    task_name: str = "BreadSetupSlicing"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1.0, 0.4))
        )
        self.init_robot_base_ref = self.counter
        self.num_bread = None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        ep_meta["lang"] = f"Place all breads on the cutting board."

        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="cutting_board",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                    size=(1, 0.4),
                    pos=(-1.0, -0.5),
                ),
            )
        )

        self.num_bread = self.rng.choice([1, 2, 3])
        for i in range(self.num_bread):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups="bread",
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(top_size=(1.0, 0.4)),
                        size=(1, 0.4),
                        pos=(-0.5, -0.8),
                        offset=(i * 0.1, i * 0.06),
                        try_to_place_in="container",
                        try_to_place_in_kwargs=dict(
                            object_scale=0.8,
                        ),
                    ),
                )
            )

        return cfgs

    def find_bread_num(self, env):
        objs = env.scene.rigid_objects.keys()
        bread_list = []
        for obj in objs:
            match = re.match(r'^obj_(\d+)$', obj)
            if match:
                bread_list.append(obj)
        return len(bread_list)

    def _check_success(self, env):
        if self.num_bread is None:
            self.num_bread = self.find_bread_num(env)
        bread_on_board = torch.stack([
            OU.check_obj_in_receptacle(env, f"obj_{i}", "receptacle")
            for i in range(self.num_bread)
        ], dim=0).all(dim=0)

        return bread_on_board & OU.gripper_obj_far(env, "obj_0")
