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


class OrganizeBakingIngredients(LwTaskBase):
    """
    Organize Baking Ingredients: composite task for Baking activity.

    Simulates the task of organizing baking ingredients.

    Steps:
        Place the eggs and milk near the bowl on the counter.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "OrganizeBakingIngredients"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the eggs and milk next to the bowl."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.50, 0.50),
                    pos=(0.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="egg1",
                obj_groups="egg",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.3, 0.3),
                    pos=(-1.0, -0.4),
                ),
            )
        )

        cfgs.append(
            dict(
                name="egg2",
                obj_groups="egg",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.3, 0.3),
                    pos=(-1.0, -0.4),
                    offset=(0.2, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="milk",
                obj_groups="milk",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.5, 0.5),
                    pos=(1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):

        th = 0.2

        bowl_pos = env.scene.rigid_objects["bowl"].data.body_com_pos_w[:, 0, :]
        egg1_pos = env.scene.rigid_objects["egg1"].data.body_com_pos_w[:, 0, :]
        egg2_pos = env.scene.rigid_objects["egg2"].data.body_com_pos_w[:, 0, :]
        milk_pos = env.scene.rigid_objects["milk"].data.body_com_pos_w[:, 0, :]

        # check if the objects are near the bowl
        bowl_egg1_close = torch.norm(bowl_pos - egg1_pos, dim=-1) < th
        bowl_egg2_close = torch.norm(bowl_pos - egg2_pos, dim=-1) < th
        bowl_milk_close = torch.norm(bowl_pos - milk_pos, dim=-1) < th

        gripper_far = OU.gripper_obj_far(env, "milk") & OU.gripper_obj_far(env, "bowl") & OU.gripper_obj_far(env, "egg1") & OU.gripper_obj_far(env, "egg2")

        return bowl_egg1_close & bowl_egg2_close & bowl_milk_close & gripper_far
