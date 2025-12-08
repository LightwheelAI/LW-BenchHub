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


class DefrostByCategory(LwTaskBase):
    """
    Defrost By Category: composite task for Defrosting Food activity.

    Simulates the task of arranging and defrosting fruits and vegetables by type.

    Steps:
        Pick place all of the fruits in the running sink and all of the
        vegetables in a bowl on the counter.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "DefrostByCategory"
    EXCLUDE_LAYOUTS: list[int] = [8, 10]  # these layouts have placement issues

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "There is a mixed pile of frozen fruits and vegetables on the counter. "
            "Locate all the frozen vegetables and place the items in a bowl on the counter. "
            "Take all the frozen fruits and defrost them in a running sink."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # Place the four objects (two fruits, two vegetables)
        placements = list()
        # Making the four regions separate - might help with
        # initialization speed
        for i in range(4):
            placements.append(
                dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.5, 0.5)
                    ),
                    size=(0.3, 0.4),
                    pos=("ref", -1),
                )
            )
        self.rng.shuffle(placements)

        for i in range(4):
            cfgs.append(
                dict(
                    name="obj" + str(i),
                    obj_groups="fruit" if i <= 1 else "vegetable",
                    graspable=True,
                    placement=placements[i],
                )
            )

        # Bowl to place the vegetables in
        cfgs.append(
            dict(
                name="container",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.5, 0.5)
                    ),
                    size=(0.3, 0.4),
                    pos=("ref", -1),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        fruits_in_sink = OU.obj_inside_of(env, "obj0", self.sink) & OU.obj_inside_of(env, "obj1", self.sink)
        vegetables_in_bowl = OU.check_obj_in_receptacle(env, "obj2", "container") & OU.check_obj_in_receptacle(env, "obj3", "container")
        gripper_obj_far = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for i in range(4):
            gripper_obj_far = gripper_obj_far & OU.gripper_obj_far(env, obj_name="obj" + str(i))
        return gripper_obj_far & fruits_in_sink & vegetables_in_bowl
