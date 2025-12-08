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


class DryDishes(LwTaskBase):
    """
    Dry Dishes: composite task for Washing Dishes activity.

    Simulates the task of drying dishes.

    Steps:
        Pick the cup and bowl from the sink and place them on the counter for drying.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "DryDishes"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter",
            dict(
                id=FixtureType.COUNTER,
                ref=self.sink,
            ),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the cup and bowl from the sink and place them on the counter for drying."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # sample a random back corner for the cup to be placed on
        cup_pos = self.rng.choice([(1.0, 1.0), (-1.0, 1.0)])
        cfgs.append(
            dict(
                name="obj1",
                obj_groups=("cup"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    # hard code the cup to be in corners so that the cup and bowl fit in the sink
                    size=(0.1, 0.1),
                    # offset=(0.25, 0.25)
                    pos=list(cup_pos),  # turn into list to allow saving
                ),
            )
        )
        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("bowl"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    # place the bowl in the middle of the sink and turn of ensure_object_boundary_in_range
                    # otherwise it becomes difficult to initialize since the bowl is so big
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
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
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        objs_on_counter = torch.logical_and(
            OU.check_contact(env, self.objects["obj1"], self.counter),
            OU.check_contact(env, self.objects["obj2"], self.counter),
        )
        gripper_objs_far = torch.logical_and(
            OU.gripper_obj_far(env, "obj1"),
            OU.gripper_obj_far(env, "obj2"),
        )
        return objs_on_counter & gripper_objs_far
