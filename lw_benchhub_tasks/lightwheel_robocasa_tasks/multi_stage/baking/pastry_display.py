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


class PastryDisplay(LwTaskBase):
    """
    Pastry Display: composite task for Baking activity.

    Simulates the task of displaying pastries.

    Steps:
        Place the pastries on the plates.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "PastryDisplay"
    EXCLUDE_LAYOUTS: list = [8, 10]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the pastries on the plates."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="plate",
                graspable=False,
                washable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="plate",
                graspable=False,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        # use offserts and to make it easier to initialize pastry1 and pastry2
        cfgs.append(
            dict(
                name="pastry1",
                obj_groups="pastry",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -0.2),
                    offset=(0.1, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pastry2",
                obj_groups="pastry",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -0.2),
                    offset=(-0.1, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_pastry1_far = OU.gripper_obj_far(env, obj_name="pastry1")
        gripper_pastry2_far = OU.gripper_obj_far(env, obj_name="pastry2")
        pastry1_in_receptacle1 = OU.check_obj_in_receptacle(
            env, "pastry1", "receptacle1"
        )
        pastry1_in_receptacle2 = OU.check_obj_in_receptacle(
            env, "pastry1", "receptacle2"
        )
        pastry2_in_receptacle1 = OU.check_obj_in_receptacle(
            env, "pastry2", "receptacle1"
        )
        pastry2_in_receptacle2 = OU.check_obj_in_receptacle(
            env, "pastry2", "receptacle2"
        )

        pastrys_placed = (pastry1_in_receptacle1 & pastry2_in_receptacle2) | \
                         (pastry1_in_receptacle2 & pastry2_in_receptacle1)
        return gripper_pastry1_far & gripper_pastry2_far & pastrys_placed
