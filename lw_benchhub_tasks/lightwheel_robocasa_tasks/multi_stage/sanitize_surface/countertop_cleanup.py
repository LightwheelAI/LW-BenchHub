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

import numpy as np

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class CountertopCleanup(LwTaskBase):
    """
    Countertop Cleanup: composite task for Sanitize Surface activity.

    Simulates the task of cleaning the countertop.

    Steps:
        Pick the fruit and vegetable from the counter and place it in the cabinet.
        Then, open the drawer and pick the cleaner and sponge from the drawer and
        place it on the counter.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER]
    task_name: str = "CountertopCleanup"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET))
        self.drawer = self.register_fixture_ref(
            "drawer", dict(id=FixtureType.TOP_DRAWER, ref=self.cab)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )

        self.init_robot_base_ref = self.drawer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the fruit and vegetable from the counter and place them in the cabinet. "
            "Then open the drawer and pick the cleaner and sponge from the drawer and place them on the counter."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """

        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        # objects appear on different sides
        direction = self.rng.choice([1.0, -1.0])

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("spray", "soap", "soap_dispenser"),
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.3),
                    pos=(-1.0 * direction, -0.5),
                    rotation=np.pi / 2,
                    rotation_axis="x",
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="sponge",
                graspable=True,
                placement=dict(
                    fixture=self.drawer,
                    size=(0.3, 0.3),
                    pos=(1.0 * direction, -0.5),
                ),
            )
        ),

        cfgs.append(
            dict(
                name="obj3",
                obj_groups=("fruit"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj4",
                obj_groups=("vegetable"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=(0.0, -1.0),
                    offset=(0.0, 0.10),
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
        gripper_obj_far = OU.gripper_obj_far(env) & OU.gripper_obj_far(env, "obj3")
        objs_on_counter = OU.check_obj_fixture_contact(
            env, "obj", self.counter
        ) & OU.check_obj_fixture_contact(env, "obj2", self.counter)
        objs_inside_cab = OU.obj_inside_of(env, "obj3", self.cab) & OU.obj_inside_of(
            env, "obj4", self.cab
        )

        return gripper_obj_far & objs_inside_cab & objs_on_counter
