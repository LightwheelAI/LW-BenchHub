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


class ServeSteak(LwTaskBase):
    """
    Serve Steak: composite task for Serving Food activity.

    Simulates the task of serving steak.

    Steps:
        Pick up the pan with the steak in it and place it on the dining table.
        Then, place the steak on the plate.

    Restricted to layouts which have a dining table (long counter area with
    stools).
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOOL, FixtureType.STOVE]
    task_name: str = "ServeSteak"
    EXCLUDE_LAYOUTS = LwTaskBase.STOOL_EXCLUDED_LAYOUT

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.init_robot_base_ref = self.stove
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the pan with the steak in it and place it on the dining table. "
            "Then place the steak on the plate."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="steak",
                placement=dict(
                    fixture=self.stove,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="pan",
                ),
            )
        )
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.30, 0.35),
                    pos=(2.0, -1.0),
                    offset=(0.02, -0.04)
                ),
            )
        )
        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups=("mug", "cup"),
                placement=dict(
                    fixture=self.dining_table,
                    sample_region_kwargs=dict(
                        ref=self.stool,
                    ),
                    size=(0.50, 0.25),
                    pos=(2.0, -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        steak_on_plate = OU.check_obj_in_receptacle(env, "obj", "plate")
        pan_on_table = OU.check_obj_fixture_contact(
            env, "obj_container", self.dining_table
        )
        return steak_on_plate & pan_on_table & OU.gripper_obj_far(env)
