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


class MealPrepStaging(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOVE]

    """
    Meal Prep Staging: composite task for Frying activity.

    Simulates the task of cooking various ingredients.

    Steps:
        Place the pans on different burners, then place the vegetable
        and meat on different pans.
    """

    task_name: str = "MealPrepStaging"
    EXCLUDE_LAYOUTS: list[int] = [6]  # challenges with placing pans on table for this layout

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name_1 = OU.get_obj_lang(self, "vegetable")
        obj_name_2 = OU.get_obj_lang(self, "meat")
        ep_meta["lang"] = (
            "Place both pans onto different burners. "
            f"Then place the {obj_name_1} and the {obj_name_2} on different pans."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):

        cfgs = []

        cfgs.append(
            dict(
                name="pan1",
                obj_groups=("pan"),
                object_scale=0.7,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                    size=(0.4, 0.1),
                    pos=("ref", -0.1),
                    offset=(0.2, -0.05),
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="pan2",
                obj_groups=("pan"),
                object_scale=0.8,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="left_right",
                    ),
                    size=(0.4, 0.1),
                    pos=("ref", -0.2),
                    offset=(-0.15, -0.05),
                    rotation=0,
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups=("vegetable"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                    size=(0.4, 0.4),
                    pos=("ref", 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups=("meat"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.stove, loc="left_right"),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):

        vegetable_on_pan1 = OU.check_obj_in_receptacle(env, "vegetable", "pan1")
        vegetable_on_pan2 = OU.check_obj_in_receptacle(env, "vegetable", "pan2")
        meat_on_pan1 = OU.check_obj_in_receptacle(env, "meat", "pan1")
        meat_on_pan2 = OU.check_obj_in_receptacle(env, "meat", "pan2")

        food_on_pans = (vegetable_on_pan1 & meat_on_pan2) | (
            vegetable_on_pan2 & meat_on_pan1
        )

        pan1_loc = self.stove.check_obj_location_on_stove(env, "pan1", need_knob_on=False)
        pan2_loc = self.stove.check_obj_location_on_stove(env, "pan2", need_knob_on=False)

        pans_on_stove = torch.tensor([loc is not None for loc in pan1_loc], device=env.device) & \
            torch.tensor([loc is not None for loc in pan2_loc], device=env.device)
        pans_diff = torch.tensor([loc1 != loc2 for loc1, loc2 in zip(pan1_loc, pan2_loc)], device=env.device)

        return pans_on_stove & pans_diff & food_on_pans
