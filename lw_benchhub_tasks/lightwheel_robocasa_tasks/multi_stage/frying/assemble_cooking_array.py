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


class AssembleCookingArray(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOVE]

    """
    Assemble Cooking Array: composite task for Frying activity.

    Simulates the task of assembling ingredients for cooking.

    Steps:
        Move the meat onto the pan on the stove. Then, move the condiment and
        vegetable from the cabinet to the counter where the plate is.
    """

    task_name: str = "AssembleCookingArray"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.counter)
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        condiment_name = self.get_obj_lang("condiment")
        vegetable_name = self.get_obj_lang("vegetable")
        ep_meta["lang"] = (
            f"Move the {meat_name} onto the pan on the stove. "
            f"Then move the {condiment_name} and {vegetable_name} from the cabinet to the counter where the plate is."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.10),
                    pos=(-1.0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.10),
                    pos=(1.0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        meat_in_pan = OU.check_obj_in_receptacle(env, "meat", "pan", th=0.07)
        gripper_vegetable_far = OU.gripper_obj_far(env, obj_name="vegetable")
        gripper_condiment_far = OU.gripper_obj_far(env, obj_name="condiment")
        gripper_meat_far = OU.gripper_obj_far(env, obj_name="meat")
        vegetable_on_counter = OU.check_obj_fixture_contact(
            env, "vegetable", self.counter
        )
        condiment_on_counter = OU.check_obj_fixture_contact(
            env, "condiment", self.counter
        )
        return (
            meat_in_pan
            & gripper_vegetable_far
            & gripper_condiment_far
            & gripper_meat_far
            & vegetable_on_counter
            & condiment_on_counter
        )
