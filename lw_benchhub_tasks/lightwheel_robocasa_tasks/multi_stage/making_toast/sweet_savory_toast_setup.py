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


class SweetSavoryToastSetup(LwTaskBase):
    """
    Sweet Savory Toast Setup: composite task for Making Toast activity.

    Simulates the task of setting up the ingredients for making sweet and savory
    toast.

    Steps:
        Pick the avocado and bread from the counter and place it on the plate.
        Then pick the jam from the cabinet and place it next to the plate.
        Lastly, close the cabinet door.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to specify the cabinet where the jam is placed.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "SweetSavoryToastSetup"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.6))
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the avocado and bread from the counter and place them on the plate. "
            "Then pick the jam from the cabinet and place it next to the plate. "
            "Lastly, close the cabinet door."
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
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="avocado",
                obj_groups="avocado",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.6, 0.6),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bread",
                obj_groups="bread",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.6, 0.6),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="jam",
                obj_groups="jam",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.4),
                    pos=(0, -1.0),
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
        gripper_obj_far = OU.gripper_obj_far(env, "plate")
        jam_on_counter = OU.check_contact(env, self.objects["jam"], self.counter)
        food_on_plate = OU.check_obj_in_receptacle(
            env, "bread", "plate"
        ) & OU.check_obj_in_receptacle(env, "avocado", "plate")
        door_state = self.cab.get_door_state(env=env)

        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        closed = (joint_positions <= 0.01).all(dim=0)  # (num_envs,)

        return gripper_obj_far & food_on_plate & jam_on_counter & closed
