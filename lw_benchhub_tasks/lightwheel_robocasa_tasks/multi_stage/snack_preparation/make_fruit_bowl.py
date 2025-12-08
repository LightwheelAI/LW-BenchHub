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


class MakeFruitBowl(LwTaskBase):
    """
    Make Fruit Bowl: composite task for Snack Preparation activity.

    Simulates the preparation of a fruit bowl snack.

    Steps:
        Pick the fruit from the cabinet and place them in the bowl.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the fruit are
            picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "MakeFruitBowl"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        fruit1_name = self.get_obj_lang("fruit1")
        fruit2_name = self.get_obj_lang("fruit2")
        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {fruit1_name} and {fruit2_name} from the cabinet and place them into the bowl. "
            "Then close the cabinet."
        )

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
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    size=(1, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit1",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="fruit2",
                obj_groups="fruit",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0.5, -1.0),
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
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        fruit1_in_bowl = OU.check_obj_in_receptacle(env, "fruit1", "bowl")
        fruit2_in_bowl = OU.check_obj_in_receptacle(env, "fruit2", "bowl")

        door_state = self.cab.get_door_state(env=env)

        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        door_closed = (joint_positions <= 0.01).all(dim=0)  # (num_envs,)

        return fruit1_in_bowl & fruit2_in_bowl & door_closed
