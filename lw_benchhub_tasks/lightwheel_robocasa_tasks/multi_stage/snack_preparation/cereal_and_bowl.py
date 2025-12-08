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


class CerealAndBowl(LwTaskBase):
    """
    Cereal And Bowl: composite task for Snack Preparation activity.

    Simulates the preparation of a cereal snack.

    Steps:
        Pick the cereal and bowl from the cabinet and place them on the counter.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the bowl and
            cereal are picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "CerealAndBowl"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Open the cabinet. Pick the cereal and bowl from the cabinet and place them on the counter. "
            "Then close the cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.close_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        # make sure bowl and cereal show up on diff sides randomly
        direction = self.rng.choice([1.0, -1.0])

        cfgs.append(
            dict(
                name="cereal",
                obj_groups="boxed_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.30, 0.30),
                    pos=(1.0 * direction, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.50),
                    pos=(-1.0 * direction, -1.0),
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
                name="milk",
                obj_groups="milk",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.5, 0.30),
                    pos=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        box_on_counter = OU.check_obj_fixture_contact(env, "cereal", self.counter)
        bowl_on_counter = OU.check_obj_fixture_contact(env, "bowl", self.counter)

        door_state = self.cab.get_door_state(env=env)
        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        cabinet_closed = (joint_positions <= 0.01).all(dim=0)  # (num_envs,)

        return box_on_counter & bowl_on_counter & cabinet_closed
