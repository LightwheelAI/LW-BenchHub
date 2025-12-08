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


class RestockBowls(LwTaskBase):
    """
    Restock Bowls: composite task for Restocking Supplies activity.

    Simulates the task of restocking bowls.

    Steps:
        Restock two bowls from the counter to the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet to which the bowls are
            restocked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "RestockBowls"
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

        obj_name_1 = self.get_obj_lang("obj1")
        obj_name_2 = self.get_obj_lang("obj2")

        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {obj_name_1} and the {obj_name_2} from the counter and place it in the cabinet directly in front. "
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
        cfgs.append(
            dict(
                name="obj1",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    size=(0.50, 0.50),
                    pos=(-0.5, -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    size=(0.50, 0.50),
                    pos=(0.5, -1),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        obj1_inside_cab = OU.obj_inside_of(env, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(env, "obj2", self.cab)

        door_state = self.cab.get_door_state(env=env)
        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        door_closed = (joint_positions <= 0.01).all(dim=0)  # (num_envs,)
        return obj1_inside_cab & obj2_inside_cab & door_closed
