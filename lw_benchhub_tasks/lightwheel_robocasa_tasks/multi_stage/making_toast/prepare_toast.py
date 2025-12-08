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


class PrepareToast(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.TOASTER]

    """
    Prepare Toast: composite task for Making Toast activity.

    Simulates the task of preparing toast.

    Steps:
        Open the cabinet, pick the bread, place it on the cutting board, pick the jam,
        place it on the counter, and close the cabinet
    """

    task_name: str = "PrepareToast"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=FixtureType.TOASTER)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the bread from the cabinet, place it on the cutting board, "
            "pick the jam, place it on the counter, and close the cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)
        self.cab.set_door_state(min=0.9, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.5, 0.5),
                    pos=(0.0, -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="obj2",
                obj_groups="jam",
                placement=dict(
                    fixture=self.cab,
                    size=(0.3, 0.15),
                    pos=(0.0, -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj3",
                obj_groups="knife",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.3, 0.3),
                    pos=(0.0, 0.0),
                    ensure_object_boundary_in_range=False,
                    offset=(-0.05, 0.05),
                ),
            )
        )

        return cfgs

    def _check_door_closed(self, env):
        door_state = self.cab.get_door_state(env=env)

        success = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for joint_p in door_state.values():
            success &= ~(joint_p > 0.05)
        return success

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env)
        jam_on_counter = OU.check_obj_fixture_contact(env, "obj2", self.counter)
        bread_on_cutting_board = OU.check_obj_in_receptacle(env, "obj", "container")
        cutting_board_on_counter = OU.check_obj_fixture_contact(
            env, "container", self.counter
        )
        cabinet_closed = self._check_door_closed(env)
        return jam_on_counter & gripper_obj_far & bread_on_cutting_board & cutting_board_on_counter & cabinet_closed
