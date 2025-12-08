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


class ArrangeTea(LwTaskBase):
    """
    Arrange Tea: composite task for Brewing activity.

    Simulates the task of arranging tea.

    Steps:
        Take the kettle from the counter and place it on the tray.
        Take the mug from the cabinet and place it on the tray.
        Close the cabinet doors.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "ArrangeTea"
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        # use a double door cabinet so that area below is large enough to initialize all the objects
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        # set the size argument to sample a large enough counter region
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.cab

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the kettle from the counter and place it on the tray. "
            "Then pick the mug from the cabinet and place it on the tray. "
            "Then close the cabinet doors."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("mug"),
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.10),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("kettle"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                    offset=(0.1, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("tray"),
                placement=dict(
                    fixture=self.counter,
                    size=(0.7, 0.7),
                    pos=("ref", -0.6),
                    offset=(-0.1, 0.0),
                    sample_region_kwargs=dict(ref=self.cab, top_size=(0.6, 0.4)),
                ),
            )
        )

        return cfgs

    def _check_door_closed(self, env):
        door_state = self.cab.get_door_state(env=env)

        success = torch.ones(list(door_state.values())[0].shape[0], device=env.device)
        for joint_p in door_state.values():
            success = torch.logical_and(success, joint_p < 0.05)

        return success

    def _check_success(self, env):
        obj1_container_contact = OU.check_obj_in_receptacle(env, "obj", "container")
        obj2_container_contact = OU.check_obj_in_receptacle(env, "obj2", "container")
        cab_closed = self.cab.is_closed(env=env)
        gripper_obj_far = OU.gripper_obj_far(
            env
        )  # no need to check all gripper objs far bc all objs in the same place

        return obj1_container_contact & obj2_container_contact & gripper_obj_far & cab_closed
