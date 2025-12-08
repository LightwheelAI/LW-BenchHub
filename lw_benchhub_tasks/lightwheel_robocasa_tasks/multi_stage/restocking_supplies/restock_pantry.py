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


class RestockPantry(LwTaskBase):
    """
    Restock Pantry: composite task for Restocking Supplies activity.

    Simulates the task of organizing cans when restocking them.

    Steps:
        Pick the cans from the counter and place them on the side of the cabinet
        that already has a can.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "RestockPantry"
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the cans from the counter and place them in their designated side in the cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(min=1.0, max=1.0, env=env)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.50, 0.30),
                    pos=("ref", -1),
                ),
            )
        )

        # randomize the side of the cabinet that already has a can
        side = int(self.rng.choice([-1, 1]))

        cfgs.append(
            dict(
                name="cab_obj1",
                obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.2, 0.30),
                    pos=(side, -0.3),
                ),
            )
        )

        cfgs.append(
            dict(
                name="cab_obj2",
                obj_groups="all",
                exclude_obj_groups="canned_food",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.2, 0.30),
                    pos=(side * -1, 0.3),
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
                    offset=(0.0, -0.05),
                ),
            )
        )

        return cfgs

    def _close_to_cab_cans(self, env, obj_name, ratio=2):
        """
        Check if the object is closer to the cabinet cans than the other object

        Args:
            obj_name (str): name of the object

            ratio (float): ratio of the distance between the object and the cabinet cans to the distance
                between the object and the other object

        Returns:
            bool: True if the object is closer to the cabinet cans than the other object, False otherwise
        """
        obj = self.objects[obj_name]
        can = self.objects["cab_obj1"]
        other_obj = self.objects["cab_obj2"]
        obj_pos = env.scene.rigid_objects[obj.task_name].data.body_com_pos_w[:, 0, :]
        can_pos = env.scene.rigid_objects[can.task_name].data.body_com_pos_w[:, 0, :]
        other_obj_pos = env.scene.rigid_objects[other_obj.task_name].data.body_com_pos_w[:, 0, :]

        can_dist = torch.linalg.norm(obj_pos - can_pos)
        other_dist = torch.linalg.norm(other_obj_pos - obj_pos)

        return can_dist * ratio < other_dist

    def _check_obj_in_cab(self, env, obj_name):
        return OU.obj_inside_of(env, obj_name, self.cab) & self._close_to_cab_cans(
            env, obj_name
        )

    def _check_success(self, env):
        obj1_inside_cab = OU.obj_inside_of(env, "obj1", self.cab)
        obj2_inside_cab = OU.obj_inside_of(env, "obj2", self.cab)

        cans_close = self._close_to_cab_cans(env, "obj1") & self._close_to_cab_cans(env, "obj2")

        gripper_obj_far = OU.gripper_obj_far(env, "obj1") & OU.gripper_obj_far(
            env, "obj2"
        )

        return obj1_inside_cab & obj2_inside_cab & cans_close & gripper_obj_far
