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


class L90K2PutTheMiddleBlackBowlOnTopOfTheCabinet(LwTaskBase):
    task_name: str = "L90K2PutTheMiddleBlackBowlOnTopOfTheCabinet"
    enable_fixtures = ["storage_furniture"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.dining_table))
        self.obj_name = []
        self.init_robot_base_ref = self.dining_table

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        for cfg in self.object_cfgs:
            self.obj_name.append(cfg["name"])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the black bowl on the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.35),
                    margin=0.02,
                    pos=(0.0, -0.9),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl1",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.35),
                    margin=0.02,
                    pos=(0.0, -0.7),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"akita_black_bowl2",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.60, 0.5),
                    margin=0.02,
                    pos=(0.0, -0.4),
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"plate",
                obj_groups=["plate"],
                graspable=True,
                washable=True,
                object_scale=0.6,
                asset_name="Plate012.usd",
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.6, 0.35),
                    margin=0.02,
                    pos=(-0.3, -0.5)
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        result_tensor = torch.tensor([False] * env.num_envs, device=env.device)
        for i in range(env.num_envs):
            bowl1 = OU.point_in_fixture(OU.get_object_pos(env, "akita_black_bowl")[i], self.drawer, only_2d=True)
            bowl2 = OU.point_in_fixture(OU.get_object_pos(env, "akita_black_bowl1")[i], self.drawer, only_2d=True)
            bowl3 = OU.point_in_fixture(OU.get_object_pos(env, "akita_black_bowl2")[i], self.drawer, only_2d=True)
            result_tensor[i] = torch.as_tensor(bowl1 | bowl2 | bowl3, dtype=torch.bool, device=env.device)
        return result_tensor & self._gripper_obj_farfrom_objects(env)

    def _gripper_obj_farfrom_objects(self, env):
        gripper_far_tensor = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for obj_name in self.obj_name:
            gripper_far_tensor = gripper_far_tensor & OU.gripper_obj_far(env, obj_name)
        return gripper_far_tensor
