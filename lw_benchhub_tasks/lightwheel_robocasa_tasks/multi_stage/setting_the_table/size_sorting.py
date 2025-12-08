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


class SizeSorting(LwTaskBase):

    """
    Size Sorting: composite task for Setting The Table activity.

    Simulates the task of stacking objects by size.

    Steps:
        Stack the objects from largest to smallest.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER]
    task_name: str = "SizeSorting"
    # exclude layout 9 because objects sometime initilize in corner area which is unreachable
    EXCLUDE_LAYOUTS: list[int] = [9]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        # sample a large enough counter for multiple stackable categories
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(1, 0.4))
        )
        self.init_robot_base_ref = self.counter
        if "object_cfgs" in scene._ep_meta:
            object_cfgs = scene._ep_meta["object_cfgs"]
            self.num_objs = len(
                [cfg for cfg in object_cfgs if cfg["name"].startswith("obj_")]
            )
        else:
            self.num_objs = self.rng.choice([2, 3])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        stackable_cat = OU.get_obj_lang(self, "obj_0")
        ep_meta["lang"] = f"Stack the {stackable_cat}s from largest to smallest."
        return ep_meta

    def _setup_scene(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        stack_cat = self.rng.choice(["cup", "bowl"])
        scale = 0.80
        # pass in object scale to the config to make the objects smaller and thus stackable
        if stack_cat == "cup":
            top_size = (0.6, 0.2)
        elif stack_cat == "bowl":
            top_size = (0.6, 0.4)
        else:
            raise ValueError
        for i in range(self.num_objs):
            obj_cfg = dict(
                name=f"obj_{i}",
                obj_groups=stack_cat,
                object_scale=scale**i,
                placement=dict(
                    fixture=self.counter,
                    ref_obj="obj_0",
                    sample_region_kwargs=dict(top_size=top_size),
                    size=top_size,
                    pos=(None, -1.0),
                    offset=(0.0, 0.0),
                ),
            )
            if i == 0:
                obj_cfg["init_robot_here"] = True
            cfgs.append(obj_cfg)

        return cfgs

    def _check_success(self, env):

        objs_stacked_inorder = torch.stack([
            OU.check_obj_in_receptacle(env, f"obj_{i}", f"obj_{i-1}")
            for i in range(1, self.num_objs)
        ], dim=0).all(dim=0)
        res = objs_stacked_inorder & OU.gripper_obj_far(env, "obj_0") & OU.gripper_obj_far(env, "obj_1")
        if self.num_objs == 3:
            res = res & OU.gripper_obj_far(env, "obj_2")
        return res
