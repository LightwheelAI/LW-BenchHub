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


class DrinkwareConsolidation(LwTaskBase):
    """
    Drinkware Consolidation: composite task for Clearing Table activity.

    Simulates the task of clearing the island drinkware and placing them back in a cabinet.

    Steps:
        Pick the drinkware from the island and place them in the open cabinet.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET]
    task_name: str = "DrinkwareConsolidation"
    EXCLUDE_LAYOUTS: list = LwTaskBase.ISLAND_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.island = self.register_fixture_ref("island", dict(id=FixtureType.ISLAND))
        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=FixtureType.CABINET, ref=self.island),
        )
        self.init_robot_base_ref = self.island

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        objs_lang = self.get_obj_lang("obj_0")
        for i in range(1, self.num_drinkware):
            objs_lang += f", {self.get_obj_lang(f'obj_{i}')}"
        ep_meta[
            "lang"
        ] = f"Pick the {objs_lang} from the island and place {'them' if self.num_drinkware > 1 else 'it'} in the open cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def reset(self):
        super().reset()

    def _get_obj_cfgs(self):
        cfgs = []
        self.num_drinkware = self.rng.choice([1, 2, 3])

        for i in range(self.num_drinkware):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups=["drink"],
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.island,
                        sample_region_kwargs=dict(
                            ref=self.cab,
                        ),
                        size=(0.30, 0.40),
                        pos=("ref", -1.0),
                    ),
                )
            )

        return cfgs

    def _check_success(self, env):
        objs_in_cab = torch.stack([
            OU.obj_inside_of(env, f"obj_{i}", self.cab)
            for i in range(self.num_drinkware)
        ], dim=0).all(dim=0)

        gripper_obj_far = torch.stack([
            OU.gripper_obj_far(env, f"obj_{i}")
            for i in range(self.num_drinkware)
        ], dim=0).all(dim=0)
        return objs_in_cab & gripper_obj_far
