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


class FoodCleanup(LwTaskBase):
    """
    Food Cleanup: composite task for Clearing Table activity.

    Simulates the task of cleaning up various food items left on the counter.

    Steps:
        Pick the food items from the counter and place them in the cabinet.
        Then close the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the
            food items are picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER]
    task_name: str = "FoodCleanup"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab
        if "object_cfgs" in scene._ep_meta:
            object_cfgs = scene._ep_meta["object_cfgs"]
            self.num_food = len(
                [cfg for cfg in object_cfgs if cfg["name"].startswith("food")]
            )
        else:
            self.num_food = self.rng.choice([i for i in range(1, 4)])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        items = OU.get_obj_lang(self, "food0")
        for i in range(1, self.num_food):
            items += f", {OU.get_obj_lang(self, f'food{i}')}"
        ep_meta[
            "lang"
        ] = f"Pick the {items} from the counter and place {'them' if self.num_food > 1 else 'it'} in the cabinet. Then close the cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        for i in range(self.num_food):
            cfgs.append(
                dict(
                    name=f"food{i}",
                    obj_groups=["fruit", "vegetable", "boxed_food"],
                    graspable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.cab,
                        ),
                        size=(0.30, 0.30),
                        pos=("ref", -1.0),
                        offset=(0.05, 0.0),
                    ),
                )
            )

        return cfgs

    def _check_success(self, env):
        food_inside_cab = torch.stack([
            OU.obj_inside_of(env, f"food{i}", self.cab)
            for i in range(self.num_food)
        ], dim=0).all(dim=0)

        gripper_obj_far = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for i in range(self.num_food):
            gripper_obj_far = gripper_obj_far & OU.gripper_obj_far(env, obj_name=f"food{i}")

        cab_closed = self.cab.is_closed(env=env)

        return food_inside_cab & gripper_obj_far & cab_closed
