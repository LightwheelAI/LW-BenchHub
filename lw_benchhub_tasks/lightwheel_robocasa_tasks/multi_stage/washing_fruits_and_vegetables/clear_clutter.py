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


class ClearClutter(LwTaskBase):
    """
    Clear Clutter: composite task for Washing Fruits And Vegetables activity.

    Simulates the task of washing fruits and vegetables.

    Steps:
        Pick up the fruits and vegetables and place them in the sink turn on the
        sink to wash them. Then, turn the sink off, put them in the tray.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "ClearClutter"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        # sample large enough region to place the food items
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.sink
        self.num_food = self.rng.choice([1, 2])
        self.num_unwashable = self.rng.choice([1, 2])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the fruits and vegetables and place them in the sink. "
            "Turn on the sink faucet to wash them. Then turn the sink off and put them in the tray."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        self.food_washed = torch.tensor([False], device=env.device).repeat(env.num_envs)
        self.washed_time = torch.tensor([0], device=env.device).repeat(env.num_envs)
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        for i in range(self.num_food):
            cfgs.append(
                dict(
                    name=f"obj_{i}",
                    obj_groups=["vegetable", "fruit"],
                    graspable=True,
                    washable=True,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.40, 0.40),
                        pos=("ref", -1.0),
                    ),
                )
            )

        for i in range(self.num_unwashable):
            cfgs.append(
                dict(
                    name=f"unwashable_obj_{i}",
                    obj_groups="all",
                    # make the object not washable and make sure there aren't 2 trays
                    exclude_obj_groups=["food", "tray"],
                    washable=False,
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.40, 0.40),
                        pos=("ref", -1.0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="receptacle",
                obj_groups="tray",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.6)
                    ),
                    size=(0.6, 0.8),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        food_in_sink = torch.stack([
            OU.obj_inside_of(env, f"obj_{i}", self.sink)
            for i in range(self.num_food)
        ]).all(dim=0)

        unwashables_not_in_sink = torch.stack([
            ~ OU.obj_inside_of(env, f"unwashable_obj_{i}", self.sink)
            for i in range(self.num_unwashable)
        ]).all(dim=0)

        water_on = self.sink.get_handle_state(env=env)["water_on"]
        # make sure the food has been washed for suffient time (10 steps)
        for env_id in range(env.num_envs):
            if food_in_sink[env_id] & unwashables_not_in_sink[env_id] & water_on[env_id]:
                self.washed_time[env_id] += 1
                self.food_washed[env_id] = self.washed_time[env_id] > 10
            else:
                self.washed_time[env_id] = 0

        food_in_tray = torch.stack([
            OU.check_obj_in_receptacle(env, f"obj_{i}", "receptacle")
            for i in range(self.num_food)
        ]).all(dim=0)

        unwashables_not_in_tray = torch.stack([
            ~ OU.check_obj_in_receptacle(env, f"unwashable_obj_{i}", "receptacle")
            for i in range(self.num_unwashable)
        ]).all(dim=0)

        return self.food_washed & food_in_tray & unwashables_not_in_tray & ~ water_on
