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


class DrainVeggies(LwTaskBase):
    """
    Drain Veggies: composite task for Washing Fruits And Vegetables activity.

    Simulates the task of draining washed vegetables.

    Steps:
        Dump the vegetables from the pot into the sink. Then turn on the sink and
        wash the vegetables. Then turn off the sink and put the vegetables back in
        the pot.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "DrainVeggies"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_name = OU.get_obj_lang(self, "obj")
        ep_meta["lang"] = (
            f"Dump the {food_name} from the pot into the sink. Then turn on the water and wash the {food_name}. "
            f"Then turn off the water and put the {food_name} back in the pot."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        # reset task progress variables
        self.vegetables_washed = torch.tensor([False], device=env.device).repeat(env.num_envs)
        self.washed_time = torch.tensor([0], device=env.device).repeat(env.num_envs)
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"obj",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.6, 0.6)
                    ),
                    try_to_place_in="pot",
                    size=(0.6, 0.4),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        vegetables_in_sink = OU.obj_inside_of(env, f"obj", self.sink)
        water_on = self.sink.get_handle_state(env=env)["water_on"]
        # make sure the vegetables are washed for at least 10 steps
        for env_id in range(env.num_envs):
            if vegetables_in_sink[env_id] & water_on[env_id]:
                self.washed_time[env_id] += 1
                self.vegetables_washed[env_id] = self.washed_time[env_id] > 10
            else:
                self.washed_time[env_id] = 0

        vegetables_in_pot = OU.check_obj_in_receptacle(env, f"obj", "obj_container")

        return self.vegetables_washed & vegetables_in_pot & ~ water_on
