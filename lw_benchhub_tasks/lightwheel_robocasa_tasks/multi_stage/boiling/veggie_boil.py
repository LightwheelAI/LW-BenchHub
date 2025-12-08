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


class VeggieBoil(LwTaskBase):
    """
    Veggie Boil: composite task for Boiling activity.

    Simulates the process of boiling vegetables.

    Steps:
        Take the pot from the counter and place it in the sink. Turn on the sink and
        let the pot fill up with water. Turn the sink off and move the pot to the
        stove. Turn on the stove and place the food in the pot for boiling.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK, FixtureType.STOVE]
    task_name: str = "VeggieBoil"
    pot_filled: bool = False
    filled_time: int = 0

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter_sink = self.register_fixture_ref(
            "counter_sink", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter_stove = self.register_fixture_ref(
            "counter_stove", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        food_name = self.get_obj_lang("food")
        ep_meta["lang"] = (
            "Pick up the pot and place it in the sink. "
            "Then turn on the sink faucet and let the pot fill up with water. "
            "Then turn the sink faucet off and move the pot to the stove. "
            f"Lastly, turn on the stove and place the {food_name} in the pot for boiling."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        self.pot_filled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.filled_time = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        self.sink.set_handle_state(mode="off", env=env)
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=f"pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.counter_sink,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="left_right", top_size=(0.5, 0.5)
                    ),
                    size=(0.3, 0.3),
                    pos=("ref", -0.55),
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"food",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="nn",
                    ),
                    size=(0.5, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter_stove,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                        loc="nn",
                    ),
                    size=(0.30, 0.30),
                    pos=(0, 1.0),
                ),
            )
        )

        # mitigate randomization errors
        if self.counter_sink != self.counter_stove:

            cfgs.append(
                dict(
                    name="distr_counter2",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.counter_sink,
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.30, 0.30),
                        pos=(0, 1.0),
                    ),
                )
            )

        return cfgs

    def _check_success(self, env):
        """
        Check if the task is successful.
        First check if the object is inside the sink and the water is on. Then make sure the pot is filled with water for enough
        time (10 timesteps). Once the pot is filled, check if the pot is on the stove and the food is in the pot.
        """
        pot_in_sink = OU.obj_inside_of(env, "pot", self.sink)
        water_on = self.sink.get_handle_state(env=env)["water_on"]

        should_fill_mask = pot_in_sink & water_on

        self.filled_time = torch.where(should_fill_mask, self.filled_time + 1, self.filled_time)

        self.pot_filled = self.filled_time > 10

        vegetables_in_pot = OU.check_obj_in_receptacle(env, "food", "pot")
        pot_locs = self.stove.check_obj_location_on_stove(env, "pot", need_knob_on=True)
        pot_on_stove = torch.tensor([(loc and len(loc) > 0 and loc[0] is not None) if loc else False for loc in pot_locs], device=env.device)
        return self.pot_filled & vegetables_in_pot & ~water_on & pot_on_stove
