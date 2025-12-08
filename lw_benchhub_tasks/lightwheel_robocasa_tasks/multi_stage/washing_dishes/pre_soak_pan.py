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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class PreSoakPan(LwTaskBase):
    """
    Pre Soak Pan: composite task for Washing Dishes activity.

    Simulates the task of pre-soaking a pan.

    Steps:
        Pick the pan and sponge and place them into the sink. Then turn on the sink.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "PreSoakPan"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the pan and sponge and place them into the sink. Then turn on the water."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.sink.set_handle_state(mode="off", env=env)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj1",
                obj_groups=("pan"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                        # make sure sampled counter region is large enough to place the pan
                        top_size=(0.6, 0.4),
                    ),
                    size=(0.35, 0.55),
                    pos=("ref", -1.0),
                ),
                # make sure the sampled pan would fit in the sink basin
                max_size=(0.35, 0.45, None),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("sponge"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.sink, loc="left_right"),
                    size=(0.2, 0.3),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.05),
                ),
            )
        )

        return cfgs

    def _check_pan_in_sink(self, env):
        return OU.obj_inside_of(env, "obj1", self.sink)

    def _check_sponge_in_sink(self, env):
        return OU.obj_inside_of(env, "obj2", self.sink)

    def _check_success(self, env):
        handle_state = self.sink.get_handle_state(env=env)
        water_on = handle_state["water_on"]
        pan_in_sink = OU.obj_inside_of(env, "obj1", self.sink, partial_check=False)
        sponge_in_sink = OU.obj_inside_of(env, "obj2", self.sink, partial_check=False)
        return (
            water_on
            & pan_in_sink
            & sponge_in_sink
            & OU.gripper_obj_far(env, "obj1")
            & OU.gripper_obj_far(env, "obj2")
        )
