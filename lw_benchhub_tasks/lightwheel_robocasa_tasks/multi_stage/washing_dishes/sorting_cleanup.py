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


class SortingCleanup(LwTaskBase):
    """
    Sorting Cleanup: composite task for Washing Dishes activity.

    Simulates the task of sorting and cleaning dishes.

    Steps:
        Pick the mug and place it in the sink. Pick the bowl and place it in the
        cabinet and then close the cabinet.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "SortingCleanup"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.sink)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.5, 0.5))
        )

        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the mug and place it in the sink. "
            "Pick the bowl and place it in the cabinet and then close the cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        # not fully open since it may come in contact with eef
        self.cab.set_door_state(min=0.5, max=0.6, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="mug",
                obj_groups=("mug"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.4, 0.4),
                    pos=("ref", -1),
                ),
            )
        )
        cfgs.append(
            dict(
                name="bowl",
                obj_groups=("bowl"),
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                        # large enough region to sample the bowl
                        top_size=(0.5, 0.5),
                    ),
                    size=(0.7, 0.7),
                    pos=("ref", -1),
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
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=(0, 1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        mug_in_sink = OU.obj_inside_of(env, "mug", self.sink)
        bowl_in_cab = OU.obj_inside_of(env, "bowl", self.cab)
        closed = torch.tensor([True], device=env.device).repeat(env.num_envs)
        door_state = self.cab.get_door_state(env=env)

        for env_id in range(env.num_envs):
            for joint_p in list(door_state.values()):
                if joint_p[env_id] > 0.01:
                    closed[env_id] = False
                    break

        return mug_in_sink & bowl_in_cab & closed & OU.gripper_obj_far(env, "mug") & OU.gripper_obj_far(env, "bowl")
