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

from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class TestObjectsTask(LwTaskBase):
    """
    TestObjectsTask: composite task for testing objects.
    """
    task_name: str = "TestObjectsTask"
    reset_objects_enabled: bool = True

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink, size=(0.6, 0.4))
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Test objects"
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        if self.test_object_paths is not None:
            for obj_path in self.test_object_paths:
                cfgs.append(
                    dict(
                        name=obj_path.split('/')[-1].split('.')[0],
                        asset_name=obj_path,
                        load_from_local=True,
                        placement=dict(
                            fixture=self.counter,
                        ),
                    )
                )
        return cfgs

    def _check_success(self, env):
        return torch.tensor([False], device=env.device).repeat(env.num_envs)
