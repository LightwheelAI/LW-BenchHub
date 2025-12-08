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

from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class OpenStandMixerHead(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.STAND_MIXER]

    task_name: str = "OpenStandMixerHead"
    enable_fixtures: list[str] = ["stand_mixer"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stand_mixer = self.register_fixture_ref("stand_mixer", dict(id=FixtureType.STAND_MIXER))
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the stand mixer head."
        return ep_meta

    def _check_success(self, env):
        """
        Check if the stand mixer head is open.

        Returns:
            bool: True if the head is open, False otherwise.
        """
        return self.stand_mixer.get_state(env)["head"] > 0.99


class CloseStandMixerHead(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.STAND_MIXER]

    task_name: str = "CloseStandMixerHead"
    enable_fixtures: list[str] = ["stand_mixer"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stand_mixer = self.register_fixture_ref("stand_mixer", dict(id=FixtureType.STAND_MIXER))
        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the stand mixer head."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.stand_mixer.set_head_pos(env)

    def _check_success(self, env):
        """
        Check if the stand mixer head is closed.

        Returns:
            bool: True if the head is closed, False otherwise.
        """
        return self.stand_mixer.get_state(env)["head"] < 0.01
