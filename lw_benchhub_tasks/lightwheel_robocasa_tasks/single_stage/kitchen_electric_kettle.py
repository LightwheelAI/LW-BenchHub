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


class CloseElectricKettleLid(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.ELECTRIC_KETTLE]
    """
    Class encapsulating the atomic close electric kettle lid task.
    """

    task_name: str = "CloseElectricKettleLid"
    enable_fixtures: list[str] = ["electric_kettle"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.electric_kettle = self.register_fixture_ref(
            "electric_kettle", dict(id=FixtureType.ELECTRIC_KETTLE)
        )
        self.init_robot_base_ref = self.electric_kettle

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the lid of the electric kettle."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.electric_kettle.init_state(env)
        self.electric_kettle.set_lid(env, lid_val=1.0)

    def _check_success(self, env):
        return self.electric_kettle.get_state(env)["lid"] <= 0.01


class OpenElectricKettleLid(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.ELECTRIC_KETTLE]
    """
    Class encapsulating the atomic open electric kettle lid task.
    """

    task_name: str = "OpenElectricKettleLid"
    enable_fixtures: list[str] = ["electric_kettle"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.electric_kettle = self.register_fixture_ref(
            "electric_kettle", dict(id=FixtureType.ELECTRIC_KETTLE)
        )
        self.init_robot_base_ref = self.electric_kettle

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Press the button to open the lid of the electric kettle."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.electric_kettle.init_state(env)

    def _check_success(self, env):
        return self.electric_kettle.get_state(env)["lid"] >= 0.95


class TurnOnElectricKettle(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.ELECTRIC_KETTLE]

    """
    Class encapsulating the atomic turn on electric kettle task.
    """

    task_name: str = "TurnOnElectricKettle"
    enable_fixtures: list[str] = ["electric_kettle"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.electric_kettle = self.register_fixture_ref(
            "electric_kettle", dict(id=FixtureType.ELECTRIC_KETTLE)
        )
        self.init_robot_base_ref = self.electric_kettle

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Press down the lever to turn on the electric kettle."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.electric_kettle.init_state(env)

    def _check_success(self, env):
        return self.electric_kettle.get_state(env)["turned_on"]
