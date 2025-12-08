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


class AdjustToasterOvenTemperature(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.TOASTER_OVEN]
    """
    Class encapsulating atomic task for adjusting the toaster oven temperature.
    """

    task_name: str = "AdjustToasterOvenTemperature"
    enable_fixtures: list[str] = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        if "initial_temp" in scene._ep_meta:
            self.initial_temp = scene._ep_meta["initial_temp"]
        else:
            self.initial_temp = float(self.rng.random())

        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "Increase" if self.should_increase else "Decrease"
        ep_meta["lang"] = f"{direction.capitalize()} the toaster oven temperature."
        ep_meta["initial_temp"] = self.initial_temp
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.toaster_oven.set_temperature(env=env, val=self.initial_temp, env_ids=env_ids)
        self.should_increase = self.initial_temp < 0.5

    def _check_success(self, env):
        toaster_oven_state = self.toaster_oven.get_state(env)
        current_temp = toaster_oven_state["temperature"]
        temp_diff = current_temp - self.initial_temp

        if self.should_increase:
            return temp_diff >= 0.15
        else:
            return temp_diff <= -0.15


class TurnOnToasterOven(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.TOASTER_OVEN]
    """
    Class encapsulating atomic task for turning on the toaster oven by setting the timer.
    """

    task_name: str = "TurnOnToasterOven"
    enable_fixtures: list[str] = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Turn on the toaster oven by setting the timer."
        return ep_meta

    def _check_success(self, env):
        return self.toaster_oven.get_state(env)["time"] >= 0.1


class SlideToasterOvenRack(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.TOASTER_OVEN]
    """
    Class encapsulating the atomic task for sliding rack in or out of the toaster oven.
    """

    task_name: str = "SlideToasterOvenRack"
    enable_fixtures: list[str] = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.init_robot_base_ref = self.toaster_oven
        if "rack_level" in scene._ep_meta:
            self.should_pull = scene._ep_meta["should_pull"]
            self.rack_level = scene._ep_meta["rack_level"]
        else:
            self.should_pull = self.rng.random() > 0.5
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        direction = "out" if self.should_pull else "in"
        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Fully slide the toaster oven {rack_pos} {self.chosen_toaster_receptacle} {direction}."
        else:
            ep_meta[
                "lang"
            ] = f"Fully slide the toaster oven {self.chosen_toaster_receptacle} {direction}."
        ep_meta["should_pull"] = self.should_pull
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.toaster_oven.open_door(env)

        if not self.should_pull:
            self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
                env, rack_level=self.rack_level, env_ids=env_ids
            )
        else:
            self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
                env, value=0.50, rack_level=self.rack_level, env_ids=env_ids
            )

    def _check_success(self, env):
        toaster_oven_state = self.toaster_oven.get_state(env, rack_level=self.rack_level)

        movable_keys = [
            k
            for k in toaster_oven_state
            if k.startswith("rack") or k.startswith("tray")
        ]

        key = movable_keys[0]
        current_pos = toaster_oven_state[key]

        if current_pos is None:
            return torch.tensor([False], device=env.device).repeat(env.num_envs)

        if self.should_pull:
            return current_pos >= 0.99
        else:
            return current_pos <= 0.01
