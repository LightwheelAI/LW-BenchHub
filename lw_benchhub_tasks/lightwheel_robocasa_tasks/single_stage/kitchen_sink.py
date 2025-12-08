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


class ManipulateSinkFaucet(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    """
    Class encapsulating the atomic manipulate sink faucet tasks.

    Args:
        behavior (str): "turn_on" or "turn_off". Used to define the desired
            sink faucet manipulation behavior for the task.
    """

    behavior: str = "turn_on"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the sink faucet tasks
        """
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink faucet tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"{self.behavior.replace('_', ' ').capitalize()} the sink faucet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Reset the environment internal state for the sink faucet tasks.
        This includes setting the sink faucet state based on the behavior
        """
        super()._setup_scene(env, env_ids)

        if self.behavior == "turn_on":
            self.sink.set_handle_state(mode="off", env=env)
        elif self.behavior == "turn_off":
            self.sink.set_handle_state(mode="on", env=env)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink faucet tasks. This includes the object placement configurations.
        Place the objects on the counter and sink as distractors.

        Returns:
            list: List of object configurations
        """
        cfgs = []

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.sink),
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(1.0, 1.0),
                        pos=("ref", -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(1.0, 1.0),
                    pos=(None, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the sink faucet manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        handle_state = self.sink.get_handle_state(env=env)
        water_on = handle_state["water_on"]

        if self.behavior == "turn_on":
            success = water_on
        elif self.behavior == "turn_off":
            success = ~ water_on

        return success


class TurnOnSinkFaucet(ManipulateSinkFaucet):
    task_name: str = "TurnOnSinkFaucet"
    behavior: str = "turn_on"


class TurnOffSinkFaucet(ManipulateSinkFaucet):
    task_name: str = "TurnOffSinkFaucet"
    behavior: str = "turn_off"


class TurnSinkSpout(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    """
    Class encapsulating the atomic turn sink spout tasks.

    Args:
        behavior (str): "left" or "right". Used to define the desired sink spout
        manipulation behavior for the task.
    """

    task_name: str = "TurnSinkSpout"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the sink spout tasks
        """
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        if "task_refs" in scene._ep_meta:
            self.behavior = scene._ep_meta["task_refs"]["behavior"]
            self.init_sink_mode = scene._ep_meta["task_refs"]["init_sink_mode"]
        else:
            self.behavior = self.rng.choice(["left", "right"])
            self.init_sink_mode = self.rng.choice(["on", "off"])
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink spout tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Turn the sink spout to the {self.behavior}."
        ep_meta["task_refs"] = dict(
            behavior=self.behavior,
            init_sink_mode=self.init_sink_mode,
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Reset the environment internal state for the sink spout tasks.
        This includes setting the sink spout state based on the behavior
        """
        super()._setup_scene(env, env_ids)
        self.sink.set_handle_state(mode=self.init_sink_mode, env=env)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink spout tasks. This includes the object placement configurations.
        Place the objects on the counter and sink as distractors.
        """
        cfgs = []

        # distractors
        num_distr = self.rng.integers(1, 4)
        for i in range(num_distr):
            cfgs.append(
                dict(
                    name=f"distr_counter_{i}",
                    obj_groups="all",
                    placement=dict(
                        fixture=self.get_fixture(FixtureType.COUNTER, ref=self.sink),
                        sample_region_kwargs=dict(
                            ref=self.sink,
                            loc="left_right",
                        ),
                        size=(0.60, 0.60),
                        pos=("ref", -1.0),
                        offset=(0.0, 0.10),
                    ),
                )
            )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.30, 0.40),
                    pos=(None, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the sink spout manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        handle_state = self.sink.get_handle_state(env=env)
        success = torch.tensor([
            ori == self.behavior for ori in handle_state["spout_ori"]
        ], device=env.device)
        return success
