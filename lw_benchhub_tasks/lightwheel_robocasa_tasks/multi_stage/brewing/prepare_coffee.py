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


class PrepareCoffee(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COFFEE_MACHINE]

    """
    Prepare Coffee: composite task for Brewing activity.

    Simulates the process of preparing coffee.

    Steps:
        Pick the mug from the cabinet, place it under the coffee machine dispenser,
        and press the start button.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the mug is
            picked.
    """

    task_name: str = "PrepareCoffee"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.coffee_machine = self.register_fixture_ref(
            "coffee_machine", dict(id=FixtureType.COFFEE_MACHINE)
        )
        self.cab = self.register_fixture_ref(
            "cab", dict(id=self.cab_id, ref=self.coffee_machine)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_name = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_name} from the cabinet, place it under the coffee machine dispenser, and press the start button."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                placement=dict(
                    fixture=self.cab,
                    size=(0.30, 0.10),
                    pos=(0, -1.0),
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

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env)
        contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(
            env, "obj"
        )
        gripper_button_far = self.coffee_machine.gripper_button_far(env)
        return (
            contact_check
            & gripper_obj_far
            & self.coffee_machine._turned_on
            & gripper_button_far
        )
