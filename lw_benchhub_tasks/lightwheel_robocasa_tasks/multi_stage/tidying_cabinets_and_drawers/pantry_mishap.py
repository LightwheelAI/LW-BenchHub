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
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.kitchen_drawer import ManipulateDrawer


class PantryMishap(ManipulateDrawer):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER]

    """
    Pantry Mishap: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of organizing the pantry after a mishap from the incorrect
    placement of items in the cabinet.

    Steps:
        Open the cabinet. Pick the vegetable and place it on the counter. Pick the
        canned food and place it in the drawer. Close the cabinet.
    """

    task_name: str = "PantryMishap"
    behavior: str = "close"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.drawer)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        vegetable = OU.get_obj_lang(self, "vegetable")
        ep_meta["lang"] = (
            f"Place the {vegetable} on the counter and the canned food in the drawer. "
            "Close the cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="vegetable",
                obj_groups="vegetable",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.1),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="canned_food",
                obj_groups="canned_food",
                placement=dict(
                    fixture=self.cab,
                    size=(0.5, 0.1),
                    pos=(0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        vegetable_on_counter = OU.check_obj_fixture_contact(
            env, "vegetable", self.counter
        )
        canned_food_in_drawer = OU.obj_inside_of(env, "canned_food", self.drawer)

        door_state = self.cab.get_door_state(env=env)

        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        door_closed = (joint_positions <= 0.01).all(dim=0)  # (num_envs,)

        return vegetable_on_counter & canned_food_in_drawer & door_closed
