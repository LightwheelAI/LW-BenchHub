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


class TurnOnToaster(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.TOASTER]
    """
    Atomic task for pushing toaster lever down to turn on.
    """

    task_name: str = "TurnOnToaster"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the toaster tasks
        """
        super()._setup_kitchen_references(scene)
        self.toaster = self.register_fixture_ref("toaster", dict(id=FixtureType.TOASTER))
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        """
        Get the episode metadata for the toaster tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Push down the lever of the toaster to turn it on."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the toaster tasks. This includes the object placement configurations.
        Place the object inside the toaster

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("sandwich_bread"),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        """
        Check if the toaster manipulation task is successful.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        result = torch.tensor([False], device=env.device).repeat(env.num_envs)
        # Check if toaster has any slots available
        if len(self.toaster.slot_pairs) == 0:
            # No slots available, return failure
            return result
        for slot_pair in range(len(self.toaster.get_state(env).keys())):
            slot_contact = self.toaster.check_slot_contact(env, "obj", slot_pair)
            turned_on = self.toaster.get_state(env, slot_pair=slot_pair)["turned_on"]
            result |= (slot_contact & turned_on)
        return result
