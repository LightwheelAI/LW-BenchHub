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

import numpy as np

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class MeatTransfer(LwTaskBase):
    """
    Meat Transfer: composite task for Chopping Food activity.

    Simulates the task of transferring meat to a container.

    Steps:
        Retrieve a container (either a pan or a bowl) from the cabinet, then place
        the raw meat into the container to avoid contamination.

    Args:
        cab_id: Enum which serves as a unique identifier for different cabinets.
            Default to FixtureType.CABINET_DOUBLE_DOOR to have space for
            initializing bowl/pan
    """

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]
    task_name: str = "MeatTransfer"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.5, 0.5))
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cont_name = self.get_obj_lang("container")
        ep_meta["lang"] = (
            f"Retrieve the {cont_name} from the cabinet, "
            f"then place the raw meat into the {cont_name} to avoid contamination."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        if self.rng.random() < 0.5:
            cfgs.append(
                dict(
                    name="container",
                    obj_groups="pan",
                    graspable=True,
                    placement=dict(
                        fixture=self.cab,
                        # ensure_object_boundary_in_range=False because the pans handle is a part of the
                        # bounding box making it hard to place it if set to True
                        ensure_object_boundary_in_range=False,
                        size=(0.05, 0.02),
                        pos=(0, 0),
                        # apply a custom rotation for the pan so that it fits better in the cabinet
                        # (if the handle sticks out it may not fit)
                        rotation=(2 * np.pi / 8, 3 * np.pi / 8),
                    ),
                )
            )
        else:
            cfgs.append(
                dict(
                    name="container",
                    obj_groups="bowl",
                    graspable=True,
                    placement=dict(
                        fixture=self.cab,
                        ensure_object_boundary_in_range=False,
                        size=(0.02, 0.02),
                        pos=(0, 0),
                    ),
                )
            )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.cab),
                    size=(0.5, 0.4),
                    pos=(0.0, -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        return (
            OU.check_obj_fixture_contact(env, "container", self.counter)
            & OU.gripper_obj_far(env, obj_name="meat")
            & OU.check_obj_in_receptacle(env, "meat", "container")
        )
