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


class CondimentCollection(LwTaskBase):
    """
    Condiment Collection: composite task for Clearing Table activity.

    Simulates the task of collecting condiments from the counter and placing
    them in the cabinet.

    Steps:
        Pick the condiments from the counter and place them in the cabinet.

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the
            condiments are picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER]
    task_name: str = "CondimentCollection"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        obj_name_1 = self.get_obj_lang("condiment1")
        obj_name_2 = self.get_obj_lang("condiment2")

        ep_meta[
            "lang"
        ] = f"Pick the {obj_name_1} and {obj_name_2} from the counter and place them in the cabinet."
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
                name="condiment1",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(0.60, -1.0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=(-0.60, -1.0),
                    offset=(-0.05, 0.0),
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
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
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
        gripper_obj_far_condiment1 = OU.gripper_obj_far(env, obj_name="condiment1")
        gripper_obj_far_condiment2 = OU.gripper_obj_far(env, obj_name="condiment2")
        condiment1_inside_cab = OU.obj_inside_of(env, "condiment1", self.cab)
        condiment2_inside_cab = OU.obj_inside_of(env, "condiment2", self.cab)
        return (
            condiment1_inside_cab
            & condiment2_inside_cab
            & gripper_obj_far_condiment1
            & gripper_obj_far_condiment2
        )
