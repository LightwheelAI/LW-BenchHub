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


class PrepMarinatingMeat(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER]

    """
    Prep Marinating Meat: composite task for Meat Preparation activity.

    Simulates the task of preparing meat for marinating.

    Steps:
        Take the meat from its container and place it on the cutting board. Then,
        take the condiment from the cabinet and place it next to the cutting board.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the condiment
            is picked.
    """

    task_name: str = "PrepMarinatingMeat"
    cab_id: FixtureType = FixtureType.CABINET_DOUBLE_DOOR
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab, size=(0.6, 0.6))
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        cond_name = self.get_obj_lang("condiment")
        meat_name = self.get_obj_lang("meat")
        cont_name = self.get_obj_lang("meat_container")
        ep_meta["lang"] = (
            f"Pick the {meat_name} from the {cont_name} and place it on the cutting board. "
            f"Then pick the {cond_name} from the cabinet and place it next to the cutting board."
        )

        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)
        self.cab.set_door_state(min=0.90, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.5),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                    try_to_place_in_kwargs=dict(
                        object_scale=0.8,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment_bottle",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.1),
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

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env, "condiment") & OU.gripper_obj_far(env, "meat")
        condiment_on_counter = OU.check_contact(
            env, self.objects["condiment"], self.counter
        )
        meat_on_cutting_board = OU.check_obj_in_receptacle(
            env, "meat", "cutting_board"
        )
        cutting_board_on_counter = OU.check_obj_fixture_contact(
            env, "cutting_board", self.counter
        )
        return gripper_obj_far & meat_on_cutting_board & cutting_board_on_counter & condiment_on_counter
