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
from lw_benchhub.core.tasks.base import LwTaskBase


class HeatMultipleWater(LwTaskBase):
    """
    Heat Multiple Water: composite task for Boiling activity.

    Simulates the process of heating water in a pot and a kettle.

    Steps:
        Take the kettle from the cabinet and place it on a stove burner.
        Take the pot from the counter and place it on another stove burner.
        Turn both burners on.

    Args:
        init_robot_base_pos (str): Specifies a fixture to initialize the robot
            in front of. Default is "stove".
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOVE]
    task_name: str = "HeatMultipleWater"
    init_robot_base_pos: str = "stove"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.ref_cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET, ref=self.stove)
        )
        self.ref_counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.ref_cab, size=(0.2, 0.2))
        )

        self.init_robot_base_ref = self.ref_cab

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("pot"),
                graspable=True,
                placement=dict(
                    fixture=self.ref_counter,
                    sample_region_kwargs=dict(
                        ref=self.ref_cab,
                    ),
                    size=(0.40, 0.50),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj2",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                placement=dict(
                    fixture=self.ref_cab,
                    size=(0.50, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick the kettle from the cabinet and place it on a stove burner. "
            "Then pick the pot from the counter and place it on another stove burner. "
            "Finally, turn both burners on."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.ref_cab.open_door(env=env, env_ids=env_ids)
        valid_knobs = self.stove.get_knobs_state(env=env).keys()

        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=env, env_ids=env_ids)

    def _check_success(self, env):
        pan_locs = self.stove.check_obj_location_on_stove(env, "obj", threshold=0.15, need_knob_on=True)
        kettle_locs = self.stove.check_obj_location_on_stove(env, "obj2", need_knob_on=True)

        # both objects placed on different parts of the stove
        successful_stove_placement = torch.tensor([
            (pan_loc and len(pan_loc) > 0 and pan_loc[0] is not None)
            and (kettle_loc and len(kettle_loc) > 0 and kettle_loc[0] is not None)
            and (pan_loc[0] != kettle_loc[0])
            for pan_loc, kettle_loc in zip(pan_locs, kettle_locs)
        ], device=env.device)

        return successful_stove_placement & OU.gripper_obj_far(env)
