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


class SearingMeat(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET_DOUBLE_DOOR, FixtureType.COUNTER, FixtureType.STOVE]

    """
    Searing Meat: composite task for Frying activity.

    Simulates the task of searing meat.

    Steps:
        Place the pan on the specified burner on the stove,
        then place the meat on the pan and turn the burner on.

    Args:
        knob_id (str): The id of the knob who's burner the pan will be placed on.
            If "random", a random knob is chosen.
    """

    task_name: str = "SearingMeat"
    knob_id: str = "random"
    EXCLUDE_LAYOUTS = LwTaskBase.DOUBLE_CAB_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )

        self.cab = self.register_fixture_ref(
            "cab", dict(id=FixtureType.CABINET_DOUBLE_DOOR, ref=self.stove)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        meat_name = self.get_obj_lang("meat")
        ep_meta["lang"] = (
            f"Grab the pan from the cabinet and place it on the {self.knob.replace('_', ' ')} burner on the stove. "
            f"Then place the {meat_name} on the pan and turn the burner on."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        valid_knobs = self.stove.get_knobs_state(env=env).keys()
        if self.knob_id == "random":
            self.knob = self.rng.choice(list(valid_knobs))
        else:
            assert self.knob_id in valid_knobs
            self.knob = self.knob

        self.stove.set_knob_state(mode="off", knob=self.knob, env=env, env_ids=env_ids)
        self.cab.set_door_state(min=0.90, max=1.0, env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="pan",
                obj_groups=("pan"),
                object_scale=0.7,
                placement=dict(
                    fixture=self.cab,
                    size=(0.8, 0.4),
                    pos=(0.0, -1.0),
                    offset=(0.0, -0.1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="meat",
                obj_groups="meat",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="meat")
        pan_locs = self.stove.check_obj_location_on_stove(env, "pan", threshold=0.15, need_knob_on=True)
        pan_on_loc = torch.tensor([(loc and len(loc) > 0 and loc[0] == self.knob) if loc else False for loc in pan_locs], device=env.device)
        meat_in_pan = OU.check_obj_in_receptacle(env, "meat", "pan", th=0.07)
        return gripper_obj_far & pan_on_loc & meat_in_pan
