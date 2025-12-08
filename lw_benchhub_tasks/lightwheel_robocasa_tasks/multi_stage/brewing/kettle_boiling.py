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
import torch

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class KettleBoiling(LwTaskBase):
    """
    Kettle Boiling: composite task for Brewing activity.

    Simulates the task of boiling water in a kettle.

    Steps:
        Take the kettle from the counter and place it on a stove burner.
        Turn the burner on.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.STOVE]
    task_name: str = "KettleBoiling"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=(0.2, 0.2))
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("kettle_non_electric"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.35, 0.35),
                    pos=("ref", -1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="stove_distr",
                obj_groups=("pan", "pot"),
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    # apply rotations so the handle doesnt stick too much
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick the kettle from the counter and place it on a stove burner. Then turn the burner on."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        valid_knobs = self.stove.get_knobs_state(env=env).keys()
        for knob in valid_knobs:
            self.stove.set_knob_state(mode="off", knob=knob, env=env, env_ids=env_ids)

    def _check_success(self, env):
        """
        Check if the kettle is placed on the stove burner and the burner is turned on.
        """

        kettle_loc = self.stove.check_obj_location_on_stove(env, "obj", threshold=0.15)
        kettle_on_stove = torch.tensor([inner is not None for loc in kettle_loc for inner in loc], device=env.device)
        gripper_obj_far = OU.gripper_obj_far(env)
        return kettle_on_stove & gripper_obj_far
