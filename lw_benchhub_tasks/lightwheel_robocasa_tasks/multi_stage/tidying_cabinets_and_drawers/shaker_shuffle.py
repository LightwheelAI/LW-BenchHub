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
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.kitchen_drawer import ManipulateDrawer


class ShakerShuffle(ManipulateDrawer):

    layout_registry_names: list[int] = [FixtureType.CABINET]

    """
    Shaker Shuffle: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of reorganizing the pantry by only placing the shakers in
    the drawer.

    Steps:
        Open the cabinet. Place the shakers in the drawer. Close the cabinet.
    """

    task_name: str = "ShakerShuffle"
    behavior: str = "close"

    def _place_robot(self, scene):
        # do nothing, skip parent placement logic
        return True

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.get_fixture(FixtureType.CABINET, ref=self.drawer)
        # Prevent the drawer and cabinet from being too close,
        # which may block the robot from approaching.
        same_rot = abs(np.dot(self.cab.rot, self.drawer.rot)) > 0.95
        threshold = 0.8 if same_rot else 1.3
        dist = np.linalg.norm(self.cab.pos - self.drawer.pos)
        if dist < threshold:
            self.cab = self.register_fixture_ref("cab", dict(id=FixtureType.CABINET, ref=self.cab))

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super(ManipulateDrawer, self).get_ep_meta()
        ep_meta[
            "lang"
        ] = "Pick and place the shaker into the drawer. Then close the cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)
        self.drawer.open_door(env=env, min=0.67, max=0.67, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="shaker1",
                obj_groups="shaker",
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.1),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="shaker2",
                obj_groups="shaker",
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.1),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment_bottle",
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, 1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        # make sure only the shakers were placed in the drawer!
        shaker_in_drawer = OU.obj_inside_of(env, "shaker1", self.drawer) \
            & OU.obj_inside_of(env, "shaker2", self.drawer) \
            & ~ OU.obj_inside_of(env, "condiment", self.drawer)

        door_state = self.cab.get_door_state(env=env)

        joint_positions = torch.stack(list(door_state.values()), dim=0)  # (num_joints, num_envs)
        door_closed = (joint_positions < 0.05).all(dim=0)  # (num_envs,)

        return shaker_in_drawer & door_closed
