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


class OrganizeCleaningSupplies(LwTaskBase):

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.SINK]

    """
    Organize Cleaning Supplies: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of preparing to clean the sink.

    Steps:
        Open the cabinet. Pick the cleaner and place it next to the sink.
        Then close the cabinet.

    Args:
        cab_id (str): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the cleaner is
            picked.
    """

    task_name: str = "OrganizeCleaningSupplies"
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id, ref=self.sink))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()

        cleaner_name = self.get_obj_lang("cleaner")

        ep_meta["lang"] = (
            "Open the cabinet. "
            f"Pick the {cleaner_name} and place it next to the sink. "
            "Then close the cabinet."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.close_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="cleaner",
                obj_groups="cleaner",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.10),
                    pos=(0, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_1",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        cfgs.append(
            dict(
                name="distr_counter_2",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _obj_sink_dist(self, env, obj_name):
        """
        Returns the distance of the object from the sink
        """
        sink_points = torch.tensor(np.stack(self.sink.get_ext_sites(all_points=True, relative=False)), device=env.device).unsqueeze(0)  # (1, 8, 3)
        obj_point = env.scene.rigid_objects[obj_name].data.body_com_pos_w[..., 0:1, :]  # (env_num, 1, 3)

        all_dists = torch.norm(sink_points - obj_point, dim=-1)  # (env_num, 8)
        return torch.min(all_dists, dim=-1).values

    def _check_success(self, env):

        # must make sure the cleaner is on the counter and close to the sink
        gripper_obj_far = OU.gripper_obj_far(env, obj_name="cleaner")
        obj_on_counter = OU.check_obj_fixture_contact(env, "cleaner", self.counter)

        obj_name = self.objects["cleaner"].task_name
        obj_sink_close = self._obj_sink_dist(env, obj_name) < 0.35

        door_state = self.cab.get_door_state(env=env)

        door_closed = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for env_id in range(env.num_envs):
            for joint_p in door_state.values():
                if joint_p[env_id] > 0.05:
                    door_closed[env_id] = False
                    break

        return gripper_obj_far & obj_on_counter & door_closed & obj_sink_close
