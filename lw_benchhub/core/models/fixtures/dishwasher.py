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

from functools import cached_property

import torch

from isaaclab.envs import ManagerBasedRLEnv

from lw_benchhub.utils import object_utils as OU
from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """
    fixture_types = [FixtureType.DISHWASHER]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._door = 0.0
        self._rack = 0.0

        self._joint_names = {
            "door": f"door_joint",
            "rack": f"rack1_joint",
        }

    def get_reset_region_names(self):
        return ("rack0", "rack1")

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def slide_rack(self, env, value=1.0, env_ids=None):
        """
        Pulls the specified dishwasher rack out completely.

        Args:
            value (float): Value to set the rack to
        """

        door_pos = self.get_joint_state(env, [self._joint_names["door"]])[
            self._joint_names["door"]
        ]
        for env_id in env_ids:
            if door_pos[env_id] <= 0.99:
                self.open_door(env, env_ids=torch.tensor([env_id]))

        self.set_joint_state(
            env=env, min=value, max=value, joint_names=[self._joint_names["rack"]], env_ids=env_ids
        )

    def update_state(self, env):
        """
        Updates internal state variables from the simulation environment.
        """
        for name, jn in self._joint_names.items():
            if jn in env.scene.articulations[self.name].data.joint_names:
                state = self.get_joint_state(env, [jn])[jn]
                setattr(self, f"_{name}", state)

    def check_rack_contact(self, env, obj):
        """
        Checks whether the specified object is in contact with top rack.

        Args:
            obj_name (rigid body): object to check
        """
        return OU.check_contact(env, obj, str(self.rack_infos[self._joint_names["rack"]][0].GetPrimPath()))

    def get_state(self, env):
        """
        Returns the current state of the dishwasher as a dictionary.
        """
        st = {}
        for name, jn in self._joint_names.items():
            if jn in env.scene.articulations[self.name].data.joint_names:
                st[name] = getattr(self, f"_{name}", None)
        return st

    @cached_property
    def rack_infos(self):
        infos = {}
        for jnt_name in self._joint_infos.keys():
            if "rack" in jnt_name:
                infos[jnt_name] = []
                for prim_path in self.prim_paths:
                    prim = self._env.sim.stage.GetObjectAtPath(prim_path)
                    rack_prim = usd.get_prim_by_name(prim, f'{usd.get_child_commonprefix_name(prim)}_{jnt_name.replace("_joint", "")}')
                    for rack in rack_prim:
                        if rack is not None and rack.IsValid():
                            infos[jnt_name].append(rack)
        for jnt_name in [j for j in self._joint_infos.keys() if "rack" in j]:
            assert jnt_name in infos.keys(), f"Rack {jnt_name} not found!"
        return infos

    @property
    def nat_lang(self):
        return "dishwasher"
