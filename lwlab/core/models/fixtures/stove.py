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
import numpy as np
from functools import cached_property

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from robocasa.models.fixtures.stove import Stove as RoboCasaStove
from robocasa.models.fixtures.stove import Stovetop as RoboCasaStovetop

from .fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd


STOVE_LOCATIONS = [
    "rear_left",
    "rear_center",
    "rear_right",
    "front_left",
    "front_center",
    "front_right",
    "center",
    "left",
    "right",
]


class Stove(Fixture, RoboCasaStove):
    _env = None

    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
        self.valid_knob_joint_names = [j for j in self._joint_infos.keys() if "knob_" in j]
        self.valid_locations = [l for l in STOVE_LOCATIONS if any(f"knob_{l}_joint" == j for j in self.valid_knob_joint_names)]

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def update_state(self, env):
        for location in self.valid_locations:
            burner_site = self.burner_sites[location]
            place_site = self.place_sites[location]
            joint = self.knob_joints[location]
            joint_qpos = joint % (2 * torch.pi)
            joint_qpos[joint_qpos < 0] += 2 * torch.pi
            for env_idx, qpos in enumerate(joint_qpos):
                if 0.35 <= torch.abs(qpos) <= 2 * torch.pi - 0.35:
                    burner_site[env_idx].GetParent().GetAttribute("visibility").Set("inherited")
                    place_site[env_idx].GetParent().GetAttribute("visibility").Set("inherited")
                else:
                    burner_site[env_idx].GetParent().GetAttribute("visibility").Set("invisible")
                    place_site[env_idx].GetParent().GetAttribute("visibility").Set("invisible")

    def set_knob_state(self, env, rng, knob, mode="on"):
        """
        Sets the state of the knob joint based on the mode parameter

        Args:
            env (ManagerBasedRLEnv): environment

            rng (np.random.RandomState): random number generator

            knob (str): location of the knob

            mode (str): "on" or "off"
        """
        assert mode in ["on", "off"]
        for env_idx in range(env.num_envs):
            if mode == "off":
                joint_val = 0.0
            else:
                if rng.uniform() < 0.5:
                    joint_val = rng.uniform(0.50, np.pi / 2)
                else:
                    joint_val = rng.uniform(2 * np.pi - np.pi / 2, 2 * np.pi - 0.50)
            knob_joint_id = env.scene.articulations[self.name].data.joint_names.index(f"knob_{knob}_joint")
            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[joint_val]]).to(env.device),
                torch.tensor([knob_joint_id]).to(env.device),
                torch.tensor([env_idx]).to(env.device)
            )

    def get_knobs_state(self, env):
        """
        Gets the angle of which knob joints are turned

        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps location of knob to the angle of the knob joint
        """
        knobs_state = {}
        for location in self.valid_locations:
            joint = self.knob_joints[location]
            joint_qpos = joint % (2 * torch.pi)
            joint_qpos[joint_qpos < 0] += 2 * torch.pi
            knobs_state[location] = joint_qpos
        return knobs_state

    @property
    def knob_joints(self):
        """
        Returns the knob joints of the stove
        """
        if not hasattr(self, "_knob_joints") or self._knob_joints is None:
            self._knob_joints = {k: None for k, v in super().knob_joints.items() if v is not None}
        if self._env is not None:
            self.valid_locations = list(self._knob_joints.keys())
            for location in self.valid_locations:
                joint_id = self._env.scene.articulations[self.name].data.joint_names.index(f"knob_{location}_joint")
                joint = self._env.scene.articulations[self.name].data.joint_pos[:, joint_id]
                self._knob_joints[location] = joint
            return self._knob_joints
        else:
            return self._knob_joints

    @cached_property
    def place_sites(self):
        """
        Returns the place site of the stove
        """
        self._place_sites = {}
        for location in self.valid_locations:
            self._place_sites[location] = []
            for prim_path in self.prim_paths:
                sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), "Sites")
                place_site = usd.get_prim_by_name(sites_prim[0], f"burner_{location}_place_site", only_xform=False)
                place_site = place_site[0] if place_site else None
                if place_site is not None and place_site.IsValid():
                    self._place_sites[location].append(place_site)
        for location in self.valid_locations:
            assert len(self._place_sites[location]) == len(self.prim_paths), f"Place site {location} not found!"
        return self._place_sites

    @cached_property
    def burner_sites(self):
        """
        Returns the burner sites of the stove
        """
        self._burner_sites = {}
        for location in self.valid_locations:
            self._burner_sites[location] = []
            for prim_path in self.prim_paths:
                sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), "Sites")
                burner_site = usd.get_prim_by_name(sites_prim[0], f"burner_on_{location}", only_xform=False)
                burner_site = burner_site[0] if burner_site else None
                if burner_site is not None and burner_site.IsValid():
                    self._burner_sites[location].append(burner_site)
        for location in self.valid_locations:
            assert location in self._burner_sites.keys(), f"Burner site {location} not found!"
        return self._burner_sites

    @property
    def nat_lang(self):
        return "stove"


class Stovetop(Stove, RoboCasaStovetop):
    pass
