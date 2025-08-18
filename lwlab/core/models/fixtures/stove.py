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


class Stove(Fixture):
    _env = None

    def __init__(self, name="stove", prim=None, *args, **kwargs):
        super().__init__(name, prim, *args, **kwargs)
        self._reset_regions = self._get_reset_regions(prim)

    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
        self.valid_knob_joint_names = [j for j in self._joint_infos.keys() if "knob_" in j]
        self.valid_locations = [l for l in STOVE_LOCATIONS if any(f"knob_{l}_joint" == j for j in self.valid_knob_joint_names)]
        self._knob_joint_ranges = {}  # store the range of each knob joint

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def update_state(self, env):
        knobs_state = self.get_knobs_state(env)
        for location in self.valid_locations:
            burner_site = self.burner_sites[location]
            place_site = self.place_sites[location]

            if burner_site is None or any(site is None for site in burner_site):
                continue

            joint_qpos = knobs_state[location]
            joint_qpos = joint_qpos % (2 * torch.pi)
            joint_qpos[joint_qpos < 0] += 2 * torch.pi

            for env_idx, qpos in enumerate(joint_qpos):
                # calculate flame intensity ratio (0-1)
                flame_scale = (qpos - 0.35) / (torch.pi - 0.35)
                flame_scale = torch.where(qpos >= 0.35, flame_scale, torch.tensor(0.0, device=qpos.device))
                flame_scale = torch.clamp(flame_scale, 0.0, 1.0)

                if 0.35 <= qpos <= 2 * torch.pi - 0.35:
                    if burner_site[env_idx] is not None:
                        burner_site[env_idx].GetAttribute("visibility").Set("inherited")
                        if hasattr(self, 'original_flame_sizes') and location in self.original_flame_sizes:
                            # get original radius and height
                            original_radius = self.original_flame_sizes[location][env_idx]["radius"]
                            original_height = self.original_flame_sizes[location][env_idx]["height"]

                            # scale radius and height
                            current_radius = float(flame_scale.item()) * original_radius
                            current_height = float(flame_scale.item()) * original_height

                            # set new radius and height
                            burner_site[env_idx].GetAttribute("radius").Set(current_radius)
                            burner_site[env_idx].GetAttribute("height").Set(current_height)
                else:
                    if burner_site[env_idx] is not None:
                        burner_site[env_idx].GetAttribute("visibility").Set("invisible")

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
            if knob not in self._knob_joint_ranges:
                self._knob_joint_ranges[knob] = env.scene.articulations[self.name].data.joint_limits[:, knob_joint_id]
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

    @cached_property
    def original_flame_sizes(self):
        """store the original flame sizes (including radius and height)"""
        sizes = {}
        for location in self.valid_locations:
            sizes[location] = []
            for burner_site in self.burner_sites[location]:
                if burner_site is not None and burner_site.IsValid():
                    sizes[location].append({
                        "radius": burner_site.GetAttribute("radius").Get(),
                        "height": burner_site.GetAttribute("height").Get()
                    })
                else:
                    sizes[location].append({"radius": 0.0, "height": 0.0})
        return sizes

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

    def get_reset_regions(self, env, locs=None):
        regions = dict()
        if locs is None:
            locs = STOVE_LOCATIONS
        for location in locs:
            regions[location] = self._reset_regions[location]
        return regions

    def _get_reset_regions(self, prim):
        regions = dict()
        locs = STOVE_LOCATIONS
        for location in locs:

            site = usd.get_prim_by_name(prim, f"burner_{location}_place_site", only_xform=False)
            site = site[0] if site else None
            if site is None:
                site = usd.get_prim_by_name(prim, f"burner_on_{location}", only_xform=False)
                site = site[0] if site else None

            if site is None:
                continue

            pos, _, _ = usd.get_prim_pos_rot_in_world(site)
            regions[location] = {
                "offset": pos,
                "size": [0.10, 0.10],
            }

        return regions


class Stovetop(Stove):
    pass
