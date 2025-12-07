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

import re
import torch
import numpy as np
from functools import cached_property

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv

from .fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd
from .fixture_types import FixtureType
import lwlab.utils.math_utils.transform_utils.torch_impl as T
import lwlab.utils.object_utils as OU

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
    fixture_types = [FixtureType.STOVE]
    STOVE_LOW_MIN = 0.35
    STOVE_HIGH_MIN = np.deg2rad(80)

    def __init__(self, name="stove", prim=None, num_envs=1, *args, **kwargs):
        super().__init__(name, prim, num_envs, *args, **kwargs)
        self.valid_knob_joint_names = [j for j in self._joint_infos.keys() if "knob_" in j]
        self.valid_locations = [loc for loc in STOVE_LOCATIONS if any(f"knob_{loc}_joint" == j for j in self.valid_knob_joint_names)]
        self._knob_joint_ranges = {}  # store the range of each knob joint
        self._reset_regions = self._init_reset_regions(prim)

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
                flame_scale = (qpos - 0.25) / ((torch.pi / 2) - 0.25)
                flame_scale = torch.where(qpos >= 0.15, flame_scale, torch.tensor(0.0, device=qpos.device))
                flame_scale = torch.clamp(flame_scale, 0.0, 1.0)

                if 0.25 <= qpos <= 2 * torch.pi - 0.25:
                    if burner_site[env_idx] is not None:
                        burner_site[env_idx].GetAttribute("visibility").Set("inherited")
                        burner_site[env_idx].GetAttribute("purpose").Set("default")
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

    def set_knob_state(self, env, knob, mode="on", env_ids=None):
        """
        Sets the state of the knob joint based on the mode parameter

        Args:
            env (ManagerBasedRLEnv): environment

            knob (str): location of the knob

            mode (str): "on", "off", "high", or "low"
        """
        assert mode in ["on", "off", "high", "low"]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs)
        for env_id in env_ids:
            knob_joint_id = env.scene.articulations[self.name].data.joint_names.index(f"knob_{knob}_joint")
            joint_pos_limits = env.scene.articulations[self.name].data.joint_pos_limits[:, knob_joint_id]
            joint_min, joint_max = joint_pos_limits[0, 0].item(), joint_pos_limits[0, 1].item()
            if knob not in self._knob_joint_ranges:
                self._knob_joint_ranges[knob] = joint_pos_limits

            if mode == "off":
                joint_val = 0.0
            elif mode == "low":
                joint_val = self.rng.uniform(self.STOVE_LOW_MIN, self.STOVE_HIGH_MIN - 1e-5)
            elif mode == "high":
                joint_val = self.rng.uniform(self.STOVE_HIGH_MIN, joint_max)
            else:
                if self.rng.uniform() < 0.5:
                    joint_val = self.rng.uniform(0.5, np.pi / 4)
                else:
                    joint_val = self.rng.uniform(np.pi / 4, np.pi / 2)

            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[joint_val]]).to(env.device),
                torch.tensor([knob_joint_id]).to(env.device),
                torch.tensor([env_id]).to(env.device)
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

    def is_burner_on(self, env, burner_loc, th=0.35):
        """
        checks if a specified burner is on or off
        """
        knobs_state = self.get_knobs_state(env=env)
        knob_value = knobs_state[burner_loc]
        knob_on = (torch.abs(knob_value) >= th) & (torch.abs(knob_value) <= (2 * torch.pi - 0.35))
        return knob_on

    def check_obj_location_on_stove(self, env, obj_name, threshold=0.08, need_knob_on=True):
        """
        Check if the object is on the stove and close to a burner and the knob is on (optional).
        Returns the location of the burner if the object is on the stove, close to a burner, and the burner is on (optional).
        None otherwise.
        """

        knobs_state = self.get_knobs_state(env=env)
        if obj_name in env.scene.articulations:
            obj_pos = env.scene.articulations[obj_name].data.body_link_pos_w[..., 0, :]
        else:
            obj_pos = env.scene.rigid_objects[obj_name].data.body_link_pos_w[..., 0, :]

        obj_on_stove = OU.check_obj_fixture_contact(env, obj_name, self)
        stove_pos = torch.tensor(self.pos, device=self.device)
        stove_rot = T.euler2mat(torch.tensor([0.0, 0.0, self.rot], device=self.device)).to(dtype=torch.float32)
        locations = [[] for _ in range(env.num_envs)]
        for env_id in range(len(obj_on_stove)):
            found_location = False
            if obj_on_stove[env_id]:
                for location, site in self.burner_sites.items():
                    if site[env_id] is not None:
                        burner_pos = stove_rot @ torch.tensor(site[env_id].GetAttribute("xformOp:translate").Get(), device=env.device) + stove_pos + env.scene.env_origins[env_id]
                        dist = torch.norm(burner_pos[:2] - obj_pos[env_id][:2])
                        obj_on_site = dist < threshold
                        knob_on = (
                            (0.35 <= torch.abs(knobs_state[location][env_id]) <= 2 * torch.pi - 0.35)
                            if location in knobs_state
                            else False
                        )
                        check_result = obj_on_site if not need_knob_on else obj_on_site and knob_on
                        if check_result:
                            found_location = True
                            locations[env_id].append(location)
                            break
            if not found_location:
                locations[env_id].append(None)
        return locations

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
        self._knob_joints = {}
        if self._env is not None:
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

    def get_reset_regions(self, env=None, locs=None):
        regions = dict()
        if locs is None:
            locs = self.valid_locations
        for location in locs:
            regions[location] = self._reset_regions[location]
        return regions

    def _init_reset_regions(self, prim):
        regions = dict()
        prim_pos = torch.tensor(list(prim.GetAttribute("xformOp:translate").Get()), device=self.device)
        for location in self.valid_locations:
            site = usd.get_prim_by_name(prim, f"burner_{location}_place_site", only_xform=False)
            site = site[0] if site else None
            if site is None:
                site = usd.get_prim_by_name(prim, f"burner_on_{location}", only_xform=False)
                site = site[0] if site else None

            if site is None:
                continue

            reg_pos, reg_quat, _ = usd.get_prim_pos_rot_in_world(site)
            reg_pos = torch.tensor(reg_pos, device=self.device)
            reg_quat = torch.tensor(reg_quat, device=self.device)
            reg_rel_pos = T.quat2mat(T.convert_quat(reg_quat, to="xyzw")).T @ (reg_pos - prim_pos)
            regions[location] = {
                "offset": reg_rel_pos.tolist(),
                "size": [0.10, 0.10],
            }

        return regions


class Stovetop(Stove):
    pass
