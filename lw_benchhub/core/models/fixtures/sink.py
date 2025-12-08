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

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class Sink(Fixture):
    fixture_types = [FixtureType.SINK]

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def update_state(self, env):
        """
        Updates the water flowing of the sink based on the handle_joint position

        Args:
            env (ManagerBasedRLEnv): environment
        """
        state = self.get_handle_state(env)
        water_on = state["water_on"]

        for env_id, (site, origin_radius) in enumerate(self.water_sites):
            if site is None or not site.IsValid():
                continue

            if water_on[env_id]:
                site.GetAttribute("visibility").Set("inherited")
                site.GetAttribute("purpose").Set("default")
                # set radius scale
                radius = float(state["water_scale"][env_id]) * origin_radius
                site.GetAttribute("radius").Set(radius)
            else:
                site.GetAttribute("visibility").Set("invisible")

    def set_handle_state(self, env, mode="on", temp=None):
        """
        Sets the state of the handle joints based on the mode parameter.
        Flow is controlled by the primary handle axis; temperature (optional)
        is controlled by the secondary handle axis if present.

        Args:
            env (ManagerBasedRLEnv): environment

            mode (str): "on", "off", or "random"
            temp (str | None): optional temperature label: "hot" | "warm" | "cold"
        """
        assert mode in ["on", "off", "random"]
        for env_id, _ in enumerate(self.prim_paths):
            if mode == "random":
                mode = self.rng.choice(["on", "off"])

            if mode == "off":
                joint_val = 0.0
            elif mode == "on":
                joint_val = self.rng.uniform(0.40, 0.50)

            joint_names = env.scene.articulations[self.name].data.joint_names
            joint_names_l = [j.lower() for j in joint_names]
            handle_indices = [i for i, n in enumerate(joint_names_l) if "handle_joint" in n]
            if not handle_indices:
                continue
            # set primary axis (flow)
            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[joint_val]]).to(env.device),
                torch.tensor([handle_indices[0]]).to(env.device),
                env_ids=torch.tensor([env_id], device=env.device)
            )
            # optionally set temperature axis if present
            if temp is not None and len(handle_indices) >= 2:
                # Directional temperature around midpoint: one side hot, opposite side cold
                temp_joint_idx = handle_indices[1]
                temp_limits = env.scene.articulations[self.name].data.joint_pos_limits[0, temp_joint_idx, :]
                jmin = float(temp_limits[0])
                jmax = float(temp_limits[1])
                mid = (jmin + jmax) * 0.5
                span = (jmax - jmin)
                delta = 0.15 * span
                if temp == "hot":
                    temp_val = mid - delta
                elif temp == "cold":
                    temp_val = mid + delta
                else:  # warm or unspecified -> near midpoint
                    temp_val = mid
                env.scene.articulations[self.name].write_joint_position_to_sim(
                    torch.tensor([[temp_val]]).to(env.device),
                    torch.tensor([temp_joint_idx]).to(env.device),
                    env_ids=torch.tensor([env_id], device=env.device)
                )

    def check_obj_under_water(self, env, obj_name, xy_thresh=None):
        if xy_thresh is None:
            xy_thresh = env.cfg.isaaclab_arena_env.task.objects[obj_name].horizontal_radius
        obj_pos = OU.get_object_pos(env, obj_name)  # shape: (3,)
        result = torch.tensor([False] * env.num_envs, device=env.device)
        for env_id, (site, origin_radius) in enumerate(self.water_sites):
            if site is None or not site.IsValid():
                continue
            water_site_pos = torch.tensor([usd.get_prim_pos_rot_in_world(site)[0]], device=env.device)  # (env_num, 3)
            # obj_pos is (3,), water_site_pos is (env_num, 3)
            obj_pos_xy = obj_pos[env_id][0:2]  # (2,)
            water_site_pos_xy = water_site_pos[:, 0:2]  # (2,)
            # Broadcast obj_pos_xy to match water_site_pos_xy
            xy_check = torch.norm(water_site_pos_xy - obj_pos_xy) < xy_thresh
            # Get cylinder height/length
            cylinder_height = 0.0
            if site.GetAttribute("height").Get():
                cylinder_height = site.GetAttribute("height").Get()

            if cylinder_height > 0:
                z_check = (
                    obj_pos[env_id][2]
                    < water_site_pos[:, 2] + cylinder_height
                )
                result |= xy_check & z_check
            else:
                result |= xy_check
        return result & self.get_handle_state(env)["water_on"]

    def get_handle_state(self, env):
        """
        Gets the state of the handle_joint

        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: keys include
              - handle_joint (Tensor): primary handle angle (flow axis)
              - water_on (Tensor[bool]): water flowing or not
              - water_scale (Tensor): relative flow scale
              - temp_joint (Tensor, optional): secondary handle angle (temperature axis) if present
              - temp_level (Tensor, optional): normalized temperature level in [-1 (cold), +1 (hot)] if present
              - temp (list[str], optional): categorical temperature labels per env if present
              - spout_joint (Tensor, optional)
              - spout_ori (list[str], optional)
        """
        handle_state = {}
        if self.handle_joint is None:
            return handle_state

        joint_names = env.scene.articulations[self.name].data.joint_names
        joint_names = [j.lower() for j in joint_names]
        handle_indices = [i for i, n in enumerate(joint_names) if "handle_joint" in n]
        if not handle_indices:
            return handle_state
        # primary (flow) axis
        handle_joint_idx = handle_indices[0]
        handle_joint_range = env.scene.articulations[self.name].data.joint_pos_limits[:, handle_joint_idx]
        handle_joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, handle_joint_idx]

        handle_joint_qpos = handle_joint_qpos % (2 * torch.pi)
        handle_joint_qpos[handle_joint_qpos < 0] += 2 * torch.pi
        handle_state["handle_joint"] = handle_joint_qpos
        handle_state["water_on"] = (0.40 < handle_joint_qpos) & (handle_joint_qpos < torch.pi)
        handle_state["water_on"] = torch.logical_and(0.1 < handle_joint_qpos, handle_joint_qpos < torch.pi)
        handle_state["water_scale"] = (handle_joint_qpos - handle_joint_range[:, 0]) / (handle_joint_range[:, 1] - handle_joint_range[:, 0])

        # secondary (temperature) axis if present
        if len(handle_indices) >= 2:
            temp_joint_idx = handle_indices[1]
            temp_joint_range = env.scene.articulations[self.name].data.joint_pos_limits[:, temp_joint_idx]
            temp_joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, temp_joint_idx]
            temp_joint_qpos = temp_joint_qpos % (2 * torch.pi)
            temp_joint_qpos[temp_joint_qpos < 0] += 2 * torch.pi
            handle_state["temp_joint"] = temp_joint_qpos
            # normalize to [0,1]
            temp_norm = (temp_joint_qpos - temp_joint_range[:, 0]) / (temp_joint_range[:, 1] - temp_joint_range[:, 0])
            # directional level: -1 => hot (below mid), 0 => warm (near mid), +1 => cold (above mid)
            eps = 0.05
            temp_dir = temp_norm - 0.5
            temp_level = torch.where(temp_dir > eps, torch.tensor(1.0, device=env.device),
                                     torch.where(temp_dir < -eps, torch.tensor(-1.0, device=env.device), torch.tensor(0.0, device=env.device)))
            handle_state["temp_level"] = temp_level
            temp_label = []
            for v in temp_dir:
                if v <= -eps:
                    temp_label.append("hot")
                elif v >= eps:
                    temp_label.append("cold")
                else:
                    temp_label.append("warm")
            handle_state["temp"] = temp_label

        spout_joint_idx = next(i for i, name in enumerate(joint_names) if "spout_joint" in name.lower())
        spout_joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, spout_joint_idx]
        spout_joint_qpos = spout_joint_qpos % (2 * torch.pi)
        spout_joint_qpos[spout_joint_qpos < 0] += 2 * torch.pi
        handle_state["spout_joint"] = spout_joint_qpos
        spout_ori = []
        for qpos in spout_joint_qpos:
            if torch.pi <= qpos <= 2 * torch.pi - torch.pi / 6:
                spout_ori.append("left")
            elif torch.pi / 6 <= qpos <= torch.pi:
                spout_ori.append("right")
            else:
                spout_ori.append("center")
        handle_state["spout_ori"] = spout_ori

        return handle_state

    @cached_property
    def handle_joint(self):
        handle_joints = []
        for prim_path in self.prim_paths:
            handle_prims = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), "handle_joint", only_xform=False)
            for handle_prim in handle_prims:
                if handle_prim is not None and handle_prim.IsValid():
                    handle_joints.append(handle_prim)
        # allow multiple axes per sink (flow + temperature); ensure at least one exists overall
        if len(handle_joints) < 1:
            return None
        return handle_joints

    @cached_property
    def spout_joint(self):
        spout_joints = []
        for prim_path in self.prim_paths:
            spout_prims = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), "spout_joint", only_xform=False)
            for spout_prim in spout_prims:
                if spout_prim is not None and spout_prim.IsValid():
                    spout_joints.append(spout_prim)
        assert len(spout_joints) == len(self.prim_paths), f"Spout joint not found!"
        return spout_joints

    @cached_property
    def water_sites(self):
        if self._env is None:
            return [None] * len(self.prim_paths)

        sites = []
        for prim_path in self.prim_paths:
            sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), "water", only_xform=False)
            for site_prim in sites_prim:
                if site_prim is not None and site_prim.IsValid():
                    origin_radius = site_prim.GetAttribute("radius").Get()
                    sites.append((site_prim, origin_radius))
        assert len(sites) == len(self.prim_paths), f"Water site not found!"
        return sites

    def get_obj_basin_loc(self, env, obj_name, partial_check=False):
        """
        Determine whether an object is in the left basin, right basin, or neither
        of a double-basin sink, taking into account the sink's rotation.
        Args:
            env (MujocoEnv): The environment instance.
            obj_name (str): Name of the object.
            partial_check (bool): If True, checks only the center of the object
                rather than full bounding box.
        Returns:
            str: "left" if in left basin, "right" if in right basin, or "none" otherwise.
        """
        if partial_check:
            threshold = 0.0
        else:
            threshold = 0.05
        locs = [[]for _ in range(env.num_envs)]
        orig_get_int = self.get_int_sites

        def get_int_sites_left(relative=False):
            sites = orig_get_int(relative=relative)
            return {rn: sites[rn] for rn in sites if "basin_left" in rn}

        def get_int_sites_right(relative=False):
            sites = orig_get_int(relative=relative)
            return {rn: sites[rn] for rn in sites if "basin_right" in rn}
        # Check if the object is inside the left basin
        self.get_int_sites = get_int_sites_left
        loc_left = OU.obj_inside_of(env, obj_name, self.name, partial_check=partial_check, th=threshold)
        for env_id in range(env.num_envs):
            if loc_left[env_id]:
                locs[env_id].append("left")
        # Check if the object is inside the right basin
        self.get_int_sites = get_int_sites_right
        loc_right = OU.obj_inside_of(env, obj_name, self.name, partial_check=partial_check, th=threshold)
        for env_id in range(env.num_envs):
            if loc_right[env_id]:
                locs[env_id].append("right")
        self.get_int_sites = orig_get_int

        return locs

    @property
    def nat_lang(self):
        return "sink"

    def get_reset_region_names(self):
        return ("basin", "basin_right", "basin_left")

    def get_reset_regions(self, env=None, side=None):
        """
        Return sink reset regions, selecting basin regions for double-basin sinks.

        Args:
            env: Unused, kept for API compatibility with other fixtures
            side (str | None): If "left" or "right", select that basin when available.
                If None, and a double basin exists, returns both basins. Otherwise falls
                back to the single "basin" region when present.

        Returns:
            dict: reset regions as computed by the base Fixture using the chosen region names.
        """
        double_basin = "basin_left" in self._regions
        single_basin = "basin" in self._regions

        if double_basin:
            chosen = side.lower() if isinstance(side, str) else None
            if chosen == "left":
                region_names = ("basin_left",)
            elif chosen == "right":
                region_names = ("basin_right",)
            else:
                region_names = ("basin_left", "basin_right")
        elif single_basin:
            if side is not None:
                raise ValueError(
                    "Side selection is not supported: this sink has a single basin"
                )
            region_names = ("basin",)
        else:
            region_names = self.get_reset_region_names()

        return super().get_reset_regions(env=env, reset_region_names=region_names)
