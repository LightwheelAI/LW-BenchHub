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

from isaaclab.envs import ManagerBasedRLEnvCfg
from .fixture import Fixture
import numpy as np
import torch
import lwlab.utils.object_utils as OU
from .fixture_types import FixtureType


class Fridge(Fixture):
    fixture_types = [FixtureType.FRIDGE]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._fridge_door_joint_names = []
        self._freezer_door_joint_names = []
        for joint_name in self._joint_infos:
            if "door" in joint_name and "fridge" in joint_name:
                self._fridge_door_joint_names.append(joint_name)
            elif "door" in joint_name and "freezer" in joint_name:
                self._freezer_door_joint_names.append(joint_name)

        self._fridge_reg_names = [
            reg_name for reg_name in self._regions.keys() if "fridge" in reg_name
        ]
        self._freezer_reg_names = [
            reg_name for reg_name in self._regions.keys() if "freezer" in reg_name
        ]

    def is_open(self, env, entity="fridge", th=0.9):
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names
        return super().is_open(env, joint_names, th)

    def is_closed(self, env, entity="fridge", th=0.005):
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names
        return super().is_closed(env, joint_names, th)

    def check_rack_contact(self, env, object_name, rack_index=None, compartment="firdge", reg_type=("shelf"),):
        """
        Check if an object is in contact with a specific shelf or drawer in the fridge.

        Args:
            env (Kitchen): the environment the fridge belongs to
            object_name (str): name of the object to check
            rack_index (int or None): if set, selects a specific shelf/drawer by index
                (0 = lowest, -1 = highest, -2 = second highest)
            compartment (str): "fridge" or "freezer" — which compartment to query
            reg_type (tuple or str): can be combination of shelf or drawer. specifies
                whether to use shelves or drawers, or both

        Returns:
            bool: True if the object is in contact with the specified shelf/drawer, False otherwise
        """
        region_names = [
            name
            for name in self.get_reset_regions(env)]
        inside = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        for i in range(env.num_envs):
            obj_pos = torch.mean(env.scene.rigid_objects[object_name].data.body_com_pos_w, dim=1)[i]
            obj_z = obj_pos[2]
            filtered_region_names = []
            for region_name in region_names:
                reg = self._regions[region_name]
                p0, pz = reg["p0"], reg["pz"]

                region_min_z = self.pos[2] + p0[2]
                region_max_z = self.pos[2] + pz[2]

                is_drawer = "drawer" in region_name
                if is_drawer:
                    restricted_max_z = region_max_z
                else:
                    restricted_max_z = region_min_z + 0.9 * (region_max_z - region_min_z)
                is_filtered_region = (restricted_max_z >= obj_z) and (region_min_z <= obj_z)
                if is_filtered_region:
                    filtered_region_names.append(region_name)

            if not filtered_region_names:
                return inside
            orig_get_int = self.get_int_sites

            def get_int_sites_filtered(relative=False):
                sites = orig_get_int(relative=relative)
                return {rn: sites[rn] for rn in filtered_region_names}

            self.get_int_sites = get_int_sites_filtered
            inside[i] = OU.obj_inside_of(env, object_name, self.name, partial_check=False)[i]
            self.get_int_sites = orig_get_int
        return inside

    def open_door(self, env, env_ids=None, min=0.9, max=1, entity="fridge"):
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names
        self.set_joint_state(min=min, max=max, env=env, env_ids=env_ids, joint_names=joint_names)

    def close_door(self, env, env_ids=None, min=0, max=0, entity="fridge"):
        joint_names = None
        if entity == "fridge":
            joint_names = self._fridge_door_joint_names
        elif entity == "freezer":
            joint_names = self._freezer_door_joint_names
        self.set_joint_state(min=min, max=max, env=env, env_ids=env_ids, joint_names=joint_names)

    def get_reset_region_names(self):
        return self._fridge_reg_names + self._freezer_reg_names

    def get_reset_regions(self, env=None, reg_type="fridge", z_range=(0.50, 1.50), rack_index=None):
        assert reg_type in ["fridge", "freezer"]
        reset_region_names = [
            reg_name
            for reg_name in self.get_reset_region_names()
            if reg_type in reg_name
        ]
        reset_regions = {}
        for reg_name in reset_region_names:
            reg_dict = self._regions.get(reg_name, None)
            if reg_dict is None:
                continue
            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]
            pz = reg_dict["pz"]
            height = pz[2] - p0[2]
            if height < 0.20:
                # region is too small, skip
                continue

            # bypass z-range check if rack_index is specified
            bypass_z_range = rack_index is not None
            if not bypass_z_range:
                reg_abs_z = self.pos[2] + p0[2]
                if reg_abs_z < z_range[0] or reg_abs_z > z_range[1]:
                    continue

            reset_regions[reg_name] = {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        # sort by Z height (top shelf/drawer first)
        sorted_regions = sorted(
            reset_regions.items(),
            key=lambda item: self.pos[2] + self._regions[item[0]]["p0"][2],
            reverse=False,
        )

        if rack_index is not None:
            if rack_index == -1:
                return dict([sorted_regions[-1]]) if sorted_regions else {}
            elif rack_index == -2:
                if len(sorted_regions) > 1:
                    return dict([sorted_regions[-2]])
                else:
                    return dict([sorted_regions[-1]]) if sorted_regions else {}
            elif 0 <= rack_index < len(sorted_regions):
                return dict([sorted_regions[rack_index]])
            else:
                raise ValueError(
                    f"rack_index {rack_index} out of range for {reg_type} regions. "
                    f"Available indices: {list(range(len(sorted_regions)))}"
                )
        return dict(sorted_regions)


class FridgeFrenchDoor(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        super().setup_cfg(cfg)


class FridgeSideBySide(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        super().setup_cfg(cfg)


class FridgeBottomFreezer(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        super().setup_cfg(cfg)
