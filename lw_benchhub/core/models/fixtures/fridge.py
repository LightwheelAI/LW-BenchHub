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

from isaaclab.envs import ManagerBasedRLEnvCfg

import lw_benchhub.utils.object_utils as OU

from .fixture import Fixture
from .fixture_types import FixtureType


class Fridge(Fixture):
    fixture_types = [FixtureType.FRIDGE]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._fridge_door_joint_names = []
        self._freezer_door_joint_names = []
        self._drawer_joint_names = []
        for joint_name in self._joint_infos:
            if "door" in joint_name and "fridge" in joint_name:
                self._fridge_door_joint_names.append(joint_name)
            elif "door" in joint_name and "freezer" in joint_name:
                self._freezer_door_joint_names.append(joint_name)
            elif "drawer" in joint_name:
                self._drawer_joint_names.append(joint_name)
        self._fridge_reg_names = [
            reg_name for reg_name in self._regions.keys() if "fridge" in reg_name
        ]
        self._freezer_reg_names = [
            reg_name for reg_name in self._regions.keys() if "freezer" in reg_name
        ]

    def update_state(self, env):
        """
        Updates the interior bounding boxes of the drawer to be matched with
        how open the drawer is. This is needed when determining if an object
        is inside the drawer or when placing an object inside an open drawer.

        Args:
            env (ManagerBasedRLEnv): environment
        """
        drawer_regions = [
            reg_name
            for reg_name in self._regions.keys()
            if "drawer" in reg_name
        ]
        for reg_name in drawer_regions:
            matching_joints = [j for j in self._drawer_joint_names if reg_name in j]
            if not matching_joints:
                continue

            joint_name = matching_joints[0]
            try:
                joint_id = env.scene.articulations[self.name].joint_names.index(joint_name)
                qpos = env.scene.articulations[self.name].data.joint_pos[:, joint_id].cpu().numpy()
                offset = np.stack([np.zeros_like(qpos), -qpos, np.zeros_like(qpos)], axis=-1)
                if "per_env_offset" not in self._regions[reg_name]:
                    self._regions[reg_name]["per_env_offset"] = np.zeros((self.num_envs, 3))
                self._regions[reg_name]["per_env_offset"] = offset
            except (ValueError, KeyError) as e:
                pass

    def is_open(self, env, entity="fridge", th=0.9, reg_type="door", drawer_rack_index=None):
        if reg_type == "door":
            joint_names = None
            if entity == "fridge":
                joint_names = self._fridge_door_joint_names
            elif entity == "freezer":
                joint_names = self._freezer_door_joint_names
            return super().is_open(env, joint_names, th)
        elif reg_type == "drawer":
            draw_joints = self._get_drawer_joints(
                compartment=entity, drawer_rack_index=drawer_rack_index
            )
            if not draw_joints:
                return torch.tensor([True], dtype=torch.bool, device=env.device).repeat(env.num_envs)
            return super().is_open(env, joint_names=draw_joints, th=th)
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def is_closed(self, env, entity="fridge", th=0.005, reg_type="door", drawer_rack_index=None):
        if reg_type == "door":
            joint_names = None
            if entity == "fridge":
                joint_names = self._fridge_door_joint_names
            elif entity == "freezer":
                joint_names = self._freezer_door_joint_names
            return super().is_closed(env, joint_names, th)
        elif reg_type == "drawer":
            draw_joints = self._get_drawer_joints(
                compartment=entity, drawer_rack_index=drawer_rack_index
            )
            if not draw_joints:
                return torch.tensor([True], dtype=torch.bool, device=env.device).repeat(env.num_envs)
            return super().is_closed(env, joint_names=draw_joints, th=th)
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def check_rack_contact(self, env, object_name, rack_index=None, compartment="fridge", reg_type="shelf", partial_check=False, th=0.05):
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
            for name in self.get_reset_regions(env, reg_type=reg_type, compartment=compartment, rack_index=rack_index)]
        inside = torch.tensor([False], dtype=torch.bool, device=env.device).repeat(env.num_envs)
        for i in range(env.num_envs):
            obj_pos = env.scene.rigid_objects[object_name].data.body_com_pos_w[i, 0, :]
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
            inside[i] = OU.obj_inside_of(env, object_name, self.name, partial_check=partial_check, th=th)[i]
            self.get_int_sites = orig_get_int
        return inside

    def open_door(self, env, env_ids=None, min=0.9, max=1, entity="fridge", reg_type="door", drawer_rack_index=None):
        if reg_type == "door":
            joint_names = None
            if entity == "fridge":
                joint_names = self._fridge_door_joint_names
            elif entity == "freezer":
                joint_names = self._freezer_door_joint_names
            self.set_joint_state(min=min, max=max, env=env, env_ids=env_ids, joint_names=joint_names)
        elif reg_type == "drawer":
            drawer_joints = self._get_drawer_joints(
                compartment=entity, drawer_rack_index=drawer_rack_index
            )
            if drawer_joints:
                self.set_joint_state(min=min, max=max, env=env, env_ids=env_ids, joint_names=drawer_joints)
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")
        self.update_state(env)

    def close_door(self, env, env_ids=None, min=0, max=0, entity="fridge", reg_type="door", drawer_rack_index=None):
        if reg_type == "door":
            joint_names = None
            if entity == "fridge":
                joint_names = self._fridge_door_joint_names
            elif entity == "freezer":
                joint_names = self._freezer_door_joint_names
            self.set_joint_state(min=min, max=max, env=env, env_ids=env_ids, joint_names=joint_names)
        elif reg_type == "drawer":
            drawer_joints = self._get_drawer_joints(
                compartment=entity, drawer_rack_index=drawer_rack_index
            )
            if drawer_joints:
                self.set_joint_state(
                    min=min, max=max, env=env, env_ids=env_ids, joint_names=drawer_joints
                )
        else:
            raise ValueError(f"Invalid reg_type: {reg_type}")

    def get_reset_region_names(self):
        return self._fridge_reg_names + self._freezer_reg_names

    def get_reset_regions(self, env, compartment="fridge", reg_type=("shelf"), z_range=(0.50, 1.50), rack_index=None):
        assert compartment in ["fridge", "freezer"]
        if isinstance(reg_type, str):
            reg_type = tuple([reg_type])
            for t in reg_type:
                assert t in ["shelf", "drawer"]
        region_names = []
        for reg_name in self.get_reset_region_names():
            if compartment not in reg_name:
                continue

            if "drawer" in reg_name:
                # it is a drawer
                if "drawer" not in reg_type:
                    continue
            else:
                # it is a shelf
                if "shelf" not in reg_type:
                    continue

            region_names.append(reg_name)

        reset_regions = {}
        for reg_name in region_names:
            reg_dict = self._regions.get(reg_name, None)
            if reg_dict is None:
                continue
            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]
            pz = reg_dict["pz"]
            height = pz[2] - p0[2]
            if height < 0.20 and "drawer" not in reg_type:
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

    def _get_drawer_joints(self, compartment="fridge", drawer_rack_index=None):
        """
        Returns the list of drawer joint names for a given compartment. If drawer_rack_index is provided,
        selects a single joint based on reverse-alphabetical sorting (highest first).

        Args:
            compartment (str): "fridge" or "freezer"
            drawer_rack_index (int or None):
                - None: return all matching drawer joints
                - -1: highest drawer
                - -2: second highest drawer (falls back to highest if only one)
                - N >= 0: Nth drawer in reverse-sorted order

        Returns:
            list[str]: selected joint names (can be empty if none found)
        """
        compartment_drawer_joints = []
        for joint_name in self._drawer_joint_names:
            if compartment in joint_name:
                compartment_drawer_joints.append(joint_name)

        if not compartment_drawer_joints:
            return []

        if drawer_rack_index is None:
            return compartment_drawer_joints

        sorted_joints = sorted(compartment_drawer_joints, reverse=True)

        if drawer_rack_index == -1:
            return [sorted_joints[0]] if sorted_joints else []
        elif drawer_rack_index == -2:
            if len(sorted_joints) > 1:
                return [sorted_joints[1]]
            return [sorted_joints[0]] if sorted_joints else []
        elif 0 <= drawer_rack_index < len(sorted_joints):
            return [sorted_joints[drawer_rack_index]]
        else:
            raise ValueError(
                f"drawer_rack_index {drawer_rack_index} out of range for {compartment} drawer joints. "
                f"Available indices: {list(range(len(sorted_joints)))}"
            )


class FridgeFrenchDoor(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        super().setup_cfg(cfg)


class FridgeSideBySide(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        super().setup_cfg(cfg)


class FridgeBottomFreezer(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        super().setup_cfg(cfg)
