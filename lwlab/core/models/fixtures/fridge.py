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


class Fridge(Fixture):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
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

    def get_reset_regions(self, env, reg_type="fridge", z_range=(0.50, 1.50)):
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

            if z_range is not None:
                reg_abs_z = self.pos[2] + p0[2]
                if reg_abs_z < z_range[0] or reg_abs_z > z_range[1]:
                    # region hard to reach, skip
                    continue

            reset_regions[reg_name] = {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        return reset_regions


class FridgeFrenchDoor(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)


class FridgeSideBySide(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)


class FridgeBottomFreezer(Fridge):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
