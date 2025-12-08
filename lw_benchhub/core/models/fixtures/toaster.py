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


class Toaster(Fixture):
    fixture_types = [FixtureType.TOASTER]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._controls = {
            "button": "button_cancel",
            "knob": "knob_doneness",
            "lever": "lever",
        }

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env
        try:
            self._state = {s: {c: torch.tensor([0.0], device=env.device).repeat(env.num_envs) for c in self._controls} for s in self.slot_pairs}
            self._turned_on = {s: torch.tensor([False], device=env.device).repeat(env.num_envs) for s in self.slot_pairs}
            self._num_steps_on = {s: torch.tensor([0], device=env.device).repeat(env.num_envs) for s in self.slot_pairs}
            self._cooldown = {s: torch.tensor([0], device=env.device).repeat(env.num_envs) for s in self.slot_pairs}
        except Exception as e:
            print("toaster setup_env failed")
            return

    def get_reset_region_names(self):
        return (
            "slotL",
            "slotR",
            "sideL_slotL",
            "sideL_slotR",
            "sideR_slotL",
            "sideR_slotR",
        )

    def get_reset_regions(self, env=None, slot_pair=None, side=None, **kwargs):
        """
        Returns the reset regions for the toaster slots.

        Args:
            env: environment instance
            slot_pair (int or None): 0 to N-1 slot pairs, or None for all pairs
            side (str or None): "left", "right", or None for both sides

        Returns:
            dict: reset regions for the specified slots
        """
        if side is not None and side not in ("left", "right"):
            raise ValueError(f"Invalid side {side!r}; must be None, 'left' or 'right'")

        all_regions = list(self.get_reset_region_names())

        if slot_pair is None:
            filtered_regions = all_regions
        elif slot_pair == 0:
            # Return left side regions
            filtered_regions = [
                r
                for r in all_regions
                if "sideL" in r or (r in ["slotL", "slotR"] and "side" not in r)
            ]
        elif slot_pair == 1:
            # Return right side regions
            filtered_regions = [r for r in all_regions if "sideR" in r]
        else:
            filtered_regions = []

        # Filter by side
        if side is not None:
            if side == "left":
                filtered_regions = [r for r in filtered_regions if "slotL" in r]
            elif side == "right":
                filtered_regions = [r for r in filtered_regions if "slotR" in r]

        return super().get_reset_regions(
            env=env, reset_region_names=filtered_regions, **kwargs
        )

    def set_doneness_knob(self, env: ManagerBasedRLEnv, env_id: int, slot_pair: int, value: float):
        """
        Sets the toasting doneness knob

        Args:
            slot_pair (int): the slot pair to set the knob for (0 to N-1 slot pairs)
            value (float): normalized value between 0 (min) and 1 (max)
        """
        if slot_pair not in self.slot_pairs:
            raise ValueError(f"Unknown slot_pair '{slot_pair}'")
        val = torch.clip(value, 0.0, 1.0)
        self._state[slot_pair]["knob"] = val

        jn = self.joint_names.get(f"knob_{slot_pair}")
        if jn and jn in env.scene.articulations[self.name].data.joint_names:
            self.set_joint_state(
                env=env,
                min=val,
                max=val,
                env_ids=[env_id],
                joint_names=[jn],
            )

    def set_lever(self, env: ManagerBasedRLEnv, env_id: int, slot_pair: int, value: float):
        """
        Sets the power lever

        Args:
            slot_pair (int): the slot pair to set the lever for (0 to N-1 slot pairs)
            value (float): normalized value between 0 (off) and 1 (on)
        """
        if slot_pair not in self.slot_pairs:
            raise ValueError(f"Unknown slot_pair '{slot_pair}'")
        val = torch.clip(torch.tensor(value, device=env.device), 0.0, 1.0)
        self._state[slot_pair]["lever"] = val

        jn = self.joint_names.get(f"lever_{slot_pair}")
        if jn and jn in env.scene.articulations[self.name].data.joint_names:
            self.set_joint_state(
                env=env,
                min=val,
                max=val,
                env_ids=[env_id],
                joint_names=[jn],
            )

    def update_state(self, env: ManagerBasedRLEnv):
        """
        Update the state of the toaster
        """
        for sp in self.slot_pairs:
            for control in self._controls:
                key = f"{control}_{sp}"
                jn = self.joint_names.get(key)
                if jn and jn in env.scene.articulations[self.name].data.joint_names:
                    q = self.get_joint_state(env, [jn])[jn]
                    self._state[sp][control] = torch.clip(q, 0.0, 1.0)

        for sp in self.slot_pairs:
            lev_val = self._state[sp]["lever"]

            self._cooldown[sp][lev_val <= 0.70] = 0

            self._turned_on[sp] = (lev_val >= 0.90) & (~self._turned_on[sp]) & (self._cooldown[sp] == 0)

            for env_id in range(self.num_envs):
                if self._turned_on[sp][env_id]:
                    if self._num_steps_on[sp][env_id] < 500:
                        self._num_steps_on[sp][env_id] += 1
                        self.set_lever(env, env_id, sp, 1.0)
                    else:
                        self._turned_on[sp][env_id] = False
                        self._num_steps_on[sp][env_id] = 0
                        self._cooldown[sp][env_id] = 1

                if 0 < self._cooldown[sp][env_id] < 1000:
                    self._cooldown[sp][env_id] += 1
                elif self._cooldown[sp][env_id] >= 1000:
                    self._cooldown[sp][env_id] = 0

    def check_slot_contact(self, env, obj_name: str, slot_pair: int | None = None, side: str | None = None):
        """
        Returns True if the specified object is in contact with any of the toaster slot-floor geom(s).

        Args:
            env: environment
            obj_name (str): name of the object to check
            slot_pair (int or None): 0 to N-1 slot pairs
            side (str or None): None = both sides; otherwise “left” or “right”

        Returns:
            bool: whether any of the object’s geoms are touching any selected slot-floor geom
        """
        if slot_pair is not None and (
            not isinstance(slot_pair, int)
            or not (0 <= slot_pair < len(self.slot_pairs))
        ):
            raise ValueError(
                f"Invalid slot_pair {slot_pair!r}; must be None or an int in 0-{len(self.slot_pairs)-1}"
            )

        # pick slot side
        if side is None:
            sides = ["left", "right"]
        elif side in ("left", "right"):
            sides = [side]
        else:
            raise ValueError(f"Invalid side {side!r}; must be None, 'left' or 'right'")

        slot_floor_names = []
        for slot in self.slots:
            slot_floor_names.append(f"{slot}_floor")

        slot_tokens = []
        if "left" in sides:
            slot_tokens.append("slotL")
        if "right" in sides:
            slot_tokens.append("slotR")

        # pick slot pair side
        if set(slot_tokens) == {"slotL", "slotR"}:
            side_filtered = slot_floor_names.copy()
        else:
            side_filtered = [
                n for n in slot_floor_names if any(tok in n for tok in slot_tokens)
            ]

        n_pairs = len(self.slot_pairs)
        if n_pairs == 1:
            if slot_pair in (None, 0):
                final_floor_names = side_filtered
            else:
                raise ValueError("Only slot_pair=0 is valid for a single-pair toaster")
        elif n_pairs == 2:
            if slot_pair is None:
                final_floor_names = side_filtered
            elif slot_pair == 0:
                final_floor_names = [n for n in side_filtered if "sideL" in n]
            elif slot_pair == 1:
                final_floor_names = [n for n in side_filtered if "sideR" in n]
            else:
                raise ValueError(
                    f"Invalid slot_pair {slot_pair!r}; "
                    f"must be None, 0 (left) or 1 (right)"
                )

        is_contact = torch.tensor([False], device=env.device).repeat(env.num_envs)

        # check contact
        for slot_floor_name in final_floor_names:
            floor_geom_path = self.floor_geoms[slot_floor_name][0].GetPrimPath()
            is_contact |= OU.check_contact(env, obj_name, str(floor_geom_path))

        return is_contact

    def get_state(self, env: ManagerBasedRLEnv, slot_pair: int | None = None):
        """
        Returns the current state of the toaster as a dictionary.

        Args:
            slot_pair (int or None): 0 to N-1 slot pairs

        Returns:
            dict: the current state of the toaster
        """
        full = {
            s: {**self._state[s], "turned_on": self._turned_on[s]}
            for s in self.slot_pairs
        }

        if slot_pair is None:
            return full

        if isinstance(slot_pair, int):
            try:
                slot_pair = self.slot_pairs[slot_pair]
            except (IndexError, TypeError):
                raise ValueError(
                    f"Invalid slot index {slot_pair!r}, must be between "
                    f"0 and {len(self._slots)-1}"
                )

        if slot_pair not in full:
            raise ValueError(
                f"Invalid slot_pair {slot_pair!r}, must be one of {list(full)}"
            )

        return full[slot_pair]

    @cached_property
    def slots(self):
        return [s.replace("_floor", "") for s in list(self.floor_geoms.keys())]

    @cached_property
    def slot_pairs(self):
        return list(range(int(len(self.slots) / 2)))

    @cached_property
    def floor_geoms(self):
        floor_geoms_dict = {}
        for prim_path in self.prim_paths:
            geoms = usd.get_prim_by_suffix(self._env.sim.stage.GetObjectAtPath(prim_path), "_floor", only_xform=False)
            for g in geoms:
                if g is not None and g.IsValid():
                    g_name = g.GetName()
                    if g_name not in floor_geoms_dict.keys():
                        floor_geoms_dict[g_name] = [g]
                    else:
                        floor_geoms_dict[g_name].append(g)
        for g_name in floor_geoms_dict.keys():
            assert len(floor_geoms_dict[g_name]) == len(self.prim_paths), f"floor_geoms {g_name} is not found!"
        return floor_geoms_dict

    @cached_property
    def joint_names(self):
        joint_names_dict = {}
        for prim_path in self.prim_paths:
            unfix_joint_prims = usd.get_all_joints_without_fixed(self._env.sim.stage.GetObjectAtPath(prim_path))
            unfix_joint_names = [prim.GetName() for prim in unfix_joint_prims]
        for ctrl, tag in self._controls.items():
            names = sorted([name for name in unfix_joint_names if tag in name])
            for pair, jn in zip(self.slot_pairs, names):
                joint_names_dict[f"{ctrl}_{pair}"] = jn
        return joint_names_dict
