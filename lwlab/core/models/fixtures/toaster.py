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
from functools import cached_property

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from robocasa.models.fixtures.toaster import Toaster as RoboCasaToaster

from .fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd


class Toaster(Fixture, RoboCasaToaster):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
        self._controls = {
            "button": "button_cancel",
            "knob": "knob_doneness",
            "lever": "lever",
        }

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env
        self._state = {s: {c: torch.tensor([0.0], device=env.device).repeat(env.num_envs) for c in self._controls} for s in self.slot_pairs}
        self._turned_on = {s: torch.tensor([False], device=env.device).repeat(env.num_envs) for s in self.slot_pairs}
        self._num_steps_on = {s: torch.tensor([0], device=env.device).repeat(env.num_envs) for s in self.slot_pairs}
        self._cooldown = {s: torch.tensor([0], device=env.device).repeat(env.num_envs) for s in self.slot_pairs}

    def set_joint_state(self, min: float, max: float, env: ManagerBasedRLEnv, env_id: int, joint_names: list[str]):
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_qpos_all = env.scene.articulations[self.name].data.joint_pos.clone()
        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]
            if joint_min >= 0:
                desired_min = joint_min + (joint_max - joint_min) * min
                desired_max = joint_min + (joint_max - joint_min) * max
            else:
                desired_min = joint_min + (joint_max - joint_min) * (1 - max)
                desired_max = joint_min + (joint_max - joint_min) * (1 - min)
            joint_qpos_all[:, joint_idx] = self.rng.uniform(desired_min.cpu(), desired_max.cpu())
        env.scene.articulations[self.name].write_joint_position_to_sim(joint_qpos_all, env_ids=[env_id])

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

        jn = self._joint_names.get(f"knob_{slot_pair}")
        if jn and jn in env.scene.articulations[self.name].data.joint_names:
            self.set_joint_state(
                env=env,
                min=val,
                max=val,
                env_id=env_id,
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
        val = torch.clip(value, 0.0, 1.0)
        self._state[slot_pair]["lever"] = val

        jn = self._joint_names.get(f"lever_{slot_pair}")
        if jn and jn in env.scene.articulations[self.name].data.joint_names:
            self.set_joint_state(
                env=env,
                min=val,
                max=val,
                env_id=env_id,
                joint_names=[jn],
            )

    # TODO: implement this
    def update_state(self, env: ManagerBasedRLEnv):
        """
        Update the state of the toaster
        """
        for sp in self.slot_pairs:
            for control in self._controls:
                key = f"{control}_{sp}"
                jn = self._joint_names.get(key)
                if jn and jn in env.scene.articulations[self.name].data.joint_names:
                    q = self.get_joint_state(env, [jn])[jn]
                    self._state[sp][control] = torch.clip(q, 0.0, 1.0)

        for sp in self.slot_pairs:
            lev_val = self._state[sp]["lever"]

            self._cooldown[sp][lev_val <= 0.70] = 0

            self._turned_on[sp] = (lev_val >= 0.90) & (~self._turned_on[sp]) & (self._cooldown[sp] == 0)

            for env_ids in range(lev_val.shape[0]):
                if self._turned_on[sp][env_ids]:
                    if self._num_steps_on[sp][env_ids] < 500:
                        self._num_steps_on[sp][env_ids] += 1
                        self.set_lever(env, env_ids, sp, 1.0)
                    else:
                        self._turned_on[sp][env_ids] = False
                        self._num_steps_on[sp][env_ids] = 0
                        self._cooldown[sp][env_ids] = 1

                if 0 < self._cooldown[sp][env_ids] < 1000:
                    self._cooldown[sp][env_ids] += 1
                elif self._cooldown[sp][env_ids] >= 1000:
                    self._cooldown[sp][env_ids] = 0

    def check_slot_contact(self, env: ManagerBasedRLEnv, obj_name: str, slot_pair: int | None = None, side: str | None = None):
        """
        Returns True if the specified object is in contact with any of the toaster slot-floor geom(s).

        Args:
            env (MujocoEnv): the environment
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
            is_contact |= env.check_contact(obj_name, str(floor_geom_path), has_sensor=False)

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

        if slot_pair.item() not in full:
            raise ValueError(
                f"Invalid slot_pair {slot_pair!r}, must be one of {list(full)}"
            )

        return full[slot_pair.item()]

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
            unfix_joint_prims = usd.get_all_joints_without_fixed(prim_path)
            unfix_joint_names = [prim.GetName() for prim in unfix_joint_prims]
        for ctrl, tag in self._controls.items():
            names = sorted([tag in name for name in unfix_joint_names])
            for pair, jn in zip(self.slot_pairs, names):
                joint_names_dict[f"{ctrl}_{pair}"] = jn
        return joint_names_dict
