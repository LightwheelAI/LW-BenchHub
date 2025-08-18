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

from isaaclab.envs import ManagerBasedRLEnvCfg

from .fixture import Fixture


class StandMixer(Fixture):

    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)

        self.mirror_placement = False
        self._button_head_lock = torch.tensor([False], device=cfg.device).repeat(cfg.num_envs)
        self._speed_dial_knob_value = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)
        self._head_value = torch.tensor([0.0], device=cfg.device).repeat(cfg.num_envs)

        # Joint name mapping
        self._joint_names = {
            "button_head_lock": "button_head_lock_joint",
            "bowl": "bowl_joint",
            "knob_speed": "knob_speed_joint",
            "head": "head_joint",
        }

    def set_speed_dial_knob(self, env, knob_val):
        """
        Sets the speed of the stand mixer

        Args:
            knob_val (float): normalized value between 0 and 1 (max speed)
        """
        self._speed_dial_knob_value = torch.clip(knob_val, 0.0, 1.0)
        jn = self._joint_names["knob_speed"]
        self.set_joint_state(
            env=env,
            min=self._speed_dial_knob_value,
            max=self._speed_dial_knob_value,
            joint_names=[jn],
        )

    def set_head_pos(self, env, head_val=1.0):
        """
        Sets the position of the head

        Args:
            head_val (float): normalized value between 0 (closed) and 1 (open)
        """
        self._head_value = torch.clip(head_val, 0.0, 1.0)
        jn = self._joint_names["head"]

        self.set_joint_state(
            env=env,
            min=self._head_value,
            max=self._head_value,
            joint_names=[jn],
        )

    def update_state(self, env):
        """
        Update the state of the stand mixer
        """
        # read power button
        btn = self._joint_names["button_head_lock"]
        if btn in env.scene.articulations[self.name].data.joint_names:
            pos = self.get_joint_state(env, [btn])[btn]
            self._button_head_lock = pos > 0.75

        # sync positions back into internal values
        mapping = {
            "_speed_dial_knob_value": "knob_speed",
            "_head_value": "head",
        }
        for attr, key in mapping.items():
            jn = self._joint_names[key]
            if jn in env.scene.articulations[self.name].data.joint_names:
                setattr(self, attr, self.get_joint_state(env, [jn])[jn])

    def check_item_in_bowl(self, env, obj_name, partial_check=False):
        """
        Check if an object is in the bowl of the stand mixer.
        """
        obj = env.objects[obj_name]
        sites = self.get_int_sites(relative=False)
        if partial_check:
            pts = torch.norm(env.scene.articulations[obj.name].data.body_com_pos_w, dim=1)  # (env_num, 3)
            tol = 0.0
        else:
            pos = torch.norm(env.scene.articulations[obj.name].data.body_com_pos_w, dim=1)  # (env_num, 3)
            quat = torch.norm(env.scene.articulations[obj.name].data.body_com_quat_w, dim=1)  # (env_num, 4)
            pts = obj.get_bbox_points(trans=pos, rot=quat)
            tol = 1e-2

        for (_, (p0, px, py, pz)) in sites.items():
            u, v, w = px - p0, py - p0, pz - p0
            mid = p0 + 0.5 * (pz - p0)
            all_in = torch.tensor([True], device=env.device).repeat(env.num_envs)
            for env_id in range(pts.shape[0]):
                cu, cv, cw = torch.dot(u, pts[env_id]), torch.dot(v, pts[env_id]), torch.dot(w, pts[env_id])
                if not (
                    (torch.dot(u, p0) - tol <= cu <= torch.dot(u, px) + tol)
                    and (torch.dot(v, p0) - tol <= cv <= torch.dot(v, py) + tol)
                    and (torch.dot(w, p0) - tol <= cw <= torch.dot(w, mid) + tol)
                ):
                    all_in[env_id] &= False
        return all_in

    def get_state(self, env):
        """
        Returns a dictionary representing the state of the stand mixer.
        """
        st = {}
        for name, jn in self._joint_names.items():
            if jn in env.scene.articulations[self.name].data.joint_names:
                st[name] = getattr(self, f"_{name}_value", None)
        return st

    @property
    def nat_lang(self):
        return "stand mixer"

    def get_reset_region_names(self):
        return ("bowl",)
