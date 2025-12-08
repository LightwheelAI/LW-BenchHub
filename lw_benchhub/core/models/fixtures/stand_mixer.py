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

from isaaclab.envs import ManagerBasedRLEnv

import lw_benchhub.utils.math_utils.transform_utils.torch_impl as T

from .fixture import Fixture
from .fixture_types import FixtureType


class StandMixer(Fixture):
    fixture_types = [FixtureType.STAND_MIXER]

    def __init__(self, name: str, prim: str, num_envs: int, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self.mirror_placement = False
        self._button_head_lock = torch.tensor([False], device=self.device).repeat(self.num_envs)
        self._speed_dial_knob_value = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
        self._head_value = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
        self._lifting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._joint_names = {
            "button_head_lock": "button_head_lock_joint",
            "bowl": "bowl_joint",
            "knob_speed": "knob_speed_joint",
            "head": "head_joint",
        }

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        try:
            self._button_head_lock = torch.tensor([False], device=self.device).repeat(self.num_envs)
            self._speed_dial_knob_value = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
            self._head_value = torch.tensor([0.0], device=self.device).repeat(self.num_envs)
            self._lifting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        except Exception as e:
            print("stand mixer setup_env failed")
            return

    def set_speed_dial_knob(self, env: ManagerBasedRLEnv, knob_val):
        """
        Sets the speed of the stand mixer

        Args:
            knob_val (float): normalized value between 0 and 1 (max speed)
        """
        self._speed_dial_knob_value = torch.clip(
            torch.tensor(knob_val, device=env.device), 0.0, 1.0)
        jn = self._joint_names["knob_speed"]
        self.set_joint_state(
            env=env,
            min=self._speed_dial_knob_value,
            max=self._speed_dial_knob_value,
            joint_names=[jn],
        )

    def set_head_pos(self, env: ManagerBasedRLEnv, head_val=1.0):
        """
        Sets the position of the head

        Args:
            head_val (float): normalized value between 0 (closed) and 1 (open)
        """
        self._head_value = torch.clip(
            torch.tensor(head_val, device=env.device), 0.0, 1.0)
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
        joint_names = env.scene.articulations[self.name].data.joint_names
        if not hasattr(self, "_lifting"):
            self._lifting = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "_button_head_lock"):
            self._button_head_lock = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # read power button
        btn = self._joint_names["button_head_lock"]
        if btn in joint_names:
            pos = self.get_joint_state(env, [btn])[btn]  # (num_envs,)
            self._button_head_lock = pos > 0.75

        # update lifting
        self._lifting = self._lifting | self._button_head_lock   # (num_envs,)

        env_id = torch.where(self._lifting)[0]  # tensor of indices

        if len(env_id) > 0:
            head_joint_name = self._joint_names["head"]
            if head_joint_name in joint_names:
                current_head_val = self.get_joint_state(env, [head_joint_name])[head_joint_name]  # (num_envs,)
                target_val = torch.ones_like(current_head_val)

                # compute only on active envs
                new_head_val = current_head_val.clone()
                new_head_val[env_id] = torch.lerp(
                    current_head_val[env_id],
                    target_val[env_id],
                    0.1,
                )
                new_head_val = torch.clamp(new_head_val, 0.0, 1.0)
                self.set_joint_state(
                    env=env,
                    min=new_head_val[env_id],
                    max=new_head_val[env_id],
                    joint_names=[head_joint_name],
                    env_ids=env_id,
                )
                self._head_value = new_head_val
                # decide which envs finished lifting
                lift_finished = torch.isclose(new_head_val, target_val, atol=1e-3)
                # reset only finished envs
                self._lifting = self._lifting & (~lift_finished)
                self._button_head_lock = self._button_head_lock & (~lift_finished)

        # sync positions back into internal values
        mapping = {
            "_speed_dial_knob_value": "knob_speed",
            "_head_value": "head",
        }
        for attr, key in mapping.items():
            jn = self._joint_names[key]
            if jn in env.scene.articulations[self.name].data.joint_names:
                setattr(self, attr, self.get_joint_state(env, [jn])[jn])

    def check_item_in_bowl(self, env, obj_name: str, partial_check=False):
        """
        Check if an object is in the bowl of the stand mixer.
        """
        obj = env.cfg.isaaclab_arena_env.task.objects[obj_name]
        sites = self.get_int_sites(relative=False)
        all_in = np.array([True]).repeat(env.num_envs)
        for env_id in range(env.num_envs):
            if partial_check:
                pts = torch.mean(env.scene.rigid_objects[obj.task_name].data.body_com_pos_w, dim=1)[env_id].cpu().numpy()  # (3, )
                pts = pts.reshape(1, 3)
                tol = 0.015
            else:
                pos = env.scene.rigid_objects[obj.task_name].data.body_com_pos_w[env_id, 0, :]  # (3, )
                pos = (pos + env.scene.env_origins[env_id]).cpu().numpy()
                quat = env.scene.rigid_objects[obj.task_name].data.body_com_quat_w[env_id, 0, :]  # (4, )
                quat = T.convert_quat(quat, to="xyzw").cpu().numpy()
                pts = obj.get_bbox_points(trans=pos, rot=quat)
                pts = np.stack(pts, axis=0)
                tol = 0.02
            for (_, int_sites) in sites.items():
                for site in int_sites:
                    site += env.scene.env_origins[env_id].cpu().numpy()
                p0, px, py, pz = int_sites
                u, v, w = px[env_id] - p0[env_id], py[env_id] - p0[env_id], pz[env_id] - p0[env_id]
                mid = p0 + 0.5 * (pz - p0)
                for pt_id in range(pts.shape[0]):
                    cu, cv, cw = np.dot(u, pts[pt_id]), np.dot(v, pts[pt_id]), np.dot(w, pts[pt_id])
                    # Check if point is within bounds for each dimension
                    u_in_bounds = (np.dot(u, p0[env_id]) - tol <= cu) & (cu <= np.dot(u, px[env_id]) + tol)
                    v_in_bounds = (np.dot(v, p0[env_id]) - tol <= cv) & (cv <= np.dot(v, py[env_id]) + tol)
                    w_in_bounds = (np.dot(w, p0[env_id]) - tol <= cw) & (cw <= np.dot(w, mid[env_id]) + tol)

                    if not (u_in_bounds and v_in_bounds and w_in_bounds):
                        all_in[env_id] = False
                        break
        return torch.from_numpy(all_in).to(env.device)

    def get_state(self, env: ManagerBasedRLEnv):
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
