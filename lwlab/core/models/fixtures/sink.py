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
from robocasa.models.fixtures.sink import Sink as RoboCasaSink

from .fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd


class Sink(Fixture, RoboCasaSink):
    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)

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

        for env_id, site in enumerate(self.water_sites):
            if water_on[env_id]:
                site.GetAttribute("visibility").Set("inherited")
            else:
                site.GetAttribute("visibility").Set("invisible")

    def set_handle_state(self, env, rng, mode="on"):
        """
        Sets the state of the handle_joint based on the mode parameter

        Args:
            env (ManagerBasedRLEnv): environment

            rng (np.random.Generator): random number generator

            mode (str): "on", "off", or "random"
        """
        assert mode in ["on", "off", "random"]
        for env_id, _ in enumerate(self.prim_paths):
            if mode == "random":
                mode = rng.choice(["on", "off"])

            if mode == "off":
                joint_val = 0.0
            elif mode == "on":
                joint_val = rng.uniform(0.40, 0.50)

            joint_id = env.scene.articulations[self.name].data.joint_names.index("handle_joint")
            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[joint_val]]).to(env.device),
                torch.tensor([joint_id]).to(env.device),
                env_ids=torch.tensor([env_id], device=env.device)
            )

    def get_handle_state(self, env):
        """
        Gets the state of the handle_joint

        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps handle_joint to the angle of the handle_joint, water_on to whether the water is flowing,
            spout_joint to the angle of the spout_joint, and spout_ori to the orientation of the spout (left, right, center)
        """
        handle_state = {}
        if self.handle_joint is None:
            return handle_state

        joint_names = env.scene.articulations[self.name].data.joint_names
        joint_names = [j.lower() for j in joint_names]
        handle_joint_idx = joint_names.index("handle_joint")
        handle_joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, handle_joint_idx]
        handle_joint_qpos = handle_joint_qpos % (2 * torch.pi)
        handle_joint_qpos[handle_joint_qpos < 0] += 2 * torch.pi
        handle_state["handle_joint"] = handle_joint_qpos
        handle_state["water_on"] = (0.40 < handle_joint_qpos) & (handle_joint_qpos < torch.pi)

        spout_joint_idx = joint_names.index("spout_joint")
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
        assert len(handle_joints) == len(self.prim_paths), f"Handle joint not found!"
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
                    sites.append(site_prim)
        assert len(sites) == len(self.prim_paths), f"Water site not found!"
        return sites

    @property
    def nat_lang(self):
        return "sink"
