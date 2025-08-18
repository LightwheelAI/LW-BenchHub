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
from .fixture import Fixture
from lwlab.utils.usd_utils import OpenUsd as usd
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv


class CoffeeMachine(Fixture):
    """
    Coffee machine object. Supports turning on coffee machine, and simulated coffee pouring
    """

    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):
        super().setup_cfg(cfg, root_prim)
        self._turned_on = torch.tensor([False], dtype=torch.bool, device=cfg.device).repeat(cfg.num_envs)
        self.force_threshold = 0.1
        self.pos_threshold = 0.05
        self._activation_time = torch.zeros(cfg.num_envs, device=cfg.device)
        self._active = torch.tensor([False], dtype=torch.bool, device=cfg.device).repeat(cfg.num_envs)
        self._display_duration = 10.0

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def get_state(self):
        """
        returns whether the coffee machine is turned on or off as a dictionary with the turned_on key
        """
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    def update_state(self, env):
        """
        Checks if the gripper is pressing the start button. If this is the first time the gripper pressed the button,
        the coffee machine is turned on, and the coffee liquid sites are turned on.

        Args:
            env (ManagerBasedRLEnv): The environment to check the state of the coffee machine in
        """
        start_button_pressed = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for _, button in self.start_button_infos.items():
            for gripper_name in [name for name in list(self._env.scene.sensors.keys()) if "gripper" in name and "contact" in name]:
                start_button_pressed |= env.cfg.check_contact(gripper_name.replace("_contact", ""), str(button[0].GetPrimPath()))

        # detect button press (only when False to True)
        self._turned_on = ~self._turned_on & start_button_pressed

        # if turned_on is True, accumulate time
        if torch.any(self._turned_on):
            self._activation_time[self._turned_on] += 1 / 50

        # judge if the coffee machine is active
        self._active = self._turned_on & (self._activation_time < self._display_duration)

        # update coffee liquid sites visibility
        for site_name in self.coffee_liquid_site_names:
            sites_for_name = self.coffee_liquid_sites[site_name]
            for i, site in enumerate(sites_for_name):
                if site is not None and site.IsValid():
                    if self._active[i]:
                        site.GetAttribute("visibility").Set("inherited")
                    else:
                        site.GetAttribute("visibility").Set("invisible")

    def check_receptacle_placement_for_pouring(self, env, obj_name, xy_thresh=0.04):
        """
        check whether receptacle is placed under coffee machine for pouring

        Args:
            env (ManagerBasedRLEnv): The environment to check the state of the coffee machine in
            obj_name (str): name of the object
            xy_thresh (float): threshold for xy distance between object and receptacle

        Returns:
            bool: True if object is placed under coffee machine, False otherwise
        """
        obj_pos = env.scene.rigid_objects[obj_name].data.body_com_pos_w[:, 0, :]
        check = []
        for i, pour_site_pos in enumerate(self.pour_sites):
            if pour_site_pos is None:
                check.append(False)
                continue

            pour_site_pos = torch.FloatTensor(pour_site_pos)
            xy_check = torch.norm(obj_pos[i, 0:2] - pour_site_pos[0:2]) < xy_thresh
            z_check = torch.abs(obj_pos[i, 2] - pour_site_pos[2]) < 0.10
            check.append(bool(xy_check) and bool(z_check))
        return torch.tensor(check, dtype=torch.bool, device=env.device)

    @cached_property
    def pour_sites(self):
        sites_list = []
        for i, site in enumerate(self.receptacle_place_sites):
            if site is None:
                sites_list.append(None)
                continue
            sites_list.append(usd.get_prim_pos_rot_in_world(site.GetPrim())[0])
        return sites_list

    def gripper_button_far(self, env, th=0.15):
        """
        check whether gripper is far from the start button

        Args:
            env (ManagerBasedRLEnv): The environment to check the state of the coffee machine in
            th (float): threshold for distance between gripper and button

        Returns:
            bool: True if gripper is far from the button, False otherwise
        """
        result = torch.tensor([True] * env.num_envs, dtype=torch.bool, device=env.device)
        gripper_site_pos = env.scene["ee_frame"].data.target_pos_w  # (env_num, ee_num, 3)
        for name, button in self.start_button_infos.items():
            button_idx = env.scene.articulations[self.name].data.body_names(name)
            button_pos = env.scene.articulations[self.name].data.body_com_pos_w[:, button_idx:button_idx + 1, :]  # (env_num, 1, 3)
            gripper_button_far = torch.norm(gripper_site_pos - button_pos, dim=-1) > th  # (env_num, ee_num)
            result &= torch.all(gripper_button_far, dim=-1)  # (env_num,)

        return result

    @cached_property
    def coffee_liquid_sites(self):
        sites_dict = {}
        for site_name in self.coffee_liquid_site_names:
            sites_list = []
            for prim_path in self.prim_paths:
                sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), site_name, only_xform=False)
                for site in sites_prim:
                    if site is not None and site.IsValid():
                        sites_list.append(site)
            sites_dict[site_name] = sites_list
        for site_name in self.coffee_liquid_site_names:
            assert site_name in sites_dict.keys(), f"Coffee liquid site {site_name} not found!"
        return sites_dict

    @cached_property
    def receptacle_place_sites(self):
        sites_list = {}
        for prim_path in self.prim_paths:
            sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), "receptacle_place_site", only_xform=False)
            for site in sites_prim:
                if site is not None and site.IsValid():
                    if "receptacle_place_site" not in sites_list:
                        sites_list["receptacle_place_site"] = [site]
                    else:
                        sites_list["receptacle_place_site"].append(site)
        assert len(list(sites_list.keys())) > 0, f"Receptacle place site not found!"
        return sites_list

    @cached_property
    def coffee_liquid_site_names(self):
        names_infos = {}
        for liquid_name in ["coffee_liquid_left", "coffee_liquid_right", "coffee_liquid"]:
            for prim_path in self.prim_paths:
                sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), liquid_name, only_xform=False)
                for site in sites_prim:
                    if site is not None and site.IsValid():
                        if liquid_name not in names_infos:
                            names_infos[liquid_name] = [site]
                        else:
                            names_infos[liquid_name].append(site)
        assert len(list(names_infos.keys())) > 0, f"Coffee liquid site not found!"
        return names_infos

    @cached_property
    def start_button_infos(self):
        button_infos = {}
        for prim_path in self.prim_paths:
            buttons_prim = usd.get_prim_by_prefix(self._env.sim.stage.GetObjectAtPath(prim_path), f"{self.folder.split('/')[-1]}_Button")
            for button in buttons_prim:
                if button is not None and button.IsValid():
                    if button.GetName() not in button_infos:
                        button_infos[button.GetName()] = [button]
                    else:
                        button_infos[button.GetName()].append(button)
        assert all(name.startswith(f"{self.folder.split('/')[-1]}_Button") for name in list(button_infos.keys())), f"Microwave Button not found!"
        return button_infos

    @property
    def nat_lang(self):
        return "coffee machine"
