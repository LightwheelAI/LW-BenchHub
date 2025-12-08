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

import re
from functools import cached_property

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv

import lw_benchhub.utils.math_utils.transform_utils.torch_impl as T
from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import Fixture
from .fixture_types import FixtureType


class CoffeeMachine(Fixture):
    """
    Coffee machine object. Supports turning on coffee machine, and simulated coffee pouring
    """
    fixture_types = [FixtureType.COFFEE_MACHINE]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self._turned_on = torch.tensor([False], dtype=torch.bool, device=self.device).repeat(self.num_envs)
        self.force_threshold = 0.1
        self.pos_threshold = 0.05
        self._activation_time = torch.zeros(self.num_envs, device=self.device)
        self._active = torch.tensor([False], dtype=torch.bool, device=self.device).repeat(self.num_envs)
        self._display_duration = 10.0

        self._coffee_liquid_site_names = []
        for liquid_name in ["coffee_liquid_left", "coffee_liquid_right", "coffee_liquid"]:
            liquid_prims = usd.get_prim_by_name(self.prim, liquid_name, only_xform=False)
            if len(liquid_prims) > 0:
                self._coffee_liquid_site_names.append(liquid_name)

        self._start_button_names = []
        button_prims = usd.get_prim_by_prefix(self.prim, "start_button", only_xform=False)
        if len(button_prims) > 0:
            for button_prim in button_prims:
                self._start_button_names.append(button_prim.GetName())

        receptacle_pouring_prims = usd.get_prim_by_name(self.prim, "receptacle_place_site", only_xform=False)
        for pouring_prim in receptacle_pouring_prims:
            site_pos = tuple(pouring_prim.GetAttribute("xformOp:translate").Get() * self.scale)
            self._receptacle_pouring_site = {"pos": site_pos}

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, usually used when initialzing a mug under the coffee machine
        """
        return {
            "bottom": {
                "offset": self._receptacle_pouring_site.get("pos"),
                "size": (0.01, 0.01),
            }
        }

    def setup_env(self, env: ManagerBasedRLEnv):
        super().setup_env(env)
        self._env = env

    def reset_state(self):
        self._turned_on = torch.tensor([False], dtype=torch.bool, device=self.device).repeat(self.num_envs)
        self._activation_time = torch.zeros(self.num_envs, device=self.device)
        self._active = torch.tensor([False], dtype=torch.bool, device=self.device).repeat(self.num_envs)

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
        start_button_pressed = torch.tensor([False], device=self.device).repeat(self.num_envs)
        button_press_states = {}
        gripper_names = [
            name for name in list(env.scene.sensors.keys())
            if "gripper" in name and "contact" in name
        ]
        coffee_keys = [
            k for k in env.scene.articulations.keys()
            if "coffee_machine" in k.lower()
        ]
        if not coffee_keys:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        coffee_key = coffee_keys[0]
        articulation = env.scene.articulations[coffee_key]
        joint_names = articulation.joint_names

        for button_key, buttons in self.start_button_infos.items():
            for button_prim in buttons:
                prim_path = str(button_prim.GetPath())

                match = re.search(r'(Button[0-9]+)', prim_path, re.IGNORECASE)
                if not match:
                    continue

                button_name = match.group(1)
                button_joint_name = f"{button_name}_joint"

                button_joint_name_lower = button_joint_name.lower()
                joint_lookup = [jn for jn in joint_names if jn.lower() == button_joint_name_lower]
                if not joint_lookup:
                    continue

                button_joint_name = joint_lookup[0]
                button_joint_index = joint_names.index(button_joint_name)

                joint_pos = articulation.data.joint_pos[:, button_joint_index]
                threshold = torch.tensor([0.0001], device=self.device).repeat(self.num_envs)
                button_pressed = (joint_pos >= threshold)

                button_press_states[button_key] = button_pressed
                start_button_pressed |= button_pressed

        # detect button press (only when False -> True)
        for env_id in range(self.num_envs):
            if not self._turned_on[env_id] and start_button_pressed[env_id]:
                self._turned_on[env_id] = True

        # accumulate time if active
        if torch.any(self._turned_on):
            self._activation_time[self._turned_on] += 1 / 50

        self._active = self._turned_on & (self._activation_time < self._display_duration)

        has_single_button = "start_button" in button_press_states
        has_dual_buttons = "start_button_1" in button_press_states and "start_button_2" in button_press_states

        for site_name in self._coffee_liquid_site_names:
            sites_for_name = self.coffee_liquid_sites[site_name]

            if has_single_button and not has_dual_buttons and site_name == "coffee_liquid":
                control_state = button_press_states["start_button"]
            elif has_dual_buttons and site_name == "coffee_liquid_left":
                control_state = button_press_states["start_button_1"]
            elif has_dual_buttons and site_name == "coffee_liquid_right":
                control_state = button_press_states["start_button_2"]
            elif has_dual_buttons and site_name == "coffee_liquid":
                control_state = button_press_states["start_button_1"] | button_press_states["start_button_2"]
            else:
                continue

            for i, site in enumerate(sites_for_name):
                if site is not None and site.IsValid():
                    if control_state[i]:
                        site.GetAttribute("visibility").Set("inherited")
                        site.GetAttribute("purpose").Set("default")
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
        obj_poses = env.scene.rigid_objects[obj_name].data.body_com_pos_w[:, 0, :]
        pour_site_poses = T.euler2mat(torch.tensor([0, 0, self.rot], device=self.device, dtype=torch.float32)) @ \
            torch.tensor(self.get_reset_regions()["bottom"]["offset"], device=self.device, dtype=torch.float32) + \
            torch.tensor(self.pos, device=self.device, dtype=torch.float32) + \
            env.scene.env_origins
        # pour_site_poses = self.pour_sites()
        xy_check = torch.norm(obj_poses[:, 0:2] - pour_site_poses[:, 0:2], dim=-1) < xy_thresh
        z_check = torch.abs(obj_poses[:, 2] - pour_site_poses[:, 2]) < 0.10
        return xy_check & z_check

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
        for name in self._start_button_names:
            button_pos = torch.tensor([usd.get_prim_pos_rot_in_world(self.start_button_infos[name][0])[0]], device=env.device) + env.scene.env_origins  # (env_num, 3)
            button_pos = button_pos.unsqueeze(1)  # (env_num, 1, 3)
            gripper_button_far = torch.norm(gripper_site_pos - button_pos, dim=-1) > th  # (env_num, ee_num)
            result &= torch.all(gripper_button_far, dim=-1)  # (env_num,)

        return result

    @cached_property
    def coffee_liquid_sites(self):
        sites_dict = {}
        for site_name in self._coffee_liquid_site_names:
            sites_list = []
            for prim_path in self.prim_paths:
                sites_prim = usd.get_prim_by_name(self._env.sim.stage.GetObjectAtPath(prim_path), site_name, only_xform=False)
                for site in sites_prim:
                    if site is not None and site.IsValid():
                        sites_list.append(site)
            sites_dict[site_name] = sites_list
        for site_name in self._coffee_liquid_site_names:
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

    def pour_sites(self):
        """Don't use this function for now"""
        sites_list = []
        for key in self.receptacle_place_sites.keys():
            sub_site_list = self.receptacle_place_sites[key]
            if sub_site_list is None:
                sites_list.append(None)
                continue
            for env_idx, site in enumerate(sub_site_list):
                sites_list.append(self.get_prim_pose_in_world_after_fixed(env_idx, site.GetPrim())[0])
        return torch.tensor(sites_list, device=self.device)

    def get_prim_pose_in_world_after_fixed(self, env_index, prim, parent_level=3):
        """Don't use this function for now"""
        target_prim = prim
        for _ in range(parent_level):
            target_prim = target_prim.GetParent()

        target_usd_pos, target_usd_quat, _ = usd.get_prim_pos_rot_in_world(target_prim)
        target_sim_pose = self._env.scene[target_prim.GetName()].data.root_com_pose_w
        target_sim_pos, target_sim_quat = target_sim_pose[env_index, :3], target_sim_pose[env_index, 3:7]
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(
            torch.tensor(target_usd_pos, device=self._env.device).unsqueeze(0),
            torch.tensor(target_usd_quat, device=self._env.device).unsqueeze(0),
            target_sim_pos.unsqueeze(0),
            target_sim_quat.unsqueeze(0),
        )

        prim_usd_pos, prim_usd_quat, _ = usd.get_prim_pos_rot_in_world(prim)
        prim_sim_pos, prim_sim_quat = math_utils.combine_frame_transforms(
            torch.tensor(prim_usd_pos, device=self._env.device).unsqueeze(0),
            torch.tensor(prim_usd_quat, device=self._env.device).unsqueeze(0),
            relative_pos,
            relative_quat,
        )

        prim_sim_pos = prim_sim_pos.squeeze(0)
        prim_sim_quat = prim_sim_quat.squeeze(0)

        return prim_sim_pos.cpu().tolist(), prim_sim_quat.cpu().tolist()

    @cached_property
    def start_button_infos(self):
        start_button_infos = {}
        for prim_path in self.prim_paths:
            root_prim = self._env.sim.stage.GetObjectAtPath(prim_path)
            button_prims = usd.get_prim_by_prefix(root_prim, "start_button", only_xform=False)
            for button_prim in button_prims:
                if button_prim is not None and button_prim.IsValid():
                    if button_prim.GetName() not in start_button_infos:
                        start_button_infos[button_prim.GetName()] = [button_prim]
                    else:
                        start_button_infos[button_prim.GetName()].append(button_prim)
        assert len(start_button_infos) > 0, f"Coffee Machine Start Button not found!"
        return start_button_infos

    @property
    def nat_lang(self):
        return "coffee machine"
