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

import lw_benchhub.utils.object_utils as OU

from .fixture import ProcGenFixture
from .fixture_types import FixtureType


class Cabinet(ProcGenFixture):
    fixture_types = [FixtureType.CABINET, FixtureType.CABINET_WITH_DOOR, FixtureType.CABINET_SINGLE_DOOR, FixtureType.CABINET_DOUBLE_DOOR]

    def _is_fixture_type(self, fixture_type: FixtureType) -> bool:
        """
        check if the fixture is of the given type
        this function is called by fixture_is_type in fixture_utils.py
        """
        reset_regions = self.get_reset_regions(
            z_range=(1.0, 1.50)
        )  # find reset regions within bounds

        return (
            super()._is_fixture_type(fixture_type) and
            "stack" not in self.name and
            not self.is_corner_cab and
            len(reset_regions) > 0
        )

    def __init__(self, name, prim, num_envs, is_corner_cab=None, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self.is_corner_cab = is_corner_cab

    def get_reset_region_names(self):
        return ("int", "int_0", "int_1", "int_2", "int_3", "int_4", "int_5")

    def set_door_state(self, min, max, env, env_ids=None):
        pass


class SingleCabinet(Cabinet):
    fixture_types = [FixtureType.CABINET, FixtureType.CABINET_WITH_DOOR, FixtureType.CABINET_SINGLE_DOOR]
    pass


class HingeCabinet(Cabinet):
    fixture_types = [FixtureType.CABINET, FixtureType.CABINET_WITH_DOOR, FixtureType.CABINET_DOUBLE_DOOR]

    def get_state(self, env):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps joint names to joint values
        """
        # angle of two door joints
        state = dict()
        joint_names = env.scene.articulations[self.name].joint_names
        for i, joint_name in enumerate(joint_names):
            state[joint_name] = env.scene.articulations[self.name].data.joint_pos[:, i]
        return state

    def set_door_state(self, min, max, env, env_ids=None):
        """
        Sets how open the doors are. Chooses a random amount between min and max.
        Min and max are percentages of how open the doors are

        Args:
            min (float): minimum percentage of how open the door is

            max (float): maximum percentage of how open the door is

            env (ManagerBasedRLEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = np.pi / 2

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        uniform_values = [self.rng.uniform(float(desired_min), float(desired_max)) for _ in self._joint_infos.keys()]

        env.scene.articulations[self.name].write_joint_position_to_sim(
            torch.tensor([uniform_values]).to(env.device),
            env_ids=env_ids
        )

    def get_door_state(self, env):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps door names to a percentage of how open they are
        """
        joint_names = env.scene.articulations[self.name].joint_names
        state = dict()
        for i, joint_name in enumerate(joint_names):
            joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, i]
            state[joint_name] = OU.normalize_joint_value(joint_qpos, joint_min=0, joint_max=np.pi / 2)
        return state


class OpenCabinet(Cabinet):
    fixture_types = [FixtureType.CABINET]
    pass


class Drawer(Cabinet):
    fixture_types = [FixtureType.DRAWER, FixtureType.TOP_DRAWER]

    def _is_fixture_type(self, fixture_type: FixtureType) -> bool:
        """
        check if the fixture is of the given type
        this function is called by fixture_is_type in fixture_utils.py
        """
        type_check = ProcGenFixture._is_fixture_type(self, fixture_type)
        if fixture_type == FixtureType.TOP_DRAWER:
            # drawer's pos is at the bottom-center of the drawer
            height_check = 0.7 <= self.pos[2] + self.size[2] / 2 <= 0.9
            return type_check and height_check
        return type_check

    def update_state(self, env):
        """
        Updates the interior bounding boxes of the drawer to be matched with
        how open the drawer is. This is needed when determining if an object
        is inside the drawer or when placing an object inside an open drawer.

        Args:
            env (ManagerBasedRLEnv): environment
        """
        door_joint_id = env.scene.articulations[self.name].joint_names.index(self.door_joint_names[0])
        door_qpos = env.scene.articulations[self.name].data.joint_pos[:, door_joint_id].cpu().numpy()
        # suppose drawer joint direction(in its local frame) is along -y axis
        door_qpos = np.stack([np.zeros_like(door_qpos), -door_qpos, np.zeros_like(door_qpos)], axis=-1)
        self._regions["int"]["per_env_offset"] = door_qpos

    def open_door(self, env, env_ids=None, min=0.9, max=1, partial_open=False):
        if partial_open:
            min *= 0.3
            max *= 0.3
        result = super().open_door(env, min, max, env_ids=env_ids)
        self.update_state(env)
        return result

    def set_door_state(self, min, max, env, env_ids=None):
        """
        Sets how open the drawer is. Chooses a random amount between min and max.
        Min and max are percentages of how open the drawer is.

        Args:
            min (float): minimum percentage of how open the drawer is

            max (float): maximum percentage of how open the drawer is

            env (ManagerBasedRLEnv): environment

            env_ids (torch.Tensor): environment ids
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = self.size[1] * 0.55  # dont want it to fully open up

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        joint_idx = env.scene.articulations[self.name].joint_names.index("drawer_door_joint")

        env.scene.articulations[self.name].write_joint_position_to_sim(
            torch.tensor([[self.rng.uniform(float(desired_min), float(desired_max))]]).to(env.device),
            torch.tensor([joint_idx]).to(env.device),
            env_ids=(env_ids.to(env.device) if isinstance(env_ids, torch.Tensor) else env_ids)
        )

    def get_door_state(self, env):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps door name to a percentage of how open the door is
        """
        joint_idx = env.scene.articulations[self.name].joint_names.index("drawer_door_joint")
        hinge_qpos = env.scene.articulations[self.name].data.joint_pos[:, joint_idx]

        # convert to percentages
        door = OU.normalize_joint_value(
            hinge_qpos, joint_min=0, joint_max=self.size[1] * 0.55
        )

        return {
            "door": door,
        }


class PanelCabinet(Cabinet):
    fixture_types = [FixtureType.CABINET]

    def get_state(self, env):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps joint names to joint values
        """
        # angle of two door joints
        state = dict()
        for i, joint_name in enumerate(list(self._joint_infos.keys())):
            joint_idx = env.scene.articulations[self.name].joint_names.index(joint_name)
            state[joint_name] = env.scene.articulations[self.name].data.joint_pos[:, joint_idx]
        return state


class HousingCabinet(Cabinet):
    fixture_types = [FixtureType.CABINET]
