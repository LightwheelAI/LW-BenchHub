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
from .fixture import Fixture
import lwlab.utils.object_utils as OU
from robocasa.models.fixtures.cabinets import Cabinet as RoboCasaCabinet
from robocasa.models.fixtures.cabinets import SingleCabinet as RoboCasaSingleCabinet
from robocasa.models.fixtures.cabinets import HingeCabinet as RoboCasaHingeCabinet
from robocasa.models.fixtures.cabinets import OpenCabinet as RoboCasaOpenCabinet
from robocasa.models.fixtures.cabinets import Drawer as RoboCasaDrawer
from robocasa.models.fixtures.cabinets import PanelCabinet as RoboCasaPanelCabinet
from robocasa.models.fixtures.cabinets import HousingCabinet as RoboCasaHousingCabinet


class Cabinet(Fixture, RoboCasaCabinet):
    def set_door_state(self, min, max, env, env_ids=None, rng=None):
        pass


class SingleCabinet(Cabinet, RoboCasaSingleCabinet):
    pass


class HingeCabinet(Cabinet, RoboCasaHingeCabinet):

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

    def set_door_state(self, min, max, env, env_ids=None, rng=None):
        """
        Sets how open the doors are. Chooses a random amount between min and max.
        Min and max are percentages of how open the doors are

        Args:
            min (float): minimum percentage of how open the door is

            max (float): maximum percentage of how open the door is

            env (ManagerBasedRLEnv): environment

            rng (np.random.Generator): random number generator
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = np.pi / 2

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        uniform_values = [rng.uniform(desired_min.cpu(), desired_max.cpu()) for _ in self._joint_infos.keys()]

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


class OpenCabinet(Cabinet, RoboCasaOpenCabinet):
    # Done
    pass


class Drawer(Cabinet, RoboCasaDrawer):

    def update_state(self, env):
        """
        Updates the interior bounding boxes of the drawer to be matched with
        how open the drawer is. This is needed when determining if an object
        is inside the drawer or when placing an object inside an open drawer.

        TODO: (1) Drawer assets require inner volume and outer volume. (floorplan2usd)
        TODO: (2) In runtime to get the world coordinates and dimensions of the geometric 
                  centroids of the inner and outer volumes in real time, and is to be time
                  consuming small. (fixture controller)

        Args:
            env (ManagerBasedRLEnv): environment
        """
        int_sites = {}

    def open_door(self, env, env_ids=None, min=0.9, max=1, partial_open=False):
        if partial_open:
            min *= 0.3
            max *= 0.3
        return super().open_door(env, min, max, env_ids=env_ids)

    def set_door_state(self, min, max, env, env_ids=None, rng=None):
        """
        Sets how open the drawer is. Chooses a random amount between min and max.
        Min and max are percentages of how open the drawer is.

        TODO: self.size needs to be rewritten. (for fully remove robocasa)

        Args:
            min (float): minimum percentage of how open the drawer is

            max (float): maximum percentage of how open the drawer is

            env (ManagerBasedRLEnv): environment

            env_ids (torch.Tensor): environment ids

            rng (np.random.Generator): random number generator
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = self.size[1] * 0.55  # dont want it to fully open up

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        joint_idx = env.scene.articulations[self.name].joint_names.index("drawer_door_joint")

        env.scene.articulations[self.name].write_joint_position_to_sim(
            torch.tensor([[self.rng.uniform(desired_min.cpu(), desired_max.cpu())]]).to(env.device),
            joint_id=torch.tensor([joint_idx]).to(env.device),
            env_ids=env_ids.to(env.device) if env_ids is not None else None
        )

    def get_door_state(self, env):
        """
        TODO: self.size needs to be rewritten. (for fully remove robocasa)

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


class PanelCabinet(Cabinet, RoboCasaPanelCabinet):

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


class HousingCabinet(Cabinet, RoboCasaHousingCabinet):
    pass
