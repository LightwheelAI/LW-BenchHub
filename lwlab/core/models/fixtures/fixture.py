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
from typing import Dict, Any
from collections import defaultdict

from robocasa.models.fixtures.fixture import Fixture as RoboCasaFixture
from robocasa.models.fixtures.fixture_stack import STACKABLE
from robocasa.models.scenes.scene_builder import FIXTURES
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg

import lwlab.utils.object_utils as OU
from lwlab.utils.usd_utils import OpenUsd as usd


# adapt for multiple names for the same fixture class
fixture_class_to_names: Dict[Any, list] = defaultdict(list)
for k, v in FIXTURES.items():
    fixture_class_to_names[v].append(k)


class Fixture:
    def __deepcopy__(self, memo):
        return self

    def __init_subclass__(cls) -> None:
        fixture_found = False
        for kls in cls.mro()[1:]:
            if issubclass(kls, Fixture):
                fixture_found = True

            elif (kls is not RoboCasaFixture and issubclass(kls, RoboCasaFixture)):
                if not fixture_found:
                    raise ValueError(f"Robocasa Fixture must be inherited after Fixture")
                names = fixture_class_to_names[kls]
                for name in names:
                    FIXTURES[name] = cls
                    if name in STACKABLE:
                        STACKABLE[name] = cls
                break
            else:
                # no robocasa fixture found
                raise ValueError(f"Robocasa Fixture must be inherited")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg, root_prim):

        prim = usd.get_prim_by_name(root_prim, self.name)

        if not prim:
            print(f"prim {self.name} not found")
            self.fixture_name = self.name
            return

        assert len(prim) == 1
        prim = prim[0]

        if not usd.has_contact_reporter(prim) and not usd.is_articulation_root(prim) and not usd.is_rigidbody(prim):
            self.fixture_name = self.name
            return

        if usd.has_contact_reporter(prim) and usd.is_rigidbody(prim):
            prim_path = f"{{ENV_REGEX_NS}}/Scene/{self.name}"
            fixture_contact_sensor = ContactSensorCfg(
                prim_path=prim_path,
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                filter_prim_paths_expr=[],
            )
            setattr(cfg.scene, f"{self.name}_contact", fixture_contact_sensor)
            return
        if not usd.is_articulation_root(prim) and not usd.is_rigidbody(prim):
            return

        corpus = usd.get_prim_by_name(prim, "corpus")

        if not corpus:
            print(f"corpus in {self.name} not found")
            self.fixture_name = usd.get_child_commonprefix_name(prim)
            if self.fixture_name:
                prim_path = f"{{ENV_REGEX_NS}}/Scene/{self.name}/{self.fixture_name}"
                fixture_contact_sensor = ContactSensorCfg(
                    prim_path=prim_path,
                    update_period=0.0,
                    history_length=1,
                    debug_vis=False,
                    filter_prim_paths_expr=[],
                )
                setattr(cfg.scene, f"{self.name}_contact", fixture_contact_sensor)
            else:
                print("error: not regular asset")

        # reset joint infos
        self._joint_infos = {}
        unfix_joints = usd.get_all_joints_without_fixed(prim)
        for joint in unfix_joints:
            self._joint_infos[joint.GetName()] = {} if joint.GetAttribute("physics:lowerLimit").Get() is None \
                else {"range": torch.tensor([joint.GetAttribute("physics:lowerLimit").Get() * torch.pi / 180,
                                             joint.GetAttribute("physics:upperLimit").Get() * torch.pi / 180])}

    def setup_env(self, env: ManagerBasedRLEnv):
        if self.name in env.scene.extras.keys():
            self.prim_paths = env.scene.extras[self.name].prim_paths
        elif self.name in env.scene.articulations.keys():
            self.prim_paths = env.scene.articulations[self.name]._root_physx_view.prim_paths
        else:
            self.prim_paths = None

    def get_joint_state(self, env, joint_names, env_ids=None):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps door names to a percentage of how open they are
        """
        joint_state = dict()

        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, joint_idx]
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]
            # convert to normalized joint value
            norm_qpos = OU.normalize_joint_value(
                joint_qpos,
                joint_min=joint_min,
                joint_max=joint_max,
            )
            if joint_min < 0:
                norm_qpos = 1 - norm_qpos
            joint_state[j_name] = norm_qpos

        return joint_state

    def set_joint_state(self, min, max, env, joint_names, env_ids=None, rng=None):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is
        Args:
            min (float): minimum percentage of how open the door is
            max (float): maximum percentage of how open the door is
            env (ManagerBasedRLEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max
        rng = self.rng if rng is None else rng

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
            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[rng.uniform(desired_min.cpu(), desired_max.cpu())]]).to(env.device),
                torch.tensor([joint_idx]).to(env.device),
                torch.as_tensor(env_ids).to(env.device) if env_ids is not None else None
            )

    def is_open(self, env, joint_names=None, th=0.90):
        if joint_names is None:
            joint_names = self.door_joint_names
        joint_state = self.get_joint_state(env, joint_names)
        is_open = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for j_name in joint_names:
            assert j_name in joint_state
            norm_qpos = joint_state[j_name]
            is_open = is_open & (norm_qpos >= th)
        return is_open

    def is_closed(self, env, joint_names=None, th=0.005):
        if joint_names is None:
            joint_names = self.door_joint_names
        joint_state = self.get_joint_state(env, joint_names)
        is_closed = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for j_name in joint_names:
            assert j_name in joint_state
            norm_qpos = joint_state[j_name]
            is_closed = is_closed & (norm_qpos <= th)
        return is_closed

    def open_door(self, env, min=0.90, max=1.0, env_ids=None, rng=None):
        """
        helper function to open the door. calls set_door_state function
        """
        self.set_joint_state(
            env=env, min=min, max=max, joint_names=self.door_joint_names, env_ids=env_ids, rng=rng
        )

    def close_door(self, env, min=0.0, max=0.0, env_ids=None, rng=None):
        """
        helper function to close the door. calls set_door_state function
        """
        self.set_joint_state(
            env=env, min=min, max=max, joint_names=self.door_joint_names, env_ids=env_ids, rng=rng
        )

    def get_door_state(self, env, joint_names=None, env_ids=None):
        if joint_names is None:
            joint_names = self.door_joint_names
        return self.get_joint_state(env, joint_names, env_ids=env_ids)

    def set_door_state(self, min, max, env, env_ids=None, rng=None):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is

        Args:
            min (float): minimum percentage of how open the door is

            max (float): maximum percentage of how open the door is

            env (MujocoEnv): environment
        """
        self.set_joint_state(
            env=env, min=min, max=max, joint_names=self.door_joint_names, env_ids=env_ids, rng=rng
        )

    @property
    def door_joint_names(self):
        return [j_name for j_name in self._joint_infos if "door" in j_name]

    @property
    def nat_lang(self):
        return self.name
