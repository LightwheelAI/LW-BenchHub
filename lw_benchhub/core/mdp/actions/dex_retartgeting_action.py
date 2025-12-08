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

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Callable, List

import numpy as np
import torch
import yaml

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from lw_benchhub.data import LW_BENCHHUB_DATA_PATH


class DexRetargetingAction(ActionTerm):
    cfg: "DexRetargetingActionCfg"
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: "DexRetargetingActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        from dex_retargeting.retargeting_config import RetargetingConfig, SeqRetargeting
        with self.cfg.config_path.open("r") as f:
            retargeting_config = yaml.safe_load(f)
        RetargetingConfig.set_default_urdf_dir(self.cfg.config_path.parent)
        self.dex_retargeting_cfg = RetargetingConfig.from_dict(retargeting_config)
        self.dex_retargetings: List[SeqRetargeting] = [None] * self.num_envs
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions
        finger_pos = self.raw_actions.reshape([self.num_envs, -1, 3])
        if self.dex_retargeting_cfg.type != "dexpilot":
            self._fingers_actions[:, :] = finger_pos
        else:
            self._fingers_actions[:, -self.num_fingers:, :] = finger_pos
            for i in range(self.num_fingers - 1):
                for j in range(i + 1, self.num_fingers):
                    self._fingers_actions[:, i, :] = finger_pos[:, j, :] - finger_pos[:, i, :]

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_ = list(env_ids) if env_ids is not None else list(range(self.num_envs))

        for env_id in env_ids_:
            self.dex_retargetings[env_id] = self.dex_retargeting_cfg.build()
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        if self.dex_retargeting_cfg.type == "dexpilot":
            processed_actions = torch.zeros(len(env_ids_), self.num_fingers + len(self.dex_retargetings[0].optimizer.projected), 3, device=self.device)
        else:
            processed_actions = torch.zeros(len(env_ids_), self.num_fingers, 3, device=self.device)
        if env_ids is None:
            self._fingers_actions = processed_actions
        else:
            self._fingers_actions[env_ids, :] = processed_actions

    def apply_actions(self):
        processed_actions = self._fingers_actions.cpu().numpy()
        if np.sum(abs(processed_actions)) > 0.01:
            retargeted_actions = np.array([
                dex_retargeting.retarget(processed_actions[i])[self.cfg.retargeting_index]
                for i, dex_retargeting in enumerate(self.dex_retargetings)
            ])
            joint_pos_rad = self.cfg.post_process_fn(retargeted_actions, len(self._joint_ids))
            joint_pos_rad = torch.tensor(joint_pos_rad, dtype=torch.float32, device=self.processed_actions.device)
            self._asset.set_joint_position_target(joint_pos_rad, self._joint_ids)

    @property
    def num_fingers(self) -> int:
        if self.dex_retargeting_cfg.type == "dexpilot":
            assert self.dex_retargeting_cfg.finger_tip_link_names
            return len(self.dex_retargeting_cfg.finger_tip_link_names)
        elif self.dex_retargeting_cfg.type == "position":
            assert self.dex_retargeting_cfg.target_link_names
            return len(self.dex_retargeting_cfg.target_link_names)
        elif self.dex_retargeting_cfg.type == "vector":
            assert self.dex_retargeting_cfg.target_origin_link_names
            return len(self.dex_retargeting_cfg.target_origin_link_names)
        else:
            raise ValueError(f"Unsupported retargeting type: {self.dex_retargeting_cfg.type}")

    @property
    def finger_names(self) -> list[str]:
        if self.dex_retargeting_cfg.type == "dexpilot":
            return self.dex_retargeting_cfg.finger_tip_link_names
        elif self.dex_retargeting_cfg.type == "position":
            return self.dex_retargeting_cfg.target_link_names
        elif self.dex_retargeting_cfg.type == "vector":
            return self.dex_retargeting_cfg.target_origin_link_names
        else:
            raise ValueError(f"Unsupported retargeting type: {self.dex_retargeting_cfg.type}")

    @property
    def action_dim(self) -> int:
        num = self.num_fingers
        # if self.dex_retargeting_cfg.type == "dexpilot":
        #     num += len(self.dex_retargeting.optimizer.projected)
        return num * 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions


@configclass
class DexRetargetingActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = DexRetargetingAction
    config_name: str = MISSING
    retargeting_index: list[int] = MISSING
    joint_names: list[str] = MISSING
    post_process_fn: Callable = lambda x, y: x

    def __post_init__(self):
        self.config_path = LW_BENCHHUB_DATA_PATH / "dex_retargeting" / f"{self.config_name}.yaml"
