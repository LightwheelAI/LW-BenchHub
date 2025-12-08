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

from lw_benchhub.core.tasks.base import LwTaskBase


class BaseTask(LwTaskBase):
    """
    BaseTask: base task just for teleoperation.

    """

    task_name: str = "BaseTask"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "base task"
        ep_meta["object_cfgs"] = {}
        return ep_meta


class RobocasaBaseTask(LwTaskBase):
    """
    RobocasaBaseTask: base task for robocasa.
    need backend=local, scene from local, and fixed robot pose.
    """
    task_name: str = "RobocasaBaseTask"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "robocasa base task"
        ep_meta["object_cfgs"] = {}
        return ep_meta

    def _check_success(self, env):
        return torch.tensor([False], device=env.device).repeat(env.num_envs)
