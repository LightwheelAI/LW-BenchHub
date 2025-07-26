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

from termcolor import colored
from tasks.base import BaseTaskEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm


class BaseUsdTaskCfg(BaseTaskEnvCfg):

    task_name: str = "BaseUsdTaskCfg"

    def __post_init__(self):
        super().__post_init__()

        def init_kitchen(env, env_ids):
            print(colored(self.get_ep_meta()["lang"], "green"))

        self.events.init_kitchen = EventTerm(func=init_kitchen, mode="startup")

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["object_cfgs"] = []
        ep_meta[
            "lang"
        ] = f"Usd-specified scene"
        return ep_meta
