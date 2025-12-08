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

from isaaclab_arena.assets.background import Background
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.scene.scene import Scene

from lw_benchhub.core.context import get_context
from lw_benchhub.utils.isaaclab_utils import NoDeepcopyMixin


class LocalScene(Scene, NoDeepcopyMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.context = get_context()
        self.scene_usd_path = self.context.scene_name
        self.scene_type = "local"
        self.fixtures = {}
        self.fxtr_placements = {}
        self.is_replay_mode = False
        assert self.scene_usd_path.endswith(".usd"), "Scene USD path must end with .usd"

    def setup_env_config(self, orchestrator):
        background = Background(
            name=self.scene_type,
            usd_path=self.scene_usd_path,
            object_min_z=0.1,
        )
        # flush self.assets
        self.assets = {}
        self.add_asset(background)

    def get_ep_meta(self):
        return {
            "floorplan_version": None,
        }

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        env_cfg.sim.render.enable_translucency = True
        return env_cfg
