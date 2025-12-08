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

import os
from copy import deepcopy

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab_arena.assets.background import Background
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from isaaclab_arena.scene.scene import Scene

from lw_benchhub.core.context import get_context
from lw_benchhub.core.models.scenes.scene_parser import parse_fixtures, get_fixture_cfgs
from lw_benchhub.core.scenes.kitchen.kitchen_arena import KitchenArena
from lw_benchhub.utils.env import ExecuteMode
from lw_benchhub.utils.isaaclab_utils import NoDeepcopyMixin
from lw_benchhub.utils.usd_utils import OpenUsd as usd


'''
What second stage need to do:

1. Scene USD Change:
    - enable_fixtures (in KitchenArena)
    - removable_fixtures (in KitchenArena)
    - usd_simplify (in KitchenArena)
2. Placement:
    - object placement
    - robot placement
    - retry (init scene_cfg / task_cfg again)
'''


class LwScene(Scene, NoDeepcopyMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.context = get_context()
        self.num_envs = self.context.num_envs
        self.scene_name = self.context.scene_name
        self.scene_backend = self.context.scene_backend
        self.layout_id: int = None
        self.style_id: int = None
        self.local_scene_path: str = None
        self._curr_gen_fixtures: str = None
        self.fixture_refs = {}
        self.is_replay_mode = self.context.execute_mode in [ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE]
        self._setup_config()

    # first stage (just init from itself)
    def _setup_config(self):
        self.floorplan_version = None
        self._ep_meta = {}
        if self.context.replay_cfgs is not None and "ep_meta" in self.context.replay_cfgs:
            self.set_ep_meta(self.context.replay_cfgs["ep_meta"])
            if "cache_usd_version" in self._ep_meta:
                self.floorplan_version = self._ep_meta["cache_usd_version"]["floorplan_version"]

        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta.get("layout_id", None)
            self.style_id = self._ep_meta.get("style_id", None)
            self.scene_type = self._ep_meta.get("scene_type", "robocasakitchen")
        else:
            if self.scene_name.endswith(".usd"):
                self.local_scene_path = self.scene_name
                self.scene_type = "usd"
            else:
                scene_name_split = self.scene_name.split("-")
                if len(scene_name_split) == 3:
                    self.scene_type, self.layout_id, self.style_id = scene_name_split
                elif len(scene_name_split) == 2:
                    self.scene_type, self.layout_id = scene_name_split
                elif len(scene_name_split) == 1:
                    self.scene_type = scene_name_split[0]
                else:
                    raise ValueError(f"Invalid scene name: {self.scene_name}")

        self.layout_id = int(self.layout_id) if self.layout_id is not None else None
        self.style_id = int(self.style_id) if self.style_id is not None else None

    # second stage (init from ArenaEnvironment)
    def setup_env_config(self, orchestrator):
        self._setup_kitchen_arena(orchestrator)
        self.scene_range = self.lw_benchhub_arena.scene_range
        self.layout_id = self.lw_benchhub_arena.layout_id
        self.style_id = self.lw_benchhub_arena.style_id
        self.scene_type = self.lw_benchhub_arena.scene_type
        self.fixture_cfgs = get_fixture_cfgs(self)
        self.floorplan_version = self.lw_benchhub_arena.version_id
        self.fxtr_placements = usd.get_fixture_placements(self.lw_benchhub_arena.stage.GetPseudoRoot(), self.fixture_cfgs)

        if self.lw_benchhub_arena.layout_id in orchestrator.task.exclude_layouts:
            raise ValueError(f"Layout {self.lw_benchhub_arena.layout_id} is excluded in task {self.task_name}")

        background = Background(
            name="Scene",
            usd_path=self.scene_usd_path,
            object_min_z=0.1,
        )

        # flush self.assets
        self.assets = {}
        self.add_asset(background)

    def _setup_kitchen_arena(self, orchestrator):
        self.lw_benchhub_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            floorplan_version=self.floorplan_version,
            exclude_layouts=orchestrator.task.exclude_layouts,
            enable_fixtures=orchestrator.task.enable_fixtures,
            movable_fixtures=orchestrator.task.movable_fixtures,
            scene_type=self.scene_type,
            local_scene_path=self.local_scene_path,
        )
        self.scene_usd_path = self.lw_benchhub_arena.usd_path
        self.fixtures = parse_fixtures(self.lw_benchhub_arena.stage, self.context.num_envs, self.context.seed, self.context.device)

    def set_ep_meta(self, meta):
        self._ep_meta = meta

    def get_ep_meta(self):
        ep_meta = {}
        ep_meta.update(deepcopy(self._ep_meta))
        ep_meta["scene_type"] = self.scene_type
        ep_meta["usd_path"] = self.scene_usd_path
        ep_meta["layout_id"] = self.layout_id
        ep_meta["style_id"] = self.style_id
        ep_meta["fixtures"] = {
            k: {"cls": v.__class__.__name__} for (k, v) in self.fixtures.items()
        }
        # export actual init pose if available in this episode, otherwise omit
        ep_meta["floorplan_version"] = self.floorplan_version
        return ep_meta

    def modify_env_cfg(self, env_cfg: IsaacLabArenaManagerBasedRLEnvCfg):
        """
        modify the environment configuration
        """
        env_cfg.num_envs = self.num_envs
        env_cfg.scene_backend = self.scene_backend

        env_cfg.sim.physx.bounce_threshold_velocity = 0.2
        env_cfg.sim.physx.bounce_threshold_velocity = 0.01
        env_cfg.sim.physx.friction_correlation_distance = 0.00625
        env_cfg.sim.render.enable_translucency = True

        scene_origin = np.mean(self.scene_range, axis=0)

        # sphere light is only work on RL
        if self.context.execute_mode in [ExecuteMode.TRAIN, ExecuteMode.EVAL]:
            env_cfg.scene.room_light = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/room_light",
                spawn=sim_utils.SphereLightCfg(
                    radius=0.4,
                    color=(0.75, 0.75, 0.75),
                    intensity=50000.0,
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(scene_origin[0], scene_origin[1], scene_origin[2] * 2)),
            )

            env_cfg.scene.global_light = AssetBaseCfg(
                prim_path="/World/global_light",
                spawn=sim_utils.DomeLightCfg(
                    color=(0.75, 0.75, 0.75),
                    intensity=500.0,
                    visible_in_primary_ray=True,
                ),
            )
        else:
            env_cfg.scene.global_light = AssetBaseCfg(
                prim_path="/World/global_light",
                spawn=sim_utils.DomeLightCfg(
                    color=(0.75, 0.75, 0.75),
                    intensity=9000.0,
                    visible_in_primary_ray=True,
                ),
            )

        if self.context.execute_mode == ExecuteMode.TELEOP:
            env_cfg.ui_window_class_type = None

        return env_cfg


class TestAssetScene(LwScene):
    # test fixture / object variables
    def __init__(self):
        super().__init__()
        self.test_fixture_path = self.context.test_fixture_path
        self.test_fixture_type = self.context.test_fixture_type

    def _replace_fixtures(self, arena):
        root_prim = arena.stage.GetPseudoRoot()
        prims = usd.get_prim_by_prefix(root_prim, self.test_fixture_type, only_xform=True)
        assert len(prims) > 0, "Fixture type not found in the scene"
        if self.test_fixture_path.endswith(".usd"):
            usd.replace_prim(arena.stage, prims[0], os.path.abspath(self.test_fixture_path))
        else:
            from lightwheel_sdk.loader import object_loader
            fixture_path, fixture_name, fixture_res = object_loader.acquire_by_registry(
                "fixtures",
                source=self.sources,
                file_name=self.test_fixture_path.split("/")[-1],
            )
            usd.replace_prim(arena.stage, prims[0], fixture_path)
        updated_usd_path = arena.usd_path.replace(".usd", "_fixture_test.usd")
        arena.stage.GetRootLayer().Export(updated_usd_path)
        return updated_usd_path

    def _setup_kitchen_arena(self, orchestrator):
        super()._setup_kitchen_arena(orchestrator)
        if self.context.execute_mode == ExecuteMode.TEST_FIXTURE:
            self.scene_usd_path = self._replace_fixtures(self.lw_benchhub_arena)
