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
import os
import numpy as np
import torch
from dataclasses import MISSING
from copy import deepcopy
from pathlib import Path
import shutil
import traceback
from collections import namedtuple

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg
from lwlab.core.tasks.base import TerminationsCfg

from ..base import BaseSceneEnvCfg
from lwlab.core.context import get_context
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
from lwlab.utils.env import ExecuteMode
from lwlab.utils.isaaclab_utils import NoDeepcopyMixin
from lwlab.utils.usd_utils import OpenUsd as usd
from lightwheel_sdk.loader import ENDPOINT
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn
import lwlab.utils.math_utils.transform_utils.torch_impl as Tt
from lwlab.utils.log_utils import get_error_logger
from lwlab.core.checks.checker_factory import get_checkers_from_cfg, form_checker_result
import lwlab.utils.place_utils.env_utils as EnvUtils
from lwlab.utils.place_utils.kitchen_object_utils import extract_failed_object_name, recreate_object
from lwlab.core.scenes.kitchen.kitchen_arena import KitchenArena
from lwlab.utils.errors import SamplingError
from lwlab.core.models.fixtures.fixture import FixtureType, Fixture
from lwlab.utils.fixture_utils import fixture_is_type
from lwlab.utils.place_utils.usd_object import USDObject
from typing import Dict, List, Any, Optional, Union

from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab.utils import configclass
from lwlab.utils.log_utils import copy_dict_for_json
from isaaclab.assets import AssetBaseCfg

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


class LwLabScene(Scene, NoDeepcopyMixin):
    def __init__(self,
                 num_envs: int = 1,
                 device: str = "cpu",
                 excute_mode: ExecuteMode = ExecuteMode.TELEOP,
                 **kwargs,
                 ):
        super().__init__()
        self.context = get_context()
        self.scene_name = self.context.scene_name
        self.layout_id: int = None
        self.style_id: int = None
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
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
            self.scene_type = self._ep_meta["scene_type"] if "scene_type" in self._ep_meta else "robocasakitchen"
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
        self.lwlab_arena = KitchenArena(
            layout_id=self.layout_id,
            style_id=self.style_id,
            exclude_layouts=orchestrator.task.exclude_layouts,
            enable_fixtures=orchestrator.task.enable_fixtures,
            movable_fixtures=orchestrator.task.movable_fixtures,
            scene_cfg=self,
            scene_type=self.scene_type,
            layout_registry_names=orchestrator.task.layout_registry_names,
        )
        self.usd_path = self.lwlab_arena.usd_path
        self.layout_id = self.lwlab_arena.layout_id
        self.style_id = self.lwlab_arena.style_id
        self.scene_type = self.lwlab_arena.scene_type
        self.fixture_cfgs = self.lwlab_arena.get_fixture_cfgs()
        self.floorplan_version = self.lwlab_arena.version_id
        self.fxtr_placements = usd.get_fixture_placements(self.lwlab_arena.stage.GetPseudoRoot(), self.fixture_cfgs, self.fixtures)

        if self.lwlab_arena.layout_id in orchestrator.task.exclude_layouts:
            raise ValueError(f"Layout {self.lwlab_arena.layout_id} is excluded in task {self.task_name}")

        background = Background(
            name=self.scene_type,
            usd_path=self.lwlab_arena.usd_path,
            object_min_z=0.1,
        )

        # flush self.assets
        self.assets = {}
        self.add_asset(background)

    def set_ep_meta(self, meta):
        self._ep_meta = meta

    def get_ep_meta(self):
        ep_meta = {}
        ep_meta.update(deepcopy(self._ep_meta))
        ep_meta["scene_type"] = self.scene_type
        ep_meta["usd_path"] = self.usd_path
        ep_meta["layout_id"] = (
            self.layout_id if isinstance(self.layout_id, dict) else int(self.layout_id)
        )
        ep_meta["style_id"] = (
            self.style_id if isinstance(self.style_id, dict) else int(self.style_id)
        )

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
        env_cfg.sim.physx.bounce_threshold_velocity = 0.2
        env_cfg.sim.physx.bounce_threshold_velocity = 0.01
        env_cfg.sim.physx.friction_correlation_distance = 0.00625
        env_cfg.sim.render.enable_translucency = True

        # add light in scene(if never added)
        if not hasattr(env_cfg.scene, "light"):
            light = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/light",
                spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=9000.0),
            )
            setattr(env_cfg.scene, "light", light)

        if self.context.execute_mode == ExecuteMode.TELEOP:
            env_cfg.ui_window_class_type = None

        return env_cfg
