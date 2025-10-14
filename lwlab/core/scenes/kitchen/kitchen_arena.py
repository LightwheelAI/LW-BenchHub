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

# base class for kitchen arena

import os
import time
import random

from lightwheel_sdk.loader import floorplan_loader
from lwlab.utils.usd_utils import OpenUsd as usd
from lwlab.core.models.scenes.scene_parser import parse_fixtures
from lwlab.utils.usd_utils import usd_simplify


class KitchenArena:
    """
    Kitchen arena class holding all of the fixtures

    Args:
        layout_id (int or LayoutType): layout of the kitchen to load

        style_id (int or StyleType): style of the kitchen to load

        scene_cfg (RoboCasaSceneCfg): scene configuration
    """

    def __init__(self,
                 layout_id,
                 style_id,
                 exclude_layouts=[],
                 enable_fixtures=None,
                 removable_fixtures=None,
                 ref_fixture_types=None,
                 ref_fixture_ids=None,
                 usd_simplify=False,
                 scene_cfg=None,
                 ):
        # download floorplan usd
        self.scene_cfg = scene_cfg
        self.enable_fixtures = enable_fixtures
        self.removable_fixtures = removable_fixtures
        self.ref_fixture_types = ref_fixture_types
        self.ref_fixture_ids = ref_fixture_ids
        self.ref_fixtures = {}
        self.ref_fixture_names = []
        self.usd_simplify = usd_simplify
        self.floorplan_version = self.scene_cfg.floorplan_usd_version
        self.load_floorplan(layout_id, style_id, exclude_layouts, scene_type=scene_cfg.scene_type)
        self.stage = usd.get_stage(self.usd_path)
        # enable / movable fixtures in usd
        is_updated_usd = self._is_updated_usd()

        # load fixtures & task-related fixture names
        self.fixtures = parse_fixtures(self.stage, scene_cfg.num_envs, scene_cfg.seed, scene_cfg.device)
        ref_fxtr_type_names = {}
        for fxtr_type in self.ref_fixture_types:
            ref_fxtr_type_names[fxtr_type] = [fxtr_name for fxtr_name, fxtr_obj in self.fixtures if fxtr_type in fxtr_obj.fixture_types]

        for fxtr_type, fxtr_names in ref_fxtr_type_names:
            if self.ref_fixture_ids[fxtr_type] is not None:
                selected_fxtr_name = fxtr_names[self.ref_fixture_ids[fxtr_type]]
            else:
                selected_fxtr_name = random.choice(fxtr_names)
            self.ref_fixture_names.append(selected_fxtr_name)
            self.ref_fixtures.update({selected_fxtr_name: self.fixtures[selected_fxtr_name]})

        # usd simplified / export
        if is_updated_usd and self.usd_simplify:
            self.stage = usd_simplify(self.stage, self.ref_fixture_names)
            dir_name = os.path.dirname(self.usd_path)
            base_name = os.path.basename(self.usd_path)
            new_path = os.path.join(dir_name, base_name.replace(".usd", "_modified.usd"))
            self.stage.GetRootLayer().Export(new_path)
            self.usd_path = new_path

    def _is_updated_usd(self):
        is_updated_usd = False
        if self.scene_cfg.enable_fixtures is not None:
            is_updated_usd = True
            for fixture in self.scene_cfg.enable_fixtures:
                usd.activate_prim(self.stage, fixture)
        if self.scene_cfg.removable_fixtures is not None:
            is_updated_usd = True
            root_prim = self.stage.GetPseudoRoot()
            for fixture in self.scene_cfg.removable_fixtures:
                prims = usd.get_prim_by_prefix(root_prim, fixture)
                for prim in prims:
                    fixed_joints = usd.get_prim_by_type(prim, include_types=["PhysicsFixedJoint"])
                    for fix_joint_prim in fixed_joints:
                        fix_joint_prim.SetActive(False)
        return is_updated_usd

    def get_fixture_cfgs(self):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for (name, fxtr) in self.fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs

    def load_floorplan(self, layout_id, style_id, exclude_layouts=[]):
        start_time = time.time()
        print(f"load floorplan usd", end="...")
        if layout_id is None:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_type, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)
        elif style_id is None:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)
        else:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_type, layout_id=layout_id, style_id=style_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)
        usd_path, self.floorplan_meta = res.result()
        self.usd_path = str(usd_path)
        self.backend = self.floorplan_meta.get("backend")
        self.scene_type = self.floorplan_meta.get("scene")
        self.layout_id = self.floorplan_meta.get("layout_id")
        self.style_id = self.floorplan_meta.get("style_id")
        self.version_id = self.floorplan_meta.get("version_id")
        print(f"{self.backend}-{self.scene_type}-{self.layout_id}-{self.style_id} loaded in {time.time() - start_time:.2f}s\n")
        print(f"version_id: {self.version_id}")
