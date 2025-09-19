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

from lightwheel_sdk.loader import floorplan_loader
from lwlab.utils.usd_utils import OpenUsd as usd
from lwlab.core.models.scenes.scene_parser import parse_fixtures


class KitchenArena:
    """
    Kitchen arena class holding all of the fixtures

    Args:
        layout_id (int or LayoutType): layout of the kitchen to load

        style_id (int or StyleType): style of the kitchen to load

        scene_cfg (RoboCasaSceneCfg): scene configuration
    """

    def __init__(self, layout_id=None, style_id=None, scene_cfg=None):
        # download floorplan usd
        self.scene_cfg = scene_cfg
        if self.scene_cfg.cache_usd_version is not None and "floorplan_version" in self.scene_cfg.cache_usd_version:
            self.floorplan_version = self.scene_cfg.cache_usd_version["floorplan_version"]
        else:
            self.floorplan_version = None
        self.load_floorplan(layout_id, style_id, self.scene_cfg.EXCLUDE_LAYOUTS)
        self.stage = usd.get_stage(self.usd_path)

        # enable fixtures in usd
        if self.scene_cfg.enable_fixtures is not None:
            for fixture in scene_cfg.enable_fixtures:
                usd.activate_prim(self.stage, fixture)
            dir_name = os.path.dirname(self.usd_path)
            base_name = os.path.basename(self.usd_path)
            new_path = os.path.join(dir_name, base_name.replace(".usd", "_enabled.usd"))
            self.stage.GetRootLayer().Export(new_path)
            self.usd_path = new_path

        # load fixtures
        self.scene_cfg.fixtures = parse_fixtures(self.stage, scene_cfg.num_envs, scene_cfg.seed, scene_cfg.device)

    def get_fixture_cfgs(self):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for (name, fxtr) in self.scene_cfg.fixtures.items():
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
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_name.split("-")[0], version=self.floorplan_version, exclude_layout_ids=exclude_layouts)
        elif style_id is None:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_name.split("-")[0], layout_id=layout_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)
        else:
            res = floorplan_loader.acquire_usd(scene=self.scene_cfg.scene_name.split("-")[0], layout_id=layout_id, style_id=style_id, version=self.floorplan_version, exclude_layout_ids=exclude_layouts)
        usd_path, self.floorplan_meta = res.result()
        self.usd_path = str(usd_path)
        self.backend = self.floorplan_meta.get("backend")
        self.scene_type = self.floorplan_meta.get("scene")
        self.layout_id = self.floorplan_meta.get("layout_id")
        self.style_id = self.floorplan_meta.get("style_id")
        self.version_id = self.floorplan_meta.get("version_id")
        print(f"{self.backend}-{self.scene_type}-{self.layout_id}-{self.style_id} loaded in {time.time() - start_time:.2f}s\n")
        print(f"version_id: {self.version_id}")
