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
import time

import numpy as np
from lightwheel_sdk.loader import floorplan_loader

from lw_benchhub.utils.usd_utils import OpenUsd as usd


class KitchenArena:
    """
    Kitchen arena class holding all of the fixtures

    Args:
        layout_id (int, optional): ID of the kitchen layout to load; may also be a LayoutType.
        style_id (int, optional): ID of the kitchen style to load; may also be a StyleType.
        cache_usd_version (dict, optional): Dictionary with USD file versioning information, e.g. {"floorplan_version": ...}.
        exclude_layouts (list, optional): List of layout IDs or types to exclude from selection.
        enable_fixtures (list, optional): List of fixture names to explicitly enable in the floorplan.
        removable_fixtures (list, optional): List of fixture names to make removable (i.e., deactivate joints).
        scene_type (str, optional): Type of the scene, e.g. "kitchen" or "usd".
        local_scene_path (str, optional): Path to the local USD file to load.
    """

    def __init__(self,
                 layout_id: int | None = None,
                 style_id: int | None = None,
                 floorplan_version: str | None = None,
                 exclude_layouts: list[int] | None = None,
                 exclude_styles: list[int] | None = None,
                 enable_fixtures: list[str] | None = None,
                 movable_fixtures: list[str] | None = None,
                 scene_type: str | None = None,
                 local_scene_path: str | None = None
                 ):
        # download floorplan usd
        self.load_floorplan(layout_id, style_id, floorplan_version, exclude_layouts, exclude_styles, scene_type, local_scene_path)
        self.stage = usd.get_stage(self.usd_path)

        scene_aabb = usd.get_prim_aabb_bounding_box(self.stage.GetPseudoRoot())
        self.scene_range = np.array([scene_aabb.min, scene_aabb.max])

        # enable fixtures in usd
        if self._is_updated_usd(enable_fixtures, movable_fixtures):
            dir_name = os.path.dirname(self.usd_path)
            base_name = os.path.basename(self.usd_path)
            new_path = os.path.join(dir_name, base_name.replace(".usd", "_enabled.usd"))
            self.stage.GetRootLayer().Export(new_path)
            self.usd_path = new_path

    def _is_updated_usd(self, enable_fixtures, movable_fixtures):
        is_updated_usd = False
        if enable_fixtures is not None:
            is_updated_usd = True
            for fixture in enable_fixtures:
                usd.activate_prim(self.stage, fixture)
        if movable_fixtures is not None:
            is_updated_usd = True
            root_prim = self.stage.GetPseudoRoot()
            for fixture in movable_fixtures:
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
        for (name, fxtr) in self.scene_cfg.fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs

    def load_floorplan(self, layout_id, style_id, floorplan_version, exclude_layouts, exclude_styles, scene_type, local_scene_path):
        start_time = time.time()
        print(f"load floorplan usd", end="...")

        if local_scene_path is not None:
            self.usd_path = local_scene_path
            self.scene_type = "usd"
            self.layout_id = None
            self.style_id = None
            self.version_id = None
        else:
            if layout_id is None:
                res = floorplan_loader.acquire_usd(backend="robocasa", scene=scene_type, version=floorplan_version, exclude_layout_ids=exclude_layouts, exclude_style_ids=exclude_styles)
            elif style_id is None:
                res = floorplan_loader.acquire_usd(backend="robocasa", scene=scene_type, layout_id=layout_id, version=floorplan_version, exclude_layout_ids=exclude_layouts, exclude_style_ids=exclude_styles)
            else:
                res = floorplan_loader.acquire_usd(backend="robocasa", scene=scene_type, layout_id=layout_id, style_id=style_id, version=floorplan_version, exclude_layout_ids=exclude_layouts, exclude_style_ids=exclude_styles)
            usd_path, self.floorplan_meta = res.result()
            self.usd_path = str(usd_path)
            self.scene_type = self.floorplan_meta.get("scene")
            self.layout_id = self.floorplan_meta.get("layout_id")
            self.style_id = self.floorplan_meta.get("style_id")
            self.version_id = self.floorplan_meta.get("version_id")
        print(f"[backend->robocasa] | [scene->{self.scene_type}] | [layout->{self.layout_id}] | [style->{self.style_id}] loaded in {time.time() - start_time:.2f}s\n")
        print(f"floorplan version_id: {self.version_id}")
