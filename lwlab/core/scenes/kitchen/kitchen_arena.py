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

from ..loader import floorplan_loader
from lwlab.utils.usd_utils import OpenUsd as usd
from lwlab.core.models.scenes.scene_parser import parse_fixtures


class KitchenArena:
    """
    Kitchen arena class holding all of the fixtures

    Args:
        layout_id (int or LayoutType): layout of the kitchen to load

        style_id (int or StyleType): style of the kitchen to load

        rng (np.random.Generator): random number generator used for initializing
            fixture state in the KitchenArena

        enable_fixtures (list of str): any fixtures to enable (some are disabled by default)
    """

    def __init__(self, layout_id, style_id, enable_fixtures=None, usd_simplify=False):
        # download floorplan usd
        self._usd_future = floorplan_loader.acquire_usd(layout_id, style_id, cancel_previous_download=True)
        start_time = time.time()
        print(f"load usd", end="...")
        self.usd_path = str(self._usd_future.result())
        del self._usd_future
        print(f"done in {time.time() - start_time:.2f}s")
        stage = usd.get_stage(self.usd_path)

        # enable fixtures in usd
        if enable_fixtures is not None:
            for fixture in self.enable_fixtures:
                usd.activate_prim(stage, fixture)

        # usd simplify
        if usd_simplify:
            new_stage = usd.usd_simplify(stage, self.usd_path, self.fixture_refs)
            dir_name = os.path.dirname(self.usd_path)
            base_name = os.path.basename(self.usd_path)
            new_path = os.path.join(dir_name, base_name.replace(".usda", "_simplified.usda"))
            new_stage.GetRootLayer().Export(new_path)
            self.usd_path = new_path
            self.fixtures = self.fixture_refs
        self.stage = stage
        # load fixtures
        self.fixtures = parse_fixtures(stage)

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
