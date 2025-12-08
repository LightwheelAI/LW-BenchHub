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

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.scene.interactive_scene import InteractiveScene
from isaaclab.utils import configclass

from .general_asset import GeneralAsset


@configclass
class GeneralAssetCfg(AssetBaseCfg):

    class_type: type = GeneralAsset

# MonkeyPatch to add the General Asset to the InteractiveScene


def add_general_asset_to_interactive_scene(scene: InteractiveScene, asset_cfg: GeneralAssetCfg):
    if asset_cfg.spawn is not None:
        asset_cfg.spawn.func(
            asset_cfg.prim_path,
            asset_cfg.spawn,
            translation=asset_cfg.init_state.pos,
            orientation=asset_cfg.init_state.rot,
        )
    general_scene = asset_cfg.class_type(asset_cfg, scene.env_regex_ns)

    articulations = general_scene.articulations
    rigid_objects = general_scene.rigid_objects

    # Handle duplicate key names by renaming them as name, name_1, name_2, etc., to be compatible with robocasa and special assets
    name_counters = {}
    for art in articulations:
        base_name = art.cfg.prim_path.split("/")[-1]
        if base_name in scene._articulations:
            name_counters[base_name] = name_counters.get(base_name, 0) + 1
            name = f"{base_name}_{name_counters[base_name]}"
        else:
            name = base_name
        scene._articulations[name] = art

    name_counters = {}
    for rb in rigid_objects:
        base_name = rb.cfg.prim_path.split("/")[-1]
        if base_name in scene._rigid_objects:
            name_counters[base_name] = name_counters.get(base_name, 0) + 1
            name = f"{base_name}_{name_counters[base_name]}"
        else:
            name = base_name
        scene._rigid_objects[name] = rb


orig_add_entities_from_cfg = InteractiveScene._add_entities_from_cfg


def _add_entities_from_cfg_with_general_asset(self: InteractiveScene):
    orig_add_entities_from_cfg(self)
    for asset_name, asset_cfg in self.cfg.__dict__.items():
        if isinstance(asset_cfg, GeneralAssetCfg):
            add_general_asset_to_interactive_scene(self, asset_cfg)


InteractiveScene._add_entities_from_cfg = _add_entities_from_cfg_with_general_asset
