# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from typing import TYPE_CHECKING

from isaac_arena.utils.singleton import SingletonMeta

# NOTE(alexmillane): Avoid circular import.
if TYPE_CHECKING:
    from isaac_arena.assets.asset import Asset  # only imported for type checking


class AssetRegistry(metaclass=SingletonMeta):

    def __init__(self):
        self.assets = {}

    def register(self, asset: type["Asset"]):
        """Register an asset with a name.

        Args:
            name (str): The name of the asset.
            asset (Asset): The asset to register.
        """
        assert asset.name not in self.assets, f"Asset {asset.name} already registered"
        assert asset.name is not None, "Asset name is not set"
        assert asset.tags is not None, "Asset tags are not set"
        self.assets[asset.name] = asset

    def is_registered(self, name: str) -> bool:
        """Check if an asset is registered.

        Args:
            name (str): The name of the asset.
        """
        return name in self.assets

    def get_asset_by_name(self, name: str) -> type["Asset"]:
        """Get an asset by name.

        Args:
            name (str): The name of the asset.

        Returns:
            Asset: The asset.
        """
        return self.assets[name]

    def get_assets_by_tag(self, tag: str) -> list[type["Asset"]]:
        """Gets a list of assets by tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            list[Asset]: The list of assets.
        """
        return [asset for asset in self.assets.values() if tag in asset.tags]

    def get_random_asset_by_tag(self, tag: str) -> type["Asset"]:
        """Gets a random asset which has the given tag.

        Args:
            tag (str): The tag of the assets.

        Returns:
            Asset: The random asset.
        """
        assets = self.get_assets_by_tag(tag)
        if len(assets) == 0:
            raise ValueError(f"No assets found with tag {tag}")
        return random.choice(assets)


def get_environment_configuration_from_registry(
    background_name: str | None = None, object_name: str | None = None, embodiment_name: str | None = None
) -> dict[str, type["Asset"]]:
    from isaac_arena.assets.asset_registry import AssetRegistry

    asset_registry = AssetRegistry()
    if background_name:
        background = asset_registry.get_asset_by_name(background_name)()
    else:
        background = asset_registry.get_random_asset_by_tag("background")()
    if object_name:
        pick_up_object = asset_registry.get_asset_by_name(object_name)()
    else:
        pick_up_object = asset_registry.get_random_asset_by_tag("object")()
    if embodiment_name:
        embodiment = asset_registry.get_asset_by_name(embodiment_name)()
    else:
        embodiment = asset_registry.get_random_asset_by_tag("embodiment")()

    environment_configuration = {
        "background": background,
        "object": pick_up_object,
        "embodiment": embodiment,
    }

    return environment_configuration


# Register all the assets. Down here at the bottom of the file because
# the assets use the AssetRegistry class in order to register themselves,
# so it needs to be fully defined to avoid a circular import.
from isaac_arena.assets.background import *  # noqa: F403, F401
from isaac_arena.assets.objects import *  # noqa: F403, F401
from isaac_arena.embodiments.franka import *  # noqa: F403, F401
from isaac_arena.embodiments.gr1t2 import *  # noqa: F403, F401
