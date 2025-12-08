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

from lightwheel_sdk.loader import object_loader

OBJECT_INFO_CACHE = {}


def postprocess_categories(obj_categories, selected_type="objects"):
    updated_categories = []
    for obj_cat in obj_categories:
        if obj_cat["removed"]:
            continue
        if obj_cat["registryType"] != selected_type:
            continue
        updated_categories.append(obj_cat)
    return updated_categories


OBJ_CATEGORIES = postprocess_categories(object_loader.list_registry(), selected_type="objects")
FIXTURE_CATEGORIES = postprocess_categories(object_loader.list_registry(), selected_type="fixtures")


def get_cats_by_type(types, obj_registries=[]):
    """
    Retrieves a list of item keys from the global `OBJ_CATEGORIES` dictionary based on the specified types.

    Args:
        types (list): A list of valid types to filter items by. Only items with a matching type will be included.
        obj_registries (list): only consider categories belonging to these object registries

    Returns:
        list: A list of keys from `OBJ_CATEGORIES` where the item's types intersect with the provided `types`.
    """
    types = set(types)

    res = []
    for obj_cat in OBJ_CATEGORIES:
        if (
            obj_cat["registryType"] != "objects"
            or "FILE_TYPE_USD" not in obj_cat["fileTypes"]
            or "types" not in obj_cat["property"]
        ):
            continue

        if obj_registries:
            if isinstance(obj_registries, str):
                obj_registries = [obj_registries]
            if not any(s in obj_registries for s in obj_cat["sources"]):
                continue

        cat_types = obj_cat["property"]["types"]
        if isinstance(cat_types, str):
            cat_types = [cat_types]
        cat_types = set(cat_types)
        if len(types & cat_types) > 0:
            res.append(obj_cat["name"])

    return res


### define all object categories ###
OBJ_GROUPS = dict(
    all=[obj_cat["name"] for obj_cat in OBJ_CATEGORIES],
)

FIXTURE_GROUPS = dict(
    all=[obj_cat["name"] for obj_cat in FIXTURE_CATEGORIES],
)

for obj_cat in OBJ_CATEGORIES:
    OBJ_GROUPS[obj_cat["name"]] = [obj_cat["name"]]

for obj_cat in FIXTURE_CATEGORIES:
    FIXTURE_GROUPS[obj_cat["name"]] = [obj_cat["name"]]

all_types = set()
# populate all_types
for obj_cat in OBJ_CATEGORIES:
    # types are common to both so we only need to examine one
    if "types" not in obj_cat["property"]:
        continue
    cat_types = obj_cat["property"]["types"]
    if isinstance(cat_types, str):
        cat_types = [cat_types]
    all_types = all_types.union(cat_types)

for t in all_types:
    OBJ_GROUPS[t] = get_cats_by_type(types=[t])

OBJ_GROUPS["food"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)
OBJ_GROUPS["in_container"] = get_cats_by_type(
    [
        "vegetable",
        "fruit",
        "sweets",
        "dairy",
        "meat",
        "bread_food",
        "pastry",
        "cooked_food",
    ]
)

# custom groups
OBJ_GROUPS["container"] = ["plate"]  # , "bowl"]
OBJ_GROUPS["kettle"] = ["kettle_non_electric"]
OBJ_GROUPS["cookware"] = ["pan", "pot", "saucepan", "kettle_non_electric"]
OBJ_GROUPS["pots_and_pans"] = ["pan", "pot"]
OBJ_GROUPS["food_set1"] = [
    "apple",
    "baguette",
    "banana",
    "carrot",
    "cheese",
    "cucumber",
    "egg",
    "lemon",
    "orange",
    "potato",
]
OBJ_GROUPS["group1"] = ["apple", "carrot", "banana", "bowl", "can"]
OBJ_GROUPS["container_set2"] = ["plate", "bowl"]
OBJ_GROUPS["oven_ready"] = [
    "corn",
    "fish",
    "steak",
    "tray",
    "potato",
    "sweet_potato",
    "chicken_drumstick",
    "eggplant",
    "broccoli",
]

SOURCE_MAPPING = {
    "objaverse": "objaverse",
    "aigen_objs": "aigen",
    "lightwheel": "lightwheel",
    "local": "local",
}
