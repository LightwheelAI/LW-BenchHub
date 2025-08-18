from lwlab.core.scenes.loader import object_loader

SOURCE_MAPPING = {
    "objaverse": "objaverse",
    "aigen_objs": "aigen",
    "lightwheel": "lightwheel",
}

OBJ_CATEGORIES = object_loader.list_registry()


def get_cats_by_type(types, obj_registries=None):
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
        # check if category is in one of valid object registries
        if (
            obj_cat["registryType"] != "objects" or
            "FILE_TYPE_USD" not in obj_cat["fileTypes"] or
            "types" not in obj_cat["property"] or
            any(s not in SOURCE_MAPPING for s in obj_cat["sources"])
        ):
            continue
        if obj_registries is not None:
            if isinstance(obj_registries, str):
                obj_registries = [obj_registries]
            if any([reg in obj_cat["property"]["types"] for reg in obj_registries]) is False:
                continue

        cat_types = obj_cat["property"]["types"]
        if isinstance(cat_types, str):
            cat_types = [cat_types]
        cat_types = set(cat_types)
        # Access the "types" key in the dictionary using the correct syntax
        if len(cat_types.intersection(types)) > 0:
            res.append(obj_cat["name"])

    return res


### define all object categories ###
OBJ_GROUPS = dict(
    all=[obj_cat["name"] for obj_cat in OBJ_CATEGORIES],
)

for obj_cat in OBJ_CATEGORIES:
    OBJ_GROUPS[obj_cat["name"]] = [obj_cat["name"]]

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
