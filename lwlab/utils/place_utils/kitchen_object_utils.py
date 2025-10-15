import numpy as np
from lightwheel_sdk.loader import object_loader
from lwlab.utils.place_utils.usd_object import USDObject
from lwlab.utils.place_utils.kitchen_objects import SOURCE_MAPPING, OBJ_GROUPS
import lwlab.utils.place_utils.env_utils as EnvUtils
from termcolor import colored
import time

OBJECT_INFO_CACHE = {}


class ObjInfo:
    def __init__(
        self,
        name,
        types,
        category,
        task_name=None,
        source="objaverse",
        size=None,
        rotate_upright=False,
        obj_path=None,
        exclude=[],
        obj_version=None,
    ):
        self.source = source
        self.name = name
        self.types = types
        self.category = category
        self.task_name = task_name
        self.size = size
        self.rotate_upright = rotate_upright
        self.obj_path = obj_path
        self.exclude = exclude
        self.obj_version = obj_version

    def get_info(self):
        return self.__dict__

    def set_attrs(self, attr_dict: dict):
        for key, value in attr_dict.items():
            setattr(self, key, value)


def sample_kitchen_object(
    object_cfgs,
    source=None,
    max_size=(None, None, None),
    object_scale=None,
    rotate_upright=False,
    projects=None,
    version=None,
):
    """
    Sample a kitchen object from the specified groups and within max_size bounds.

    Args:
        object_cfgs (dict): object configuration

        source (str): source to sample from

        max_size (tuple): max size of the object. If the sampled object is not within bounds of max size, function will resample

        object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value

        rotate_upright (bool): whether to rotate the object to be upright

        projects (list[str]): list of projects to sample from. If set will filter the objects by the projects.

        version (str): version of the object to sample from. If set will filter the objects by the version.


    Returns:
        model (USDObject): the sampled object

        obj_info (dict): the info of the sampled object
    """

    valid_object_sampled = False
    while not valid_object_sampled:
        cache_key = object_cfgs.get("task_name")

        if cache_key and cache_key in OBJECT_INFO_CACHE:
            print(f"--- Fast Reset: Found '{cache_key}' in runtime cache. Bypassing loader. ---")
            acquire_start_time = time.time()
            cached_data = OBJECT_INFO_CACHE[cache_key]
            obj_path = cached_data['obj_path']
            obj_name = cached_data['obj_name']
            obj_res = cached_data['obj_res']
            category = cached_data['category']
            acquire_end_time = time.time()
            total_acquire_time = acquire_end_time - acquire_start_time
            print(f"Total Acquire Time: {total_acquire_time:.4f}s")
        else:
            if cache_key:
                print(f"--- First Run: '{cache_key}' not in cache. Using object_loader. ---")

            if version is not None:
                acquire_start_time = time.time()
                obj_path, obj_name, obj_res = object_loader.acquire_by_file_version(version)
                category = find_most_similar_category(obj_name)
                acquire_end_time = time.time()
                total_acquire_time = acquire_end_time - acquire_start_time
                print(f"Total Acquire Time: {total_acquire_time:.4f}s")

            elif isinstance(object_cfgs["obj_groups"], str) and object_cfgs["obj_groups"].endswith(".usd"):
                if "/" in object_cfgs["obj_groups"]:
                    filename = object_cfgs["obj_groups"].split("/")[-1].split(".")[0]
                    category = find_most_similar_category(object_cfgs["obj_groups"].split("/")[-2])
                else:
                    filename = object_cfgs["obj_groups"].split(".")[0]
                    category = find_most_similar_category(filename)

                acquire_start_time = time.time()
                obj_path, obj_name, obj_res = object_loader.acquire_by_registry(
                    "objects",
                    registry_name=[category],
                    file_name=filename,
                    source=list(source) if source is not None else [],
                    projects=list(projects) if projects is not None else [],
                )
                acquire_end_time = time.time()
                total_acquire_time = acquire_end_time - acquire_start_time
                print(f"Total Acquire Time: {total_acquire_time:.4f}s")

            else:
                category = object_cfgs["obj_groups"]
                if isinstance(category, list):
                    registry_name = [item for c in category for item in OBJ_GROUPS[c]]
                elif isinstance(category, str):
                    registry_name = OBJ_GROUPS[category]

                acquire_start_time = time.time()
                obj_path, obj_name, obj_res = object_loader.acquire_by_registry(
                    "objects",
                    registry_name=registry_name,
                    eqs=None if not object_cfgs["properties"] else object_cfgs["properties"],
                    source=list(source) if source is not None else [],
                    projects=list(projects) if projects is not None else [],
                    contains=None,
                    exclude_registry_name=[] if object_cfgs["exclude_obj_groups"] is None else object_cfgs["exclude_obj_groups"],
                )
                acquire_end_time = time.time()
                total_acquire_time = acquire_end_time - acquire_start_time
                print(f"Total Acquire Time: {total_acquire_time:.4f}s")

            if cache_key:
                OBJECT_INFO_CACHE[cache_key] = {
                    'obj_path': obj_path,
                    'obj_name': obj_name,
                    'obj_res': obj_res,
                    'category': category,
                }

        sampled_category = find_most_similar_category(obj_res["assetName"])
        if sampled_category is None:
            sampled_category = category
        obj_info = ObjInfo(
            name=obj_name,
            types=obj_res["property"]["types"] if "types" in obj_res["property"] else [],
            category=sampled_category,
            task_name=object_cfgs.get("task_name"),
            source=SOURCE_MAPPING[obj_res["source"]],
            rotate_upright=rotate_upright,
            obj_path=obj_path,
            obj_version=obj_res.get("fileVersionId", None),
        )

        metadata = {"scale": 1.0, "exclude": []}
        if obj_info.source in obj_res["metadata"]:
            for key, value in metadata.items():
                if key in obj_res["metadata"][obj_info.source]:
                    metadata[key] = obj_res["metadata"][obj_info.source][key]
        obj_info.scale = metadata["scale"]
        obj_info.exclude = metadata["exclude"]

        # TODO: exclude issue
        # obj_source_exclude = []
        # if "exclude" in metadata:
        #     obj_source_exclude = metadata["exclude"]
        # if obj_name in obj_source_exclude:
        #     print(f"Sampled Object {obj_name} is excluded from {obj_info.source}, Try again...")
        #     continue

        obj_scale = np.array([1.0, 1.0, 1.0])
        obj_scale *= obj_info.scale
        if object_scale is not None:
            obj_scale *= object_scale

        model = USDObject(
            name=obj_info.name,
            task_name=object_cfgs["task_name"],
            category=obj_info.category,
            obj_path=obj_path,
            object_scale=obj_scale,
            rotate_upright=rotate_upright,
        )
        obj_info.size = model.size
        obj_info.set_attrs(obj_res["property"])

        valid_object_sampled = True
        for i in range(3):
            if max_size[i] is not None and obj_info.size[i] > max_size[i]:
                valid_object_sampled = False
                break

        groups_containing_sampled_obj = []
        for type, groups in OBJ_GROUPS.items():
            if obj_info.category in groups:
                groups_containing_sampled_obj.append(type)
        obj_info.groups_containing_sampled_obj = groups_containing_sampled_obj
    print(colored(f"Sampled {object_cfgs['task_name']}: {obj_info.name} from {obj_info.source}", "green"))

    return model, obj_info.get_info()


def find_most_similar_category(filename):
    def normalize_name(name):
        return name.replace("_", "").lower()
    filename_norm = normalize_name(filename)
    groups = max(
        [g for g in OBJ_GROUPS if filename_norm.startswith(normalize_name(g))],
        key=len,
        default=None
    )
    if groups is not None:
        return OBJ_GROUPS[groups][0]
    from difflib import get_close_matches
    candidates = list(OBJ_GROUPS.keys())
    matches = get_close_matches(filename_norm, [normalize_name(c) for c in candidates], n=1, cutoff=0.7)
    if matches:
        idx = [normalize_name(c) for c in candidates].index(matches[0])
        return OBJ_GROUPS[candidates[idx]][0]
    else:
        return None


def extract_failed_object_name(error_message):
    import re
    match = re.search(r"Failed to place object '([^']+)'", error_message)
    if match:
        return match.group(1)
    return None


def recreate_object(env, failed_obj_name):
    try:
        obj_cfg = next((cfg for cfg in env.object_cfgs if cfg.get("name") == failed_obj_name), None)
        if not obj_cfg:
            print(f"Could not find config for failed object: {failed_obj_name}")
            return False

        env.objects.pop(failed_obj_name, None)
        obj_cfg.pop("info", None)

        model, info = EnvUtils.create_obj(env, obj_cfg)
        obj_cfg["info"] = info
        env.objects[model.task_name] = model
        for obj_version in env.cache_usd_version["objects_version"]:
            if failed_obj_name in obj_version:
                obj_version.update({failed_obj_name: info.get("obj_version", None)})
                break

        print(f"Successfully replaced object: {failed_obj_name} with {model.name} (from {info.get('source', 'unknown')})")
        return True
    except Exception as e:
        print(f"Failed to replace object {failed_obj_name}: {str(e)}")
        return False
