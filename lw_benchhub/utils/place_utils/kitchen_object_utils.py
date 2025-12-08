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

import glob
import os
import time

import numpy as np
from lightwheel_sdk.loader import object_loader
from termcolor import colored

import lw_benchhub.utils.place_utils.env_utils as EnvUtils
from lw_benchhub.utils.place_utils.kitchen_objects import FIXTURE_GROUPS, OBJ_GROUPS, OBJECT_INFO_CACHE, SOURCE_MAPPING
from lw_benchhub.utils.place_utils.usd_object import USDObject


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
    rgb_replace=None,
    projects=None,
    version=None,
    load_from_local=False,
    ignore_cache=False,
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

        load_from_local (bool): whether to load the object from local.

        ignore_cache (bool): whether to ignore the cache and force to sample a new object


    Returns:
        model (USDObject): the sampled object

        obj_info (dict): the info of the sampled object
    """

    valid_object_sampled = False
    while not valid_object_sampled:
        cache_key = object_cfgs.get("task_name")

        if not ignore_cache and cache_key and cache_key in OBJECT_INFO_CACHE:
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
        elif load_from_local:
            obj_path = object_cfgs["asset_name"]
            obj_name = object_cfgs["asset_name"].split("/")[-1].split(".")[0]
            category = None
            obj_res = {
                "assetName": obj_name,
                "source": "local",
                "metadata": {},
                "property": {},
            }
        else:
            if cache_key:
                print(f"--- First Run: '{cache_key}' not in cache. Using object_loader. ---")

            if version is not None:
                acquire_start_time = time.time()
                obj_path, obj_name, obj_res = object_loader.acquire_by_file_version(version)
                acquire_end_time = time.time()
                total_acquire_time = acquire_end_time - acquire_start_time
                print(f"Total Acquire Time: {total_acquire_time:.4f}s")

            elif object_cfgs["asset_name"]:
                if "/" in object_cfgs["asset_name"]:
                    filename = object_cfgs["asset_name"].split("/")[-1].split(".")[0]
                else:
                    filename = object_cfgs["asset_name"].split(".")[0]

                acquire_start_time = time.time()
                obj_path, obj_name, obj_res = object_loader.acquire_by_registry(
                    object_cfgs["asset_type"],
                    file_name=filename,
                    source=list(source) if source is not None else [],
                    projects=list(projects) if projects is not None else [],
                )
                acquire_end_time = time.time()
                total_acquire_time = acquire_end_time - acquire_start_time
                print(f"Total Acquire Time: {total_acquire_time:.4f}s")

            else:
                acquire_start_time = time.time()
                obj_groups = FIXTURE_GROUPS if object_cfgs["asset_type"] == "fixtures" else OBJ_GROUPS
                categories = [item for c in object_cfgs["obj_groups"] for item in obj_groups[c]]
                obj_path, obj_name, obj_res = object_loader.acquire_by_registry(
                    object_cfgs["asset_type"],
                    registry_name=categories,
                    eqs=None if not object_cfgs["properties"] else object_cfgs["properties"],
                    source=list(source) if source is not None else [],
                    projects=list(projects) if projects is not None else [],
                    contains=None,
                    exclude_registry_name=[] if object_cfgs["exclude_obj_groups"] is None else object_cfgs["exclude_obj_groups"],
                )
                acquire_end_time = time.time()
                total_acquire_time = acquire_end_time - acquire_start_time
                print(f"Total Acquire Time: {total_acquire_time:.4f}s")
            category = obj_res["registryName"]

        obj_info = ObjInfo(
            name=obj_name,
            types=obj_res["property"]["types"] if "types" in obj_res["property"] else [],
            category=category,
            task_name=object_cfgs.get("task_name"),
            source=SOURCE_MAPPING[obj_res["source"]],
            rotate_upright=rotate_upright,
            obj_path=obj_path,
            obj_version=obj_res.get("fileVersionId", None),
        )

        # detect deformable flag from registry properties
        prop_dict = obj_res.get("property", {})
        is_deformable = bool(prop_dict.get("is_deformable", False)) or str(prop_dict.get("body_type", "")).lower() in [
            "deformable",
            "soft_body",
            "soft",
        ]
        # store on obj_info so it appears in returned info dict
        setattr(obj_info, "is_deformable", is_deformable)

        metadata = {"scale": 1.0, "exclude": []}
        if obj_info.source in obj_res["metadata"]:
            for key, value in metadata.items():
                if key in obj_res["metadata"][obj_info.source]:
                    metadata[key] = obj_res["metadata"][obj_info.source][key]
        obj_info.scale = metadata["scale"]
        obj_info.exclude = metadata["exclude"]
        if object_cfgs.get("mjcf_path", None):
            obj_info.mjcf_path = object_cfgs["mjcf_path"]

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
            obj_path=obj_path,
            object_scale=obj_scale,
            rotate_upright=rotate_upright,
            rgb_replace=rgb_replace,
            asset_type=object_cfgs["asset_type"],
            is_deformable=is_deformable,
        )
        obj_info.size = model.size
        obj_info.set_attrs(obj_res["property"])
        obj_info.obj_path = model.obj_path

        valid_object_sampled = True
        for i in range(3):
            if max_size[i] is not None and obj_info.size[i] > max_size[i]:
                valid_object_sampled = False
                OBJECT_INFO_CACHE.clear()
                break

        groups_containing_sampled_obj = []
        for type, groups in OBJ_GROUPS.items():
            if obj_info.category in groups:
                groups_containing_sampled_obj.append(type)
        obj_info.groups_containing_sampled_obj = groups_containing_sampled_obj

    if cache_key:
        OBJECT_INFO_CACHE[cache_key] = {
            'obj_path': obj_path,
            'obj_name': obj_name,
            'obj_res': obj_res,
            'category': category,
        }

        base_obj_dir = os.path.dirname(obj_path)
        pattern = os.path.join(base_obj_dir, f"{obj_name}_*/{obj_name}_*.usd")
        merged_obj_files = glob.glob(pattern)
        if merged_obj_files:
            for merged_obj_path in merged_obj_files:
                merged_obj_name = os.path.basename(os.path.dirname(merged_obj_path))
                OBJECT_INFO_CACHE[f"{cache_key}_{merged_obj_name.lower().split('_')[1]}"] = {
                    'obj_path': merged_obj_path,
                    'obj_name': merged_obj_name,
                    'obj_res': obj_res,
                    'category': category,
                }

    print(colored(f"Sampled {object_cfgs['task_name']}: {obj_info.name} from {obj_info.source}", "green"))

    return model, obj_info.get_info()


def extract_failed_object_name(error_message):
    import re
    patterns = [
        r"Failed to place object\s*'([^']+)'",  # With single quotes
        r'Failed to place object\s*"([^"]+)"',  # With double quotes
        r"Failed to place object\s*([^\s,'\"]+)",  # Without quotes, stops at space or comma or quote
    ]
    for pat in patterns:
        match = re.search(pat, error_message)
        if match:
            return match.group(1)
    return None


def recreate_object(orchestrator, failed_obj_name):
    try:
        obj_cfg = next((cfg for cfg in orchestrator.task.object_cfgs if cfg.get("name") == failed_obj_name), None)
        if not obj_cfg:
            print(f"Could not find config for failed object: {failed_obj_name}")
            return False

        orchestrator.task.objects.pop(failed_obj_name, None)
        obj_cfg.pop("info", None)

        # when recreating object, we should forced update runtime_cache, otherwise will recreate forever
        model, info = EnvUtils.create_obj(orchestrator.task, obj_cfg, ignore_cache=True)
        obj_cfg["info"] = info
        orchestrator.task.objects[model.task_name] = model
        orchestrator.task.assets[info["task_name"]].usd_path = info["obj_path"]
        orchestrator.task.assets[info["task_name"]].object_cfg.spawn.usd_path = info["obj_path"]
        orchestrator.task.contact_sensors[f"{info['task_name']}_contact"].prim_path = f"{{ENV_REGEX_NS}}/Scene/{info['task_name']}/{info['name']}"

        for obj_version in orchestrator.task.objects_version:
            if failed_obj_name in obj_version:
                obj_version.update({failed_obj_name: info.get("obj_version", None)})
                break

        orchestrator.task.placement_initializer = EnvUtils._get_placement_initializer(orchestrator, orchestrator.task.object_cfgs, orchestrator.context.seed)

        print(f"Successfully replaced object: {failed_obj_name} with {model.name} (from {info.get('source', 'unknown')})")
        return True
    except Exception as e:
        print(f"Failed to replace object {failed_obj_name}: {str(e)}")
        return False


def clear_obj_cache():
    OBJECT_INFO_CACHE.clear()
