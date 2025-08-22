import numpy as np
from lwlab.core.scenes.loader import object_loader
from lwlab.utils.place_utils.usd_object import USDObject
from lwlab.utils.place_utils.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS, SOURCE_MAPPING


class ObjInfo:
    def __init__(
        self,
        name,
        types,
        category,
        source="objaverse",
        size=None,
        rotate_upright=False,
        obj_path=None,
        exclude=[],
    ):
        self.source = SOURCE_MAPPING[source]
        self.name = name
        self.types = types
        self.category = category
        self.size = size
        self.rotate_upright = rotate_upright
        self.obj_path = obj_path
        self.exclude = exclude

    def get_info(self):
        return self.__dict__

    def set_attrs(self, attr_dict: dict):
        for key, value in attr_dict.items():
            setattr(self, key, value)


def sample_kitchen_object(
    object_cfgs,
    source=None,
    max_size=(None, None, None),
    scale_factor=None,
    rotate_upright=False,
):
    """
    Sample a kitchen object from the specified groups and within max_size bounds.

    Args:
        object_cfgs (dict): object configuration

        source (str): source to sample from

        max_size (tuple): max size of the object. If the sampled object is not within bounds of max size, function will resample

        scale_factor (float): scale of the object. If set will multiply the scale of the sampled object by this value

        rotate_upright (bool): whether to rotate the object to be upright


    Returns:
        dict: kwargs to apply to the MJCF model for the sampled object

        dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to, the category of the object
              the sampling split the object came from, and the groups the object was sampled from
    """

    valid_object_sampled = False
    while not valid_object_sampled:
        if isinstance(object_cfgs["obj_groups"], str) and object_cfgs["obj_groups"].endswith(".usd"):
            filename = object_cfgs["obj_groups"].split("/")[-1].split(".")[0]
            category = max([g for g in OBJ_GROUPS if filename.startswith(g)], key=len)
            obj_path, obj_name, obj_res = object_loader.acquire_by_registry(
                "objects",
                registry_name=[category],
                file_name=filename,
            )
        else:
            category = object_cfgs["obj_groups"]
            if isinstance(category, list):
                registry_name = [item for c in category for item in OBJ_GROUPS[c]]
            elif isinstance(category, str):
                registry_name = OBJ_GROUPS[category]
            obj_path, obj_name, obj_res = object_loader.acquire_by_registry(
                "objects",
                registry_name=registry_name,
                eqs=None if not object_cfgs["properties"] else object_cfgs["properties"],
                source=list(source) if source is not None else [],
                contains=None,
                exclude_registry_name=[] if object_cfgs["exclude_obj_groups"] is None else object_cfgs["exclude_obj_groups"],
            )

        obj_info = ObjInfo(
            name=obj_name,
            types=obj_res["property"]["types"],
            category=category,
            source=obj_res["source"],
            rotate_upright=rotate_upright,
            obj_path=obj_path,
        )

        metadata = {"scale": 0.9, "exclude": []}
        if obj_info.source in obj_res["metadata"]:
            for key, value in metadata.items():
                if key in obj_res["metadata"][obj_info.source]:
                    metadata[key] = obj_res["metadata"][obj_info.source][key]
        obj_info.scale = metadata["scale"]
        obj_info.exclude = metadata["exclude"]

        obj_source_exclude = []
        if "exclude" in metadata:
            obj_source_exclude = metadata["exclude"]
        if obj_name in obj_source_exclude:
            print(f"Sampled Object {obj_name} is excluded from {obj_info.source}, Try again...")
            continue

        obj_scale = np.array([1.0, 1.0, 1.0])
        if "scale" in metadata:
            obj_scale *= np.array(metadata["scale"])
        if scale_factor is not None:
            obj_scale *= scale_factor

        model = USDObject(
            name=obj_info.name,
            task_name=object_cfgs["task_name"],
            category=obj_info.category,
            obj_path=obj_path,
            scale_factor=obj_scale,
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

    return model, obj_info.get_info()
