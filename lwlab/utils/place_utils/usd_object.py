import numpy as np
from lwlab.utils.usd_utils import OpenUsdWrapper as Usd
import lwlab.utils.math_utils.transform_utils.numpy_impl as T


class USDObject():
    """
    Blender object with support for changing the scaling
    """

    def __init__(
        self,
        name: str,
        task_name: str,
        category,
        obj_path,
        object_scale=(1.0, 1.0, 1.0),
        rotate_upright=False,
    ):
        # get scale in x, y, z
        if isinstance(object_scale, float):
            object_scale = [object_scale, object_scale, object_scale]
        elif isinstance(object_scale, tuple) or isinstance(object_scale, list):
            assert len(object_scale) == 3
            object_scale = np.array(object_scale)
        elif isinstance(object_scale, np.ndarray):
            assert object_scale.shape[0] == 3
        else:
            raise Exception("got invalid object_scale: {}".format(object_scale))
        object_scale = np.array(object_scale)
        self.name = name
        self.task_name = task_name
        self.category = category
        self.obj_path = obj_path
        self.object_scale = object_scale
        self.rotate_upright = rotate_upright
        self.init_quat = np.array([0, 0, 0, 1])  # xyzw
        self._regions = dict()
        usd = Usd(self.obj_path)
        self._setup_region_dict(usd)
        usd.scale_size(scale_factor=self.object_scale)
        usd.set_contact_force_threshold(name=self.name, contact_force_threshold=0.0)
        usd.export(self.obj_path)

    def _setup_region_dict(self, usd):
        if self.rotate_upright:
            self.init_quat = np.array([0.5, 0.5, 0.5, 0.5])
        reg_bboxes = usd.get_prim_by_prefix("reg_", only_xform=False)
        for reg_bbox in reg_bboxes:
            reg_dict = dict()
            if reg_bbox.GetTypeName() == "Cylinder" or reg_bbox.GetTypeName() == "Mesh":
                reg_halfsize = np.array(reg_bbox.GetAttribute("extent").Get()[1])
            else:
                reg_halfsize = np.array(reg_bbox.GetAttribute("xformOp:scale").Get())
            reg_pos, _ = usd.get_prim_pos_rot_in_world(reg_bbox)
            reg_offset = reg_bbox.GetAttribute("xformOp:translate").Get()
            if reg_offset is None:
                reg_offset = np.array([0, 0, 0])
            else:
                reg_offset = np.array(reg_offset)
            if reg_pos is None:
                reg_pos = np.array([0, 0, 0])
            else:
                reg_pos = np.array(reg_pos)
            reg_halfsize = reg_halfsize * self.object_scale
            p0 = reg_pos + [-reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            px = reg_pos + [reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            py = reg_pos + [-reg_halfsize[0], reg_halfsize[1], -reg_halfsize[2]]
            pz = reg_pos + [-reg_halfsize[0], -reg_halfsize[1], reg_halfsize[2]]
            reg_dict["p0"] = p0
            reg_dict["px"] = px
            reg_dict["py"] = py
            reg_dict["pz"] = pz
            reg_dict["reg_halfsize"] = reg_halfsize
            reg_dict["reg_pos"] = reg_pos
            reg_dict["reg_offset"] = reg_offset
            prefix = reg_bbox.GetName().replace("reg_", "")
            self._regions[prefix] = reg_dict

    @property
    def bounded_region_name(self):
        if "main" in self._regions:
            return "main"
        elif "bbox" in self._regions:
            return "bbox"
        else:
            raise ValueError(f"No bounded region found for object {self.name}")

    @property
    def bounded_region(self):
        return self._regions[self.bounded_region_name]

    @property
    def horizontal_radius(self):
        _horizontal_radius = self._regions[self.bounded_region_name]["reg_halfsize"][
            0:2
        ]
        return np.linalg.norm(_horizontal_radius)

    @property
    def bottom_offset(self):
        return -1 * self._regions[self.bounded_region_name]["reg_halfsize"]

    @property
    def top_offset(self):
        return self._regions[self.bounded_region_name]["reg_halfsize"]

    def get_bbox_points(self, trans=None, rot=None, name=None):
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        if name is None:
            name = self.bounded_region_name
        bbox_offsets = []
        center = self._regions[name]["reg_pos"]
        half_size = self._regions[name]["reg_halfsize"]

        bbox_offsets = [
            center + half_size * np.array([-1, -1, -1]),  # p0
            center + half_size * np.array([1, -1, -1]),  # px
            center + half_size * np.array([-1, 1, -1]),  # py
            center + half_size * np.array([-1, -1, 1]),  # pz
            center + half_size * np.array([1, 1, 1]),
            center + half_size * np.array([-1, 1, 1]),
            center + half_size * np.array([1, -1, 1]),
            center + half_size * np.array([1, 1, -1]),
        ]

        if trans is None:
            trans = np.array([0, 0, 0])
        if rot is not None:
            rot = T.quat2mat(rot)
        else:
            rot = np.eye(3)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points

    @property
    def size(self):
        half_size = self._regions[self.bounded_region_name]["reg_halfsize"]
        return list(half_size * 2)

    def get_reset_regions(self):
        reset_regions = {}
        for reg_name, reg_dict in self._regions.items():
            reset_regions[reg_name] = {
                "offset": (reg_dict["reg_pos"][0], reg_dict["reg_pos"][1]),
                "size": (reg_dict["reg_halfsize"][0] * 2, reg_dict["reg_halfsize"][1] * 2),
                "height": (reg_dict["reg_halfsize"][2] * 2),
            }
        return reset_regions
