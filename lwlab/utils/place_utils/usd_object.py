import numpy as np
from lwlab.utils.usd_utils import OpenUsdWrapper as Usd


class USDObject():
    """
    Blender object with support for changing the scaling
    """

    def __init__(
        self,
        name,
        task_name,
        category,
        obj_path,
        scale_factor=(1.0, 1.0, 1.0),
        rotate_upright=False,
    ):
        # get scale in x, y, z
        if isinstance(scale_factor, float):
            scale_factor = [scale_factor, scale_factor, scale_factor]
        elif isinstance(scale_factor, tuple) or isinstance(scale_factor, list):
            assert len(scale_factor) == 3
            scale_factor = np.array(scale_factor)
        elif isinstance(scale_factor, np.ndarray):
            assert scale_factor.shape[0] == 3
        else:
            raise Exception("got invalid scale_factor: {}".format(scale_factor))
        scale_factor = np.array(scale_factor)
        self.name = name
        self.task_name = task_name
        self.category = category
        self.obj_path = obj_path
        self.scale_factor = scale_factor
        self.rotate_upright = rotate_upright
        self._regions = dict()
        self._setup_region_dict()

    def _setup_region_dict(self):
        reg_dict = dict()
        usd = Usd(self.obj_path)
        usd.scale_size(scale_factor=self.scale_factor)
        reg_bbox = usd.get_prim_by_name("reg_bbox", only_xform=False)[0]
        reg_size = reg_bbox.GetAttribute("extent").Get()
        reg_pos = np.array(reg_bbox.GetAttribute("xformOp:translate").Get())
        if self.rotate_upright:
            usd.rotate_upright()
        usd.export(self.obj_path)

        reg_halfsize = np.array(reg_size[1]) * self.scale_factor
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
        prefix = 'bbox'
        self._regions[prefix] = reg_dict

    @property
    def horizontal_radius(self):
        _horizontal_radius = self._regions["bbox"]["reg_halfsize"][
            0:2
        ]
        return np.linalg.norm(_horizontal_radius)

    @property
    def bottom_offset(self):
        pos = self._regions["bbox"]["reg_pos"]
        half_size = self._regions["bbox"]["reg_halfsize"]
        return np.array([pos[0], pos[1], pos[2] - half_size[2]])

    @property
    def top_offset(self):
        pos = self._regions["bbox"]["reg_pos"]
        half_size = self._regions["bbox"]["reg_halfsize"]
        return np.array([pos[0], pos[1], pos[2] + half_size[2]])

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = []
        center = self._regions["bbox"]["reg_pos"]
        half_size = self._regions["bbox"]["reg_halfsize"]

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
            from lwlab.utils.math_utils.transform_utils import quat2mat
            rot = quat2mat(rot)
        else:
            rot = np.eye(3)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points

    @property
    def size(self):
        half_size = self._regions["bbox"]["reg_halfsize"]
        return list(half_size * 2)
