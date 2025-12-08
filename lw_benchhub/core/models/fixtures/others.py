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

import numpy as np

import lw_benchhub.utils.math_utils.transform_utils.numpy_impl as T

from .fixture import ProcGenFixture


class Box(ProcGenFixture):
    def get_bounding_box_half_size(self):
        return np.array([self.size[0], self.size[1], self.size[2]])


class Wall(Box):
    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
        self.wall_side = prim.GetAttribute("wall_side").Get()

    def _get_pos_after_rel_tranformation(self, offset, quat):
        fixture_mat = T.quat2mat(T.convert_quat(quat))
        return self.pos + np.dot(fixture_mat, offset)

    def _get_reordered_bbox_pts(self, pts):
        """
        Reorder the points of the bounding box to be in a specific order
        This is because after a rotation px may not represent an offset in the x direction anymore.
        pz may not represent an offset in the z direction anymore, etc.
        """
        offs = [p - self.pos for p in pts]
        # for each corner, build its “signature” (+1 or -1 on each axis)
        signs = [tuple(int(np.sign(o[i])) for i in range(3)) for o in offs]

        # map signatures → index in the final list
        sig_to_idx = {
            (-1, -1, -1): 0,  # p0
            (1, -1, -1): 1,  # px
            (-1, 1, -1): 2,  # py
            (-1, -1, 1): 3,  # pz
            (1, 1, -1): 4,  # pxy
            (1, -1, 1): 5,  # pxz
            (-1, 1, 1): 6,  # pyz
            (1, 1, 1): 7,  # pxyz
        }

        # now build the reordered list
        out = [None] * len(pts)
        for p, s in zip(pts, signs):
            out[sig_to_idx[s]] = p

        return out

    def get_ext_sites(self, all_points=False, relative=True):
        """
        Get the exterior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            list: 4 or 8 points
        """
        reg_halfsize = self.get_bounding_box_half_size()
        p0 = [-reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
        px = [reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
        py = [-reg_halfsize[0], reg_halfsize[1], -reg_halfsize[2]]
        pz = [-reg_halfsize[0], -reg_halfsize[1], reg_halfsize[2]]
        sites = [
            p0,
            px,
            py,
            pz,
        ]

        p0, px, py, pz = sites
        sites += [
            np.array([p0[0], py[1], pz[2]]),
            np.array([px[0], py[1], pz[2]]),
            np.array([px[0], py[1], p0[2]]),
            np.array([px[0], p0[1], pz[2]]),
        ]

        quat = np.array(self.get_quat())
        if relative is False:
            sites = [
                self._get_pos_after_rel_tranformation(offset, quat) for offset in sites
            ]
            sites = self._get_reordered_bbox_pts(sites)

        if not all_points:
            return sites[:4]
        return sites

    def get_quat(self):
        """
        Returns the quaternion of the object based on the wall side

        Returns:
            list: quaternion
        """
        side_rots = {
            "back": [-0.707, 0.707, 0, 0],
            "front": [0, 0, 0.707, -0.707],
            "left": [0.5, 0.5, -0.5, -0.5],
            "right": [0.5, -0.5, -0.5, 0.5],
            "floor": [0.707, 0, 0, 0.707],
        }
        if self.wall_side not in side_rots:
            raise ValueError()
        return side_rots[self.wall_side]


class Floor(Wall):
    pass
