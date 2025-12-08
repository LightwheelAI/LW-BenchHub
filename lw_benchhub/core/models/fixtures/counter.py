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

import json

import numpy as np

from lw_benchhub.utils import object_utils as OU
from lw_benchhub.utils.usd_utils import OpenUsd as usd

from .fixture import ProcGenFixture
from .fixture_types import FixtureType

SIDES = ["left", "right", "front", "back"]


class Counter(ProcGenFixture):
    fixture_types = [FixtureType.COUNTER, FixtureType.COUNTER_NON_CORNER, FixtureType.DINING_COUNTER, FixtureType.ISLAND]

    def _is_fixture_type(self, fixture_type: FixtureType) -> bool:
        """
        check if the fixture is of the given type
        this function is called by fixture_is_type in fixture_utils.py
        """
        if fixture_type == FixtureType.ISLAND:
            return "island" in self.name
        elif fixture_type in [FixtureType.COUNTER, FixtureType.COUNTER_NON_CORNER]:
            return "corner" not in self.name
        valid_dining_prefix = self.name.startswith("dining") or self.name.startswith("island")
        is_dining = valid_dining_prefix or sum(self.base_opening) > 0
        if fixture_type == FixtureType.DINING_COUNTER:
            return is_dining
        elif fixture_type == FixtureType.COUNTER_NON_DINING:
            return not is_dining
        return False

    def __init__(self, name="counter", prim=None, *args, **kwargs,):

        size = tuple(float(x) for x in prim.GetAttribute("size").Get().split(","))
        self.size = kwargs.get("size", (0.72, 0.60, 0.60)) if size is None else size
        self.overhang = kwargs.get("overhang", 0)

        # Set regions BEFORE calling super().__init__()
        self._regions = {
            'main': {
                'p0': np.array([-0.0001, -0.0001, -0.0001]),
                'px': np.array([0.0001, -0.0001, -0.0001]),
                'py': np.array([-0.0001, 0.0001, -0.0001]),
                'pz': np.array([-0.0001, -0.0001, 0.0001])
            },
            'int': {
                'p0': np.array([-0.0001, -0.0001, -0.0001]),
                'px': np.array([0.0001, -0.0001, -0.0001]),
                'py': np.array([-0.0001, 0.0001, -0.0001]),
                'pz': np.array([-0.0001, -0.0001, 0.0001])
            }
        }

        # Store custom regions to restore after parent init
        custom_regions = self._regions.copy()

        super().__init__(name, prim, *args, **kwargs)

        # Restore custom regions after parent initialization
        self._regions = custom_regions

        self.base_opening = prim.GetAttribute("base_opening").Get()
        self.has_opening = prim.GetAttribute("has_opening").Get()

        # set sites
        x, y, z = np.array(self.size) / 2

        main_p0 = np.array([-x, -y + self.overhang, -z])
        main_p1 = np.array([x, y, z])
        self.set_regions(
            {
                "main": {
                    "pos": (main_p0 + main_p1) / 2,
                    "halfsize": (main_p1 - main_p0) / 2,
                }
            }
        )
        self._counter_geoms = self._get_counter_geoms(prim)

    def _get_counter_geoms(self, prim):
        """
        searches for geoms corresponding to each of the four components of the counter.
        Currently does not return collision geoms for top because does not account for the chunking!
        """

        geoms = dict()
        for side in SIDES:
            geoms[f"geometry_{side}"] = list()

        if self.has_opening:
            for side in SIDES:
                geoms[f"top_geometry_{side}"] = list()
        else:
            geoms[f"top_geometry"] = list()
        for geom_name in geoms.keys():
            g = usd.get_prim_by_name(prim, geom_name, only_xform=False)
            g = g[0] if g else None
            if g is not None:
                pos = g.GetAttribute("xformOp:translate").Get()
                scale = g.GetAttribute("xformOp:scale").Get()
                g = dict(pos=pos, size=scale)
            geoms[geom_name].append(g)

        return geoms

    def get_reset_regions(
        self, env=None, ref=None, loc="nn", top_size=(0.40, 0.25), ref_rot_flag=False, full_depth_region=False,
    ):

        all_geoms = []
        for (k, v) in self._counter_geoms.items():
            # only reset on top geoms
            if not k.startswith("top"):
                continue
            geom = v[-1]
            pos = geom.get("pos")
            scale = geom.get("size")
            top_pos = pos
            this_top_size = scale * 2
            # make sure region is sufficiently large
            if this_top_size[0] >= top_size[0] and this_top_size[1] >= top_size[1]:
                all_geoms.append(
                    dict(
                        pos=list(top_pos),
                        size=list(scale),
                    )
                )
        is_island_group = hasattr(self, "name") and "island_group" in self.name
        if full_depth_region and is_island_group and len(all_geoms) > 1:
            region_sizes = [np.array(g.get("size")) * 2 for g in all_geoms]
            areas = [sz[0] * sz[1] for sz in region_sizes]
            sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i])
            min_area = areas[sorted_indices[0]]
            next_min_area = areas[sorted_indices[1]] if len(areas) > 1 else min_area
            if min_area < 0.8 * next_min_area:
                all_geoms = [
                    g for i, g in enumerate(all_geoms) if i != sorted_indices[0]
                ]
        reset_regions = {}

        if ref is None:
            geom_i = 0
            for g in all_geoms:
                top_pos = np.array(g.get("pos"))
                top_half_size = np.array(g.get("size"))
                offset = (top_pos[0], top_pos[1], self.size[2] / 2)
                size = (top_half_size[0] * 2, top_half_size[1] * 2)
                reset_regions[f"geom_{geom_i}"] = dict(size=size, offset=offset)
                geom_i += 1

        else:
            ref_fixture = env.get_fixture(ref)

            ### find an appropriate geom to sample ###
            counter_top_geom_info = {}
            for g in all_geoms:
                g_info = {}

                # get global coordinates of geom center and four corners
                g_pos_in_counter_frame = np.array(g.get("pos"))
                g_halfsize = np.array(g.get("size"))

                # use a slightly smaller bounding box
                SC = 0.99
                g_pos = OU.get_pos_after_rel_offset(self, np.array(g.get("pos")))
                g_pos_c1 = OU.get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [-SC * g_halfsize[0], -SC * g_halfsize[1], 0],
                )
                g_pos_c2 = OU.get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [-SC * g_halfsize[0], SC * g_halfsize[1], 0],
                )
                g_pos_c3 = OU.get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [SC * g_halfsize[0], -SC * g_halfsize[1], 0],
                )
                g_pos_c4 = OU.get_pos_after_rel_offset(
                    self,
                    g_pos_in_counter_frame
                    + [SC * g_halfsize[0], SC * g_halfsize[1], 0],
                )

                g_info["pos"] = g_pos
                g_info["pos_c1"] = g_pos_c1
                g_info["pos_c2"] = g_pos_c2
                g_info["pos_c3"] = g_pos_c3
                g_info["pos_c4"] = g_pos_c4
                g_info["halfsize"] = g_halfsize

                ref_fixture_pos = ref_fixture.pos

                x_min = np.min([g_pos_c1[0], g_pos_c2[0], g_pos_c3[0], g_pos_c4[0]])
                x_max = np.max([g_pos_c1[0], g_pos_c2[0], g_pos_c3[0], g_pos_c4[0]])
                y_min = np.min([g_pos_c1[1], g_pos_c2[1], g_pos_c3[1], g_pos_c4[1]])
                y_max = np.max([g_pos_c1[1], g_pos_c2[1], g_pos_c3[1], g_pos_c4[1]])

                fixture_in_2d_bounds = (
                    ref_fixture_pos[0] >= x_min
                    and ref_fixture_pos[0] <= x_max
                    and ref_fixture_pos[1] >= y_min
                    and ref_fixture_pos[1] <= y_max
                )
                g_info["fixture_in_2d_bounds"] = fixture_in_2d_bounds

                if fixture_in_2d_bounds:
                    g_info["dist_to_fixture"] = 0.0
                else:
                    g_info["dist_to_fixture"] = np.min(
                        [
                            OU.project_point_to_segment(
                                ref_fixture_pos, g_pos_c1, g_pos_c2
                            )[1],
                            OU.project_point_to_segment(
                                ref_fixture_pos, g_pos_c2, g_pos_c4
                            )[1],
                            OU.project_point_to_segment(
                                ref_fixture_pos, g_pos_c3, g_pos_c4
                            )[1],
                            OU.project_point_to_segment(
                                ref_fixture_pos, g_pos_c1, g_pos_c3
                            )[1],
                        ]
                    )

                rel_offset = OU.get_fixture_to_point_rel_offset(ref_fixture, g_pos)
                g_info["fixture_to_geom_rel_offset"] = rel_offset
                g = json.dumps(g)
                counter_top_geom_info[g] = g_info

            valid_geoms = []

            geom_containing_fixture = None
            for g, g_info in counter_top_geom_info.items():
                fixture_in_2d_bounds = g_info["fixture_in_2d_bounds"]
                if fixture_in_2d_bounds:
                    geom_containing_fixture = g
                    break

            if loc == "any":
                valid_geoms = all_geoms
            elif loc == "nn":
                min_dist = None
                chosen_top = None
                for g, g_info in counter_top_geom_info.items():
                    g_dist = g_info["dist_to_fixture"]
                    if min_dist is None or g_dist < min_dist:
                        chosen_top = g
                        min_dist = g_dist
                valid_geoms.append(chosen_top)
            elif loc in ["left_right", "right", "left"]:
                if geom_containing_fixture is not None:
                    valid_geoms.append(g)
                else:
                    # add all regions to the left
                    if loc in ["left_right", "left"]:
                        for g, g_info in counter_top_geom_info.items():
                            offset = g_info["fixture_to_geom_rel_offset"]
                            fixture_in_2d_bounds = g_info["fixture_in_2d_bounds"]
                            if fixture_in_2d_bounds or offset[0] < -0.30:
                                valid_geoms.append(g)

                    # add all regions to the right
                    if loc in ["left_right", "right"]:
                        for g, g_info in counter_top_geom_info.items():
                            offset = g_info["fixture_to_geom_rel_offset"]
                            fixture_in_2d_bounds = g_info["fixture_in_2d_bounds"]
                            if fixture_in_2d_bounds or offset[0] > 0.30:
                                valid_geoms.append(g)
            else:
                raise ValueError

            geom_i = 0

            for g in valid_geoms:
                g = json.loads(g)
                top_pos = np.array(g.get("pos"))
                top_half_size = np.array(g.get("size"))
                offset = [top_pos[0], top_pos[1], 0]  # top counter center is at its top!!!
                size = [top_half_size[0] * 2, top_half_size[1] * 2]

                if (
                    loc in ["left", "right", "left_right"]
                    and geom_containing_fixture is not None
                ):
                    # set size and offset appropriately
                    fixture_size = ref_fixture.size

                    g_pos_c1 = counter_top_geom_info[json.dumps(g)]["pos_c1"]
                    g_pos_c3 = counter_top_geom_info[json.dumps(g)]["pos_c3"]

                    point_to_fixture = OU.get_fixture_to_point_rel_offset(
                        ref_fixture, g_pos_c1, rot=self.rot
                    )
                    fixture_x = np.abs(point_to_fixture[0])

                    if loc in ["left", "left_right"]:
                        x1 = top_pos[0] - top_half_size[0]
                        x2 = (
                            top_pos[0]
                            - top_half_size[0]
                            + fixture_x
                            - fixture_size[0] / 2
                        )
                        this_offset = [np.mean((x1, x2)), offset[1], offset[2]]
                        this_size = [x2 - x1, size[1]]
                        if this_size[0] > 0.20:
                            reset_regions[f"geom_{geom_i}"] = dict(
                                size=this_size, offset=this_offset
                            )
                            geom_i += 1

                    if loc in ["right", "left_right"]:
                        x1 = (
                            top_pos[0]
                            - top_half_size[0]
                            + fixture_x
                            + fixture_size[0] / 2
                        )
                        x2 = top_pos[0] + top_half_size[0]
                        this_offset = [np.mean((x1, x2)), offset[1], offset[2]]
                        this_size = [x2 - x1, size[1]]
                        if this_size[0] > 0.20:
                            reset_regions[f"geom_{geom_i}"] = dict(
                                size=this_size, offset=this_offset
                            )
                            geom_i += 1
                else:
                    min_x = top_pos[0] - top_half_size[0]
                    max_x = top_pos[0] + top_half_size[0]

                    ref_pos, _ = OU.get_rel_transform(self, ref_fixture)
                    if ref_rot_flag is False:
                        if min_x <= ref_pos[0] <= max_x:
                            new_size = min(ref_pos[0] - min_x, max_x - ref_pos[0]) * 2
                            # ensure that new size is not too small
                            if new_size >= 0.20:
                                # TODO: recompute so that region is adjcent to ref fixture
                                offset[0] = ref_pos[0]
                                size[0] = new_size

                    reset_regions[f"geom_{geom_i}"] = dict(size=size, offset=offset)
                    geom_i += 1

        return reset_regions

    # to overwrite Fixture class default
    @property
    def width(self):
        return self.size[0]

    @property
    def depth(self):
        return self.size[1]

    @property
    def height(self):
        return self.size[2]
