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

import torch
import numpy as np
from pxr import UsdGeom, Gf
from copy import deepcopy
from collections import defaultdict
from functools import cached_property
from typing import List

from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg

import lwlab.utils.object_utils as OU
from lwlab.utils.usd_utils import OpenUsd as usd
import lwlab.utils.math_utils.transform_utils.numpy_impl as T
from lwlab.utils.errors import SamplingError
from .fixture_types import FixtureType

FIXTURES = {}
FIXTURE_TYPES = defaultdict(list)


class Fixture:
    fixture_types: List[FixtureType] = []

    def __deepcopy__(self, memo):
        return self

    def __init_subclass__(cls) -> None:
        FIXTURES[cls.__name__] = cls

    def _is_fixture_type(self, fixture_type: FixtureType) -> bool:
        """
        check if the fixture is of the given type
        this function is called by fixture_is_type in fixture_utils.py

        Args:
            fixture_type (FixtureType): the type to check

        Returns:
            bool: True if the fixture is of the given type, False otherwise
        """
        return fixture_type in self.fixture_types

    def __init__(
            self,
            name,
            prim,
            num_envs,
            pos=None,
            size=None,
            max_size=None,
            placement=None,
            rng=None,
            device="cpu",
    ):
        self.name = name
        self.prim = prim
        self.num_envs = num_envs
        self.device = device

        # Preserve existing regions if they exist
        if not hasattr(self, '_regions'):
            self._regions = dict()

        geom_prim_list = usd.get_prim_by_type(self.prim, exclude_types=["Xform", "Scope"])
        self._pos = np.array(list(prim.GetAttribute("xformOp:translate").Get()))
        self._scale = prim.GetAttribute("xformOp:scale").Get()
        self._scale = np.array(self._scale) if self._scale is not None else np.array([1, 1, 1])
        euler_angles = prim.GetAttribute("xformOp:rotateXYZ").Get()
        if euler_angles is not None:
            euler_radians = np.radians(np.array(euler_angles))
            self.set_euler(euler_radians)

        reg_geom_prims = []
        for geom_prim in geom_prim_list:
            g_name = geom_prim.GetName()
            if g_name.startswith("reg_") and "bbox" not in g_name:
                reg_geom_prims.append(geom_prim)

        for geom_prim in reg_geom_prims:
            if "main" in geom_prim.GetName():
                continue
            reg_dict = {}
            g_name = geom_prim.GetName()
            reg_pos, reg_quat = usd.get_prim_pos_rot_in_world(geom_prim)
            if reg_pos is None or reg_quat is None:
                print(f"Error getting prim pos, rot, scale for {g_name} / {prim.GetName()}")
                continue
            else:
                reg_quat = np.array(reg_quat)
                reg_pos = np.array(reg_pos)
                # TODO: fixture standard (extent or scale)
                reg_extent = np.array(geom_prim.GetAttribute("extent").Get()[1])
                reg_scale = np.array(geom_prim.GetAttribute("xformOp:scale").Get())
                reg_halfsize = np.where(abs(reg_extent) > abs(reg_scale), reg_scale, reg_extent)
                reg_halfsize = reg_halfsize * self.scale
                reg_rel_pos = (T.quat2mat(T.convert_quat(reg_quat, to="xyzw")).T @ (np.array(reg_pos) - self.pos))

            p0 = reg_rel_pos + [-reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            px = reg_rel_pos + [reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            py = reg_rel_pos + [-reg_halfsize[0], reg_halfsize[1], -reg_halfsize[2]]
            pz = reg_rel_pos + [-reg_halfsize[0], -reg_halfsize[1], reg_halfsize[2]]
            # reg_dict["prim"] = geom_prim
            reg_dict["p0"] = p0
            reg_dict["px"] = px
            reg_dict["py"] = py
            reg_dict["pz"] = pz
            reg_dict["per_env_offset"] = np.zeros((num_envs, 3))
            self._regions[g_name.replace("reg_", "")] = reg_dict

        # add outer bounding box region(reg_main)
        reg_pos, reg_quat = usd.get_prim_pos_rot_in_world(prim)
        if reg_pos is None or reg_quat is None:
            print(f"Error getting prim pos, rot, scale for main / {prim.GetName()}")
        else:
            reg_pos = np.array(reg_pos)
            reg_quat = np.array(reg_quat)
            reg_halfsize = np.fromstring(prim.GetAttribute("size").Get(), sep=',') / 2
            reg_rel_pos = T.quat2mat(T.convert_quat(reg_quat, to="xyzw")).T @ (np.array(reg_pos) - self.pos)
            reg_dict = {}
            reg_dict["p0"] = reg_rel_pos + [-reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            reg_dict["px"] = reg_rel_pos + [reg_halfsize[0], -reg_halfsize[1], -reg_halfsize[2]]
            reg_dict["py"] = reg_rel_pos + [-reg_halfsize[0], reg_halfsize[1], -reg_halfsize[2]]
            reg_dict["pz"] = reg_rel_pos + [-reg_halfsize[0], -reg_halfsize[1], reg_halfsize[2]]
            reg_dict["per_env_offset"] = np.zeros((num_envs, 3))
            self._regions["main"] = reg_dict

        # if size is not None:
        #     self.set_scale_from_size(size, max_size=max_size)

        self.size = np.array([self.width, self.depth, self.height])

        if self.width is not None:
            try:
                # calculate based on bounding points
                reg_key = None
                if "main" in self._regions:
                    reg_key = "main"
                elif "bbox" in self._regions:
                    reg_key = "bbox"
                else:
                    raise ValueError
                p0 = self._regions[reg_key]["p0"]
                px = self._regions[reg_key]["px"]
                py = self._regions[reg_key]["py"]
                pz = self._regions[reg_key]["pz"]
                self.origin_offset = np.array(
                    [
                        np.mean((p0[0], px[0])),
                        np.mean((p0[1], py[1])),
                        np.mean((p0[2], pz[2])),
                    ]
                ) - np.array(self.prim.GetAttribute("xformOp:translate").Get())
            except Exception as e:
                raise RuntimeError(f"The counter self._regions is None.")
        else:
            self.origin_offset = np.array([0, 0, 0])

        # placement config, for determining where to place fixture (most fixture will not use this)
        self._placement = placement

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

        # track information about all joints
        self._joint_infos = dict()
        joint_prims = usd.get_all_joints_without_fixed(prim)
        for jnt in joint_prims:
            jnt_range = {}
            if jnt.GetAttribute("physics:lowerLimit").Get() is not None:
                if jnt.GetTypeName() == "PhysicsRevoluteJoint":
                    jnt_range = {"range": torch.tensor([jnt.GetAttribute("physics:lowerLimit").Get() * torch.pi / 180,
                                                        jnt.GetAttribute("physics:upperLimit").Get() * torch.pi / 180])}
                else:
                    jnt_range = {"range": torch.tensor([jnt.GetAttribute("physics:lowerLimit").Get(),
                                                        jnt.GetAttribute("physics:upperLimit").Get()])}
            self._joint_infos[jnt.GetName()] = jnt_range

    def get_reset_region_names(self):
        return ("int", )

    def set_euler(self, euler):
        self._euler = euler
        self._rot = euler

    # def set_origin(self, origin):
    #     """
    #     Set the origin of the fixture to a specified position

    #     Args:
    #         origin (3-tuple): new (x, y, z) position of the fixture
    #     """
    #     # compute new position
    #     fixture_rot = np.array([0, 0, self.rot])
    #     fixture_mat = T.euler2mat(fixture_rot)

    #     pos = origin + np.dot(fixture_mat, -self.origin_offset)
    #     self.set_pos(pos)

    # def set_scale_from_size(self, size, max_size=None):
    #     """
    #     Set the scale of the fixture based on the desired size. If any of the dimensions are None,
    #     the scaling factor will be the same as one of the other two dimensions

    #     Args:
    #         size (3-tuple): (width, depth, height) of the fixture
    #         max_size (3-tuple): maximum allowable size (width, depth, height) of the fixture
    #     """
    #     # check that the argument is valid
    #     assert len(size) == 3

    #     # calculate and set scale according to specification
    #     scale = [None, None, None]
    #     cur_size = [self.width, self.depth, self.height]

    #     for (i, t) in enumerate(size):
    #         if t is not None:
    #             scale[i] = t / cur_size[i]

    #     scale[0] = scale[0] or scale[2] or scale[1]
    #     scale[1] = scale[1] or scale[0] or scale[2]
    #     scale[2] = scale[2] or scale[0] or scale[1]
    #     scale = np.array(scale)

    #     if max_size is not None:
    #         # recompute the scaling as needed
    #         scaling_adjustment = 1.0
    #         for i in range(3):
    #             if max_size[i] is None:
    #                 continue
    #             scaling_adjustment = min(
    #                 scaling_adjustment, max_size[i] / (cur_size[i] * scale[i])
    #             )
    #         scale *= scaling_adjustment

    #     self.set_scale(scale)

    #     for (reg_name, reg_dict) in self._regions.items():
    #         for (k, v) in reg_dict.items():
    #             if isinstance(v, np.ndarray):
    #                 reg_dict[k] = v * scale

    # def set_pos(self, pos):
    #     xformable = UsdGeom.Xformable(self._prim)
    #     xformable.SetTranslate(Gf.Vec3f(pos[0], pos[1], pos[2]))

    # def set_scale(self, scale, prim=None):
    #     if prim is None:
    #         prim = self._prim
    #     xformable = UsdGeom.Xformable(prim)
    #     xformable.SetScale(Gf.Vec3f(scale[0], scale[1], scale[2]))

    def setup_cfg(self, cfg: ManagerBasedRLEnvCfg):
        if not usd.has_contact_reporter(self.prim) or \
           (not usd.is_articulation_root(self.prim) and
                not usd.is_rigidbody(self.prim)):
            return

        if usd.is_rigidbody(self.prim) and not usd.is_articulation_root(self.prim):
            prim_path = f"{{ENV_REGEX_NS}}/Scene/{self.name}"
            fixture_contact_sensor = ContactSensorCfg(
                prim_path=prim_path,
                update_period=0.0,
                history_length=1,
                debug_vis=False,
                filter_prim_paths_expr=[],
            )
            setattr(cfg.scene, f"{self.name}_contact", fixture_contact_sensor)
            return
        else:
            self.fixture_name = usd.get_child_commonprefix_name(self.prim)
            corpus = usd.get_prim_by_name(self.prim, "corpus")
            if not corpus:
                print(f"corpus in {self.name} not found")
                if self.fixture_name:
                    prim_path = f"{{ENV_REGEX_NS}}/Scene/{self.name}/{self.fixture_name}"
                    fixture_contact_sensor = ContactSensorCfg(
                        prim_path=prim_path,
                        update_period=0.0,
                        history_length=1,
                        debug_vis=False,
                        filter_prim_paths_expr=[],
                    )
                    setattr(cfg.scene, f"{self.name}_contact", fixture_contact_sensor)
                else:
                    print("error: not regular asset")

    def setup_env(self, env: ManagerBasedRLEnv):
        if self.name in env.scene.extras.keys():
            self.prim_paths = env.scene.extras[self.name].prim_paths
        elif self.name in env.scene.articulations.keys():
            self.prim_paths = env.scene.articulations[self.name]._root_physx_view.prim_paths
        else:
            self.prim_paths = None

    def get_joint_state(self, env, joint_names, env_ids=None):
        """
        Args:
            env (ManagerBasedRLEnv): environment

        Returns:
            dict: maps door names to a percentage of how open they are
        """
        joint_state = dict()

        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_qpos = env.scene.articulations[self.name].data.joint_pos[:, joint_idx]
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]
            # convert to normalized joint value
            norm_qpos = OU.normalize_joint_value(
                joint_qpos,
                joint_min=joint_min,
                joint_max=joint_max,
            )
            if joint_min < 0:
                norm_qpos = 1 - norm_qpos
            joint_state[j_name] = norm_qpos

        return joint_state

    def set_joint_state(self, min, max, env, joint_names, env_ids=None):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is
        Args:
            min (float): minimum percentage of how open the door is
            max (float): maximum percentage of how open the door is
            env (ManagerBasedRLEnv): environment
        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        for j_name in joint_names:
            joint_idx = env.scene.articulations[self.name].data.joint_names.index(j_name)
            joint_range = env.scene.articulations[self.name].data.joint_pos_limits[0, joint_idx, :]
            joint_min, joint_max = joint_range[0], joint_range[1]
            if joint_min >= 0:
                desired_min = joint_min + (joint_max - joint_min) * min
                desired_max = joint_min + (joint_max - joint_min) * max
            else:
                desired_min = joint_min + (joint_max - joint_min) * (1 - max)
                desired_max = joint_min + (joint_max - joint_min) * (1 - min)
            env.scene.articulations[self.name].write_joint_position_to_sim(
                torch.tensor([[self.rng.uniform(float(desired_min), float(desired_max))]]).to(env.device),
                torch.tensor([joint_idx]).to(env.device),
                torch.as_tensor(env_ids).to(env.device) if env_ids is not None else None
            )

    def is_open(self, env, joint_names=None, th=0.90):
        if joint_names is None:
            joint_names = self.door_joint_names
        joint_state = self.get_joint_state(env, joint_names)
        is_open = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for j_name in joint_names:
            assert j_name in joint_state
            norm_qpos = joint_state[j_name]
            is_open = is_open & (norm_qpos >= th)
        return is_open

    def is_closed(self, env, joint_names=None, th=0.005):
        if joint_names is None:
            joint_names = self.door_joint_names
        joint_state = self.get_joint_state(env, joint_names)
        is_closed = torch.tensor([True], device=env.device).repeat(env.num_envs)
        for j_name in joint_names:
            assert j_name in joint_state
            norm_qpos = joint_state[j_name]
            is_closed = is_closed & (norm_qpos <= th)
        return is_closed

    def open_door(self, env, min=0.90, max=1.0, env_ids=None):
        """
        helper function to open the door. calls set_door_state function
        """
        self.set_joint_state(
            env=env, min=min, max=max, joint_names=self.door_joint_names, env_ids=env_ids
        )

    def close_door(self, env, min=0.0, max=0.0, env_ids=None):
        """
        helper function to close the door. calls set_door_state function
        """
        self.set_joint_state(
            env=env, min=min, max=max, joint_names=self.door_joint_names, env_ids=env_ids
        )

    def get_door_state(self, env, joint_names=None, env_ids=None):
        if joint_names is None:
            joint_names = self.door_joint_names
        return self.get_joint_state(env, joint_names, env_ids=env_ids)

    def set_door_state(self, min, max, env, env_ids=None):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is

        Args:
            min (float): minimum percentage of how open the door is

            max (float): maximum percentage of how open the door is

            env (MujocoEnv): environment
        """
        self.set_joint_state(
            env=env, min=min, max=max, joint_names=self.door_joint_names, env_ids=env_ids
        )

    def get_reset_regions(self, env=None, reset_region_names=None, z_range=(0.45, 1.50)):
        """
        Get reset regions from USD file using existing USDObject pattern
        """
        reset_regions = dict()
        if reset_region_names is None:
            reset_region_names = self.get_reset_region_names()
        for reg_name in reset_region_names:
            reg_dict = self._regions.get(reg_name, None)
            if reg_dict is None:
                continue
            p0 = reg_dict["p0"]
            px = reg_dict["px"]
            py = reg_dict["py"]
            pz = reg_dict["pz"]

            if z_range is not None:
                reg_abs_z = self.pos[2] + p0[2]
                if reg_abs_z < z_range[0] or reg_abs_z > z_range[1]:
                    continue

            reset_regions[reg_name] = {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
                "height": (pz[2] - p0[2]),
            }

        return reset_regions

    def update_state(self, env):
        pass

    @cached_property
    def width(self):
        reg_key = None
        if "main" in self._regions:
            reg_key = "main"
        elif "bbox" in self._regions:
            reg_key = "bbox"
        else:
            return None

        reg_p0 = self._regions[reg_key]["p0"]
        reg_px = self._regions[reg_key]["px"]
        w = reg_px[0] - reg_p0[0]
        return w

    @cached_property
    def depth(self):
        reg_key = None
        if "main" in self._regions:
            reg_key = "main"
        elif "bbox" in self._regions:
            reg_key = "bbox"
        else:
            return None

        reg_p0 = self._regions[reg_key]["p0"]
        reg_py = self._regions[reg_key]["py"]
        d = reg_py[1] - reg_p0[1]
        return d

    @cached_property
    def height(self):
        reg_key = None
        if "main" in self._regions:
            reg_key = "main"
        elif "bbox" in self._regions:
            reg_key = "bbox"
        else:
            return None

        reg_p0 = self._regions[reg_key]["p0"]
        reg_pz = self._regions[reg_key]["pz"]
        h = reg_pz[2] - reg_p0[2]
        return h

    @property
    def rot(self):
        return self._rot[2] if hasattr(self, "_rot") else 0

    @property
    def euler(self):
        return self._rot[2] if hasattr(self, "_rot") else 0

    @cached_property
    def door_joint_names(self):
        return [j_name for j_name in self._joint_infos if "door" in j_name]

    @cached_property
    def nat_lang(self):
        return self.name

    @property
    def pos(self):
        return self._pos if hasattr(self, "_pos") else np.array([0, 0, 0])

    @property
    def scale(self):
        return self._scale if hasattr(self, "_scale") else np.array([1, 1, 1])

    @property
    def quat(self):
        return self._quat

    def get_ext_sites(self, all_points=False, relative=True):
        """
        Get the exterior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            list: 4 or 8 points
        """
        reg_key = None
        if "main" in self._regions:
            reg_key = "main"
        elif "bbox" in self._regions:
            reg_key = "bbox"
        else:
            print(f"Fixture {self.name} has no reg_main or reg_bbox")
            return [np.array([2.79, -0.625, 0.62]), np.array([3.51, -0.625, 0.62]), np.array([2.79, -0.025, 0.62]), np.array([2.79, -0.625, 1.22])]

        sites = [
            self._regions[reg_key]["p0"],
            self._regions[reg_key]["px"],
            self._regions[reg_key]["py"],
            self._regions[reg_key]["pz"],
        ]

        if all_points:
            p0, px, py, pz = sites
            sites += [
                np.array([p0[0], py[1], pz[2]]),
                np.array([px[0], py[1], pz[2]]),
                np.array([px[0], py[1], p0[2]]),
                np.array([px[0], p0[1], pz[2]]),
            ]

        if relative is False:
            sites = [OU.get_pos_after_rel_offset(self, offset) for offset in sites]

        return sites

    def get_int_sites(self, all_points=False, relative=True):
        """
        Get the interior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            dict: a dictionary of interior areas, each with 4 or 8 points
        """
        sites_dict = {}
        for prefix in self.get_reset_region_names():
            reg_dict = self._regions.get(prefix, None)
            if reg_dict is None:
                continue

            sites = [
                reg_dict["p0"],
                reg_dict["px"],
                reg_dict["py"],
                reg_dict["pz"],
            ]

            if all_points:
                p0, px, py, pz = sites
                sites += [
                    np.array([p0[0], py[1], pz[2]]),
                    np.array([px[0], py[1], pz[2]]),
                    np.array([px[0], py[1], p0[2]]),
                    np.array([px[0], p0[1], pz[2]]),
                ]

            sites = [s + reg_dict["per_env_offset"] for s in sites]

            if relative is False:
                sites = [OU.get_pos_after_rel_offset(self, offset) for offset in sites]

            sites_dict[prefix] = sites

        return sites_dict

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full set of bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = self.get_ext_sites(all_points=True, relative=True)

        if trans is None:
            trans = self.pos
        if rot is not None:
            rot = T.quat2mat(rot)
        else:
            rot = np.array([0, 0, self.rot])
            rot = T.euler2mat(rot)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points

    def get_all_valid_reset_region(self, min_size=None, *args, **kwargs):
        """
        Sample a reset region from available regions
        """
        from lwlab.utils.fixture_utils import fixture_is_type

        if min_size is not None:
            assert len(min_size) in [2, 3]

        ref_rot = None
        if "ref" in kwargs:
            ref_fixture = kwargs["ref"]
            if hasattr(ref_fixture, "rot"):
                ref_rot = ref_fixture.rot

        # checks if the host fixture is a dining counter, and the reference fixture faces a different direction
        if (
            ref_rot is not None
            and abs(ref_rot - self.rot) > 0.01
            and fixture_is_type(self, FixtureType.DINING_COUNTER)
        ):
            ref_rot_flag = True
        else:
            ref_rot_flag = False

        if fixture_is_type(self, FixtureType.DINING_COUNTER):
            all_regions_dict = self.get_reset_regions(
                *args, **kwargs, ref_rot_flag=ref_rot_flag
            )
        else:
            all_regions_dict = self.get_reset_regions(*args, **kwargs)

        valid_regions = []
        for reg_name, reg_dict in all_regions_dict.items():
            reg_height = reg_dict.get("height", None)
            reg_size = reg_dict["size"]
            if min_size is not None:
                if max(min_size[:2]) > max(reg_size[:2]):
                    # object cannot fit plane
                    continue
                if (
                    reg_height is not None
                    and len(min_size) == 3
                    and min_size[2] > reg_height
                ):
                    # object cannot fit height of region
                    continue
            reg_dict_copy = deepcopy(reg_dict)
            reg_dict_copy["name"] = reg_name
            valid_regions.append(reg_dict_copy)

        if len(valid_regions) < 1:
            raise SamplingError(
                f"Could not find suitable region to sample from {self.name}"
            )
        return valid_regions

    def sample_reset_region(self, min_size=None, *args, **kwargs):
        """
        Sample a reset region from available regions
        """
        valid_regions = self.get_all_valid_reset_region(min_size, *args, **kwargs)
        return self.rng.choice(valid_regions)

    def set_regions(self, region_dict):
        """
        Set the positions of the exterior and interior bounding box sites of the object

        Args:
            region_dict (dict): Dictionary of regions (containing pos, halfsize)
        """
        for (name, reg) in region_dict.items():
            pos = np.array(reg["pos"])
            halfsize = np.array(reg["halfsize"])
            # self._regions[name]["elem"].set("pos", pos)
            # self._regions[name]["elem"].set("size", halfsize)
            self._regions[name] = {}
            self._regions[name]["pos"] = pos
            self._regions[name]["size"] = halfsize

            # compute boundary points for reference
            p0 = pos + np.array([-halfsize[0], -halfsize[1], -halfsize[2]])
            px = pos + np.array([halfsize[0], -halfsize[1], -halfsize[2]])
            py = pos + np.array([-halfsize[0], halfsize[1], -halfsize[2]])
            pz = pos + np.array([-halfsize[0], -halfsize[1], halfsize[2]])
            self._regions[name]["p0"] = p0
            self._regions[name]["px"] = px
            self._regions[name]["py"] = py
            self._regions[name]["pz"] = pz


class ProcGenFixture(Fixture):
    pass
