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

import collections
from copy import copy

import numpy as np
from scipy.spatial.transform import Rotation as R

from lw_benchhub.core.models.fixtures import Fixture
from lw_benchhub.utils.errors import SamplingError
from lw_benchhub.utils.math_utils.transform_utils.numpy_impl import (
    convert_quat,
    euler2mat,
    mat2quat,
    quat2axisangle,
    quat2mat,
    quat_multiply,
    rotate_2d_point,
)
from lw_benchhub.utils.object_utils import obj_in_region, objs_intersect
from lw_benchhub.utils.place_utils.usd_object import USDObject


class ObjectPositionSampler:
    """
    Base class of object placement sampler.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        ensure_object_boundary_in_range (bool): If True, will ensure that the object is enclosed within a given boundary
            (should be implemented by subclass)

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        seed,
        mujoco_objects=None,
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_object=None,
        reference_pos=(0, 0, 0),
        reference_rot=0,
        z_offset=0.0,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        # Setup attributes
        self.name = name
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            # Shallow copy the list so we don't modify the inputted list but still keep the object references
            self.mujoco_objects = (
                [mujoco_objects]
                if isinstance(mujoco_objects, USDObject)
                else copy(mujoco_objects)
            )
        self.ensure_object_boundary_in_range = ensure_object_boundary_in_range
        self.ensure_valid_placement = ensure_valid_placement
        self.reference_object = reference_object if reference_object is not None else []
        self.reference_pos = reference_pos
        self.reference_rot = reference_rot
        self.z_offset = z_offset

    def add_objects(self, mujoco_objects):
        """
        Add additional objects to this sampler. Checks to make sure there's no identical objects already stored.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): single model or list of MJCF object models
        """
        mujoco_objects = (
            [mujoco_objects]
            if isinstance(mujoco_objects, USDObject)
            else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), "Object '{}' already in sampler!".format(obj.name)
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        Resets this sampler. Removes all mujoco objects from this sampler.
        """
        self.mujoco_objects = []

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        Uniformly sample on a surface (not necessarily table surface).

        Args:
            fixtures (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object.

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form
        """
        raise NotImplementedError

    @property
    def sides_combinations(self):
        return {
            "left": ["front_left", "back_left"],
            "right": ["front_right", "back_right"],
            "front": ["front_left", "front_right"],
            "back": ["back_left", "back_right"],
            "all": ["front_left", "front_right", "back_left", "back_right"],
        }

    @property
    def valid_sides(self):
        return set(
            [
                "left",
                "right",
                "front",
                "back",
                "all",
                "front_left",
                "front_right",
                "back_left",
                "back_right",
            ]
        )


class UniformRandomSampler(ObjectPositionSampler):
    """
    Places all objects within the table uniformly random.

    Args:
        name (str): Name of this sampler.

        mujoco_objects (None or MujocoObject or list of MujocoObject): single model or list of MJCF object models

        x_range (2-array of float): Specify the (min, max) relative x_range used to uniformly place objects

        y_range (2-array of float): Specify the (min, max) relative y_range used to uniformly place objects

        rotation (None or float or Iterable):
            :`None`: Add uniform random random rotation
            :`Iterable (a,b)`: Uniformly randomize rotation angle between a and b (in radians)
            :`value`: Add fixed angle rotation

        rotation_axis (str): Can be 'x', 'y', or 'z'. Axis about which to apply the requested rotation

        ensure_object_boundary_in_range (bool):
            :`True`: The center of object is at position:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]

        ensure_valid_placement (bool): If True, will check for correct (valid) object placements

        reference_pos (3-array): global (x,y,z) position relative to which sampling will occur

        z_offset (float): Add a small z-offset to placements. This is useful for fixed objects
            that do not move (i.e. no free joint) to place them above the table.
    """

    def __init__(
        self,
        name,
        seed,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_object=None,
        reference_pos=(0, 0, 0),
        reference_rot=0,
        z_offset=0.0,
        side="all",
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        self.rotation = rotation
        self.rotation_axis = rotation_axis

        if side not in self.valid_sides:
            raise ValueError(
                "Invalid value for side, must be one of:", self.valid_sides
            )

        super().__init__(
            name=name,
            seed=seed,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_object=reference_object,
            reference_pos=reference_pos,
            reference_rot=reference_rot,
            z_offset=z_offset,
        )

    def _sample_x(self, obj_size=None):
        """
        Samples the x location for a given object

        Returns:
            float: sampled x position
        """
        minimum, maximum = self.x_range
        if obj_size is not None:
            buffer = min(obj_size[0], obj_size[1]) / 2
            if self.ensure_object_boundary_in_range:
                minimum += buffer
                maximum -= buffer

        if minimum > maximum:
            raise Exception(
                f"Invalid x range for placement initializer: ({minimum}, {maximum})"
            )

        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_y(self, obj_size=None):
        """
        Samples the y location for a given object

        Returns:
            float: sampled y position
        """
        minimum, maximum = self.y_range
        if obj_size is not None:
            buffer = min(obj_size[0], obj_size[1]) / 2
            if self.ensure_object_boundary_in_range:
                minimum += buffer
                maximum -= buffer

        if minimum > maximum:
            raise Exception(
                f"Invalid y range for placement initializer: ({minimum}, {maximum})"
            )

        return self.rng.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        Samples the orientation for a given object

        Returns:
            np.array: sampled object quaternion in (w,x,y,z) form

        Raises:
            ValueError: [Invalid rotation axis]
        """
        if self.rotation is None:
            rot_angle = self.rng.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            if isinstance(self.rotation[0], collections.abc.Iterable):
                rotation = self.rng.choice(self.rotation)
            else:
                rotation = self.rotation
            rot_angle = self.rng.uniform(high=max(rotation), low=min(rotation))
        else:
            rot_angle = self.rotation

        # Return angle based on axis requested
        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # Invalid axis specified, raise error
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, placed_objects=None, reference=None, ref_fixture=None, on_top=True, max_attempts=None):
        """
        Uniformly sample relative to this sampler's reference_pos or @reference (if specified).

        Args:
            placed_objects (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            PlacementError: [Cannot place all objects]
            AssertionError: [Reference object name does not exist, invalid inputs]
        """
        # Standardize inputs
        placed_objects = {} if placed_objects is None else copy(placed_objects)

        if ref_fixture is not None:
            if self.reference_object is None:
                self.reference_object = [ref_fixture.name if isinstance(ref_fixture, Fixture) else ref_fixture]
            else:
                self.reference_object = [self.reference_object, ref_fixture.name if isinstance(ref_fixture, Fixture) else ref_fixture]

        reference_rot = self.reference_rot
        ref_quat = mat2quat(euler2mat([0, 0, self.reference_rot]))
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            # replace ref_quat with reference object's quat
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset[-1] += abs(quat2mat(ref_quat) @ ref_obj.top_offset)[-1]
            reference_rot = (quat2axisangle(ref_quat))[2]
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert (
                obj.task_name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.task_name)

            success = False

            if (
                isinstance(obj, USDObject) or isinstance(obj, Fixture)
            ):
                obj_points = obj.get_bbox_points()
                p0 = obj_points[0]
                px = obj_points[1]
                py = obj_points[2]
                pz = obj_points[3]
                obj_size = (px[0] - p0[0], py[1] - p0[1], pz[2] - p0[2])
            else:
                obj_size = None

            for _ in range(max_attempts):  # 5000 retries
                sample_region_idx = self.rng.choice(len(self.x_ranges))
                self.x_range = self.x_ranges[sample_region_idx]
                self.y_range = self.y_ranges[sample_region_idx]

                ### get boundary points ###
                region_points = np.array(
                    [
                        [self.x_range[0], self.y_range[0], 0],
                        [self.x_range[1], self.y_range[0], 0],
                        [self.x_range[0], self.y_range[1], 0],
                    ]
                )
                for i in range(len(region_points)):
                    region_points[i][0:2] = rotate_2d_point(
                        region_points[i][0:2], rot=reference_rot
                    )
                region_points += base_offset

                # random rotation
                quat = convert_quat(self._sample_quat(), to="xyzw")
                # multiply this quat by the object's initial rotation if it has the attribute specified
                if hasattr(obj, "init_quat"):
                    quat = quat_multiply(quat, obj.init_quat)

                if obj_size is not None and quat is not None:
                    rot = R.from_quat(quat)
                    size_vecs = [
                        [obj_size[0], 0, 0],
                        [0, obj_size[1], 0],
                    ]
                    x_vec = rot.apply(size_vecs[0])
                    y_vec = rot.apply(size_vecs[1])
                    x_proj = abs(x_vec[0]) + abs(y_vec[0])
                    y_proj = abs(x_vec[1]) + abs(y_vec[1])
                    rotated_obj_size = (x_proj, y_proj, obj_size[2] if len(obj_size) > 2 else 0)

                if self.ensure_object_boundary_in_range and \
                   ((self.x_range[1] - self.x_range[0]) < rotated_obj_size[0] or
                        (self.y_range[1] - self.y_range[0]) < rotated_obj_size[1]):
                    continue

                # sample object coordinates
                relative_x = self._sample_x(rotated_obj_size)
                relative_y = self._sample_y(rotated_obj_size)

                # apply rotation
                object_x, object_y = rotate_2d_point(
                    [relative_x, relative_y], rot=reference_rot
                )

                object_x = object_x + base_offset[0]
                object_y = object_y + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z += abs(quat2mat(quat) @ obj.bottom_offset)[-1]

                quat = quat_multiply(ref_quat, quat)

                location_valid = True

                # ensure object placed fully in region
                if self.ensure_object_boundary_in_range and not obj_in_region(
                    obj,
                    obj_pos=[object_x + obj.bounded_region["reg_offset"][0],
                             object_y + obj.bounded_region["reg_offset"][1],
                             object_z],
                    obj_quat=quat,
                    p0=region_points[0],
                    px=region_points[1],
                    py=region_points[2],
                ):
                    location_valid = False
                    continue

                # objects cannot overlap
                if self.ensure_valid_placement:
                    for placed_obj_name, (
                        (x, y, z),
                        other_quat,
                        other_obj,
                    ) in placed_objects.items():
                        if placed_obj_name in self.reference_object:
                            continue
                        if objs_intersect(
                            obj=obj,
                            obj_pos=[object_x + obj.bounded_region["reg_offset"][0],
                                     object_y + obj.bounded_region["reg_offset"][1],
                                     object_z],
                            obj_quat=quat,
                            other_obj=other_obj,
                            other_obj_pos=[x, y, z],
                            other_obj_quat=other_quat,
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # location is valid, put the object down
                    print(f"Placed object '{obj.task_name}' successfully")
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.task_name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                debug_info = f"Failed to place object '{obj.task_name}' after {max_attempts} attempts\n"
                debug_info += f"  Object size: {obj.size}\n"
                debug_info += f"  X range: {self.x_range}\n"
                debug_info += f"  Y range: {self.y_range}\n"
                debug_info += f"  Placed objects count: {len(placed_objects)}\n"
                raise SamplingError(debug_info)

        return placed_objects


class SequentialCompositeSampler(ObjectPositionSampler):
    """
    Samples position for each object sequentially. Allows chaining
    multiple placement initializers together - so that object locations can
    be sampled on top of other objects or relative to other object placements.

    Args:
        name (str): Name of this sampler.
    """

    def __init__(self, name, seed):
        # Samplers / args will be filled in later
        self.samplers = collections.OrderedDict()
        self.tmp_samplers = collections.OrderedDict()
        self.sample_args = collections.OrderedDict()
        self.samplers_with_args = []

        super().__init__(name=name, seed=seed)

    def append_sampler(self, sampler, sample_args=None):
        """
        Adds a new placement initializer with corresponding @sampler and arguments

        Args:
            sampler (ObjectPositionSampler): sampler to add
            sample_args (None or dict): If specified, should be additional arguments to pass to @sampler's sample()
                call. Should map corresponding sampler's arguments to values (excluding @fixtures argument)

        Raises:
            AssertionError: [Object name in samplers]
        """
        # Verify that all added mujoco objects haven't already been added, and add to this sampler's objects dict
        for obj in sampler.mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"{obj.task_name} '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        self.samplers[sampler.name] = sampler
        self.tmp_samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args

        self.update_samplers_with_args()

    def update_samplers_with_args(self):
        """
        Update the samplers with args. Called by append_sampler, hide, unhide.
        """
        # sort samplers by object size (large -> small)
        self.samplers_with_args = list(zip(self.samplers.values(), self.sample_args.values()))
        self.samplers_with_args.sort(key=lambda x: self.get_obj_size(x[0]), reverse=True)
        self.samplers_with_args = self.adjust_order_by_container(self.samplers_with_args)

    def hide(self, sampler_name):
        """
        Helper method to remove an object from the workspace.

        Args:
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to hide
        """
        hide_sampler = UniformRandomSampler(
            name=sampler_name,
            seed=self.seed,
            mujoco_objects=self.samplers[sampler_name].mujoco_objects,
            x_ranges=[[-20, -10]],
            y_ranges=[[-20, -10]],
            rotation=[0, 0],
            rotation_axis="z",
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
        )
        self.samplers[sampler_name] = hide_sampler
        self.update_samplers_with_args()

    def unhide(self, sampler_name):
        """
        Helper method to restore an object to the workspace.
        """
        unhide_sampler = self.tmp_samplers[sampler_name]
        self.samplers[sampler_name] = unhide_sampler
        self.update_samplers_with_args()

    def add_objects(self, mujoco_objects):
        """
        Override super method to make sure user doesn't call this (all objects should implicitly belong to sub-samplers)
        """
        raise AttributeError(
            "add_objects() should not be called for SequentialCompsiteSamplers!"
        )

    def add_objects_to_sampler(self, sampler_name, mujoco_objects):
        """
        Adds specified @mujoco_objects to sub-sampler with specified @sampler_name.

        Args:
            sampler_name (str): Existing sub-sampler name
            mujoco_objects (MujocoObject or list of MujocoObject): Object(s) to add
        """
        # First verify that all mujoco objects haven't already been added, and add to this sampler's objects dict
        mujoco_objects = (
            [mujoco_objects]
            if isinstance(mujoco_objects, USDObject)
            else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        # Make sure sampler_name exists
        assert sampler_name in self.samplers.keys(), (
            "Invalid sub-sampler specified, valid options are: {}, "
            "requested: {}".format(self.samplers.keys(), sampler_name)
        )
        # Add the mujoco objects to the requested sub-sampler
        self.samplers[sampler_name].add_objects(mujoco_objects)

    def reset(self):
        """
        Resets this sampler. In addition to base method, iterates over all sub-samplers and resets them
        """
        super().reset()
        for sampler in self.samplers.values():
            sampler.reset()

    def sample(self, placed_objects=None, reference=None, ref_fixture=None, on_top=True, max_attempts=None):
        """
        Sample from each placement initializer sequentially, in the order
        that they were appended.

        Args:
            placed_objects (dict): dictionary of current object placements in the scene as well as any other relevant
                obstacles that should not be in contact with newly sampled objects. Used to make sure newly
                generated placements are valid. Should be object names mapped to (pos, quat, MujocoObject)

            reference (str or 3-tuple or None): if provided, sample relative placement. This will override each
                sampler's @reference argument if not already specified. Can either be a string, which
                corresponds to an existing object found in @fixtures, or a direct (x,y,z) value. If None, will sample
                relative to this sampler's `'reference_pos'` value.

            on_top (bool): if True, sample placement on top of the reference object. This will override each
                sampler's @on_top argument if not already specified. This corresponds to a sampled
                z-offset of the current sampled object's bottom_offset + the reference object's top_offset
                (if specified)

        Return:
            dict: dictionary of all object placements, mapping object_names to (pos, quat, obj), including the
                placements specified in @fixtures. Note quat is in (w,x,y,z) form

        Raises:
            PlacementError: [Cannot place all objects]
        """
        # Standardize inputs
        placed_objects = {} if placed_objects is None else copy(placed_objects)

        # Iterate through all samplers to sample
        for sampler, s_args in self.samplers_with_args:
            # Pre-process sampler args
            if s_args is None:
                s_args = {}
            for arg_name, arg in zip(("reference", "ref_fixture", "on_top"), (reference, ref_fixture, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg
            # Run sampler
            new_placements = sampler.sample(placed_objects=placed_objects, max_attempts=max_attempts, **s_args)
            # Update placements
            placed_objects.update(new_placements)

        # only return placements for newly placed objects
        sampled_obj_names = [
            obj.task_name
            for sampler in self.samplers.values()
            for obj in sampler.mujoco_objects
        ]
        return {k: v for (k, v) in placed_objects.items() if k in sampled_obj_names}

    def get_obj_size(self, sampler):
        if hasattr(sampler, "mujoco_objects") and len(sampler.mujoco_objects) > 0:
            return np.array([o.size for o in sampler.mujoco_objects]).max()
        return np.inf

    def adjust_order_by_container(self, sample_with_args):
        sample_num = len(sample_with_args)
        for i in range(sample_num):
            for j in range(i + 1, sample_num):
                if sample_with_args[i][1] is None:
                    continue
                if sample_with_args[i][1]["reference"] in (o.task_name for o in sample_with_args[j][0].mujoco_objects):
                    sample_with_args[i], sample_with_args[j] = sample_with_args[j], sample_with_args[i]
        return sample_with_args
