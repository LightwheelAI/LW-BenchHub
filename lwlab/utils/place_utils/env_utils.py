import lwlab.utils.math_utils.transform_utils as T

import os
import yaml

import numpy as np
import contextlib

from copy import deepcopy
import lwlab.utils.object_utils as OU
from lwlab.utils.place_utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from lwlab.core.scenes.loader import object_loader


_ROBOT_POS_OFFSETS: dict[str, list[float]] = {
    "G1ArmsOnly": [0, 0, 0.97],
    "G1": [0, 0, 0.97],
    "H1ArmsOnly": [0, 0, 1.05],
    "H1": [0, 0, 1.05],
    "GR1ArmsOnly": [0, 0, 0.97],
    "GR1FloatingBody": [0, 0, 0.97],
    "GR1": [0, 0, 0.97],
    "GR1FixedLowerBody": [0, 0, 0.97],
    "G1FloatingBody": [0, -0.33, 0],
    "G1": [0, -0.33, 0],
    "G1FixedLowerBody": [0, -0.33, 0],
    "GoogleRobot": [0, 0, 0],
    "DummyRobot": [0, -0.3, 0],
}

VALID_PROPERTY_KEYS = set(
    {
        "graspable",
        "washable",
        "microwavable",
        "cookable",
        "freezable",
        "fridgable",
        "dishwashable",
        "washable",
    }
)

VALID_CFG_KEYS = set(
    {
        "type",
        "name",
        "model",
        "obj_groups",
        "exclude_obj_groups",
        "max_size",
        "object_scale",
        "placement",
        "info",
        "init_robot_here",
        "reset_region",
        "rotate_upright",
        *VALID_PROPERTY_KEYS,
    }
)

VALID_PLACEMENT_KEYS = set(
    {
        "size",
        "pos",
        "offset",
        "margin",
        "rotation_axis",
        "rotation",
        "ensure_object_boundary_in_range",
        "ensure_valid_placement",
        "sample_args",
        "sample_region_kwargs",
        "ref_obj",
        "fixture",
        "try_to_place_in",
    }
)


def determine_face_dir(fixture_rot, ref_rot, epsilon=1e-2):
    delta = ref_rot - fixture_rot
    delta = ((delta + np.pi) % (2 * np.pi)) - np.pi

    # compare to the four cardinal angles
    if abs(delta - 0.0) < epsilon:
        return -1
    if abs(delta - (np.pi / 2)) < epsilon:
        return -2
    if abs(abs(delta) - np.pi) < epsilon:
        return 1
    if abs(delta + (np.pi / 2)) < epsilon:
        return 2


def get_current_layout_stool_rotations(env):
    """
    Automatically detect the current layout and extract unique stool rotation values (z_rot)
    from a YAML layout file associated with the environment.

    Args:
        env: The environment object (must have layout_id attribute)

    Returns:
        list: List of unique rotation values (floats) found in stool configurations
    """
    from lwlab.utils.usd_utils import OpenUsd

    root_prim = env.lwlab_arena.stage.GetPseudoRoot()
    stool_prims = OpenUsd.get_prim_by_prefix(root_prim, "stool", only_xform=True)

    unique_rots = set()
    for stool_prim in stool_prims:
        stool_rot = stool_prim.GetAttribute("xformOp:rotateXYZ").Get()
        if stool_rot is not None:
            unique_rots.add(stool_rot[2] / 180 * np.pi)
    return sorted(list(unique_rots))


def categorize_stool_rotations(stool_rotations, ground_fixture_rot=None):
    """
    Categorize stool rotations into 4 cardinal directions, relative to the ground fixture rotation,
    using vector rotation and cosine similarity.

    Args:
        stool_rotations (list): List of rotation values in radians
        ground_fixture_rot (float): Rotation of the ground fixture in radians

    Returns:
        list: List of categorized rotation directions: [1, 2, -1, -2]
    """
    # Define canonical direction vectors
    category_vectors = {
        1: np.array([0, -1]),
        2: np.array([1, 0]),
        -1: np.array([0, 1]),
        -2: np.array([-1, 0]),
    }

    categorized_rotations = []

    for rotation in stool_rotations:
        # Adjust rotation relative to ground fixture
        if ground_fixture_rot is not None:
            rel_yaw = rotation - ground_fixture_rot
        else:
            rel_yaw = rotation

        # Create rotation matrix
        cos_yaw = np.cos(rel_yaw)
        sin_yaw = np.sin(rel_yaw)
        rot_matrix = np.array(
            [
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw],
            ]
        )

        # Rotate the base direction vector [0, 1]
        rotated_vec = rot_matrix @ np.array([0, 1])

        # Compare to each cardinal direction using dot product
        best_category = None
        best_similarity = -float("inf")  # cos(angle) ranges from -1 to 1

        for category, canonical_vec in category_vectors.items():
            similarity = np.dot(rotated_vec, canonical_vec)
            if similarity > best_similarity:
                best_similarity = similarity
                best_category = category

        categorized_rotations.append(best_category)

    return categorized_rotations


def get_island_group_counter_names(env):
    """
    Automatically detect all counter fixture names under all island_group groups in the current layout.
    Used to get bounding box combo of multiple counters.
    Args:
        env: The environment object (must have layout_id attribute)
    Returns:
        list: List of counter fixture names (str) under all island_group groups
    """
    from lwlab.utils.usd_utils import OpenUsd

    root_prim = env.lwlab_arena.stage.GetPseudoRoot()
    island_prims = OpenUsd.get_prim_by_suffix(root_prim, "island_group", only_xform=True)
    counter_names = []
    for island_prim in island_prims:
        if island_prim.GetAttribute("type").Get() == "Counter":
            counter_names.append(island_prim.GetName())
    return counter_names


def get_combined_counters_2d_bbox_corners(env, counter_names):
    """
    Used to get bounding box combo of multiple counters - useful for dining counters with multiple defined counters.
    """
    all_pts = []
    for name in counter_names:
        fx = env.get_fixture(name)
        all_pts.extend(fx.get_ext_sites(all_points=False, relative=False))
    all_pts = np.asarray(all_pts)

    abs_sites = all_pts[:4].copy()

    min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
    min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()

    xs = abs_sites[:, 0]
    ys = abs_sites[:, 1]

    i_min_x = np.argmin(xs)
    i_max_x = np.argmax(xs)
    i_min_y = np.argmin(ys)
    i_max_y = np.argmax(ys)

    abs_sites[i_min_x, 0] = min_x
    abs_sites[i_max_x, 0] = max_x
    abs_sites[i_min_y, 1] = min_y
    abs_sites[i_max_y, 1] = max_y

    return abs_sites


def compute_robot_base_placement_pose(env, ref_fixture, ref_object=None, offset=None):
    """
    steps:
    1. find the nearest counter to this fixture
    2. compute offset relative to this counter
    3. transform offset to global coordinates

    Args:
        ref_fixture (Fixture): reference fixture to place th robot near

        offset (list): offset to add to the base position

    """
    from lwlab.core.models.fixtures import (
        Counter,
        Stove,
        Stovetop,
        HousingCabinet,
        Fridge,
        fixture_is_type,
        FixtureType,
    )

    # step 1: find ground fixture closest to robot
    ground_fixture = None

    # manipulate drawer envs are exempt from dining counter/stool placement rules
    manipulate_drawer_env = any(
        cls.__name__ == "ManipulateDrawer" for cls in env.__class__.__mro__
    )

    if not fixture_is_type(ref_fixture, FixtureType.DINING_COUNTER):
        # get all base fixtures in the environment
        ground_fixtures = [
            fxtr
            for fxtr in env.fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, Fridge)
        ]

        for fxtr in ground_fixtures:
            # get bounds of fixture
            point = ref_fixture.pos
            # if fxtr.name == "counter_corner_1_main_group_1":
            #     print("point:", point)
            #     p0, px, py, pz = fxtr.get_ext_sites(relative=False)
            #     print("p0:", p0)
            #     print("px:", px)
            #     print("py:", py)
            #     print("pz:", pz)
            #     print()

            if not OU.point_in_fixture(point=point, fixture=fxtr, only_2d=True):
                continue
            ground_fixture = fxtr
            break

    # set the stool fixture as the ref fixture itself if cannot find fixture containing ref
    if ground_fixture is None:
        if fixture_is_type(ref_fixture, FixtureType.STOOL):
            stool_only = True
        else:
            stool_only = False
        ground_fixture = ref_fixture
    else:
        stool_only = False
    # assert base_fixture is not None

    # step 2: compute offset relative to this counter
    ground_to_ref, _ = OU.get_rel_transform(ground_fixture, ref_fixture)

    # find the reference fixture to dining counter if it exists
    ref_to_fixture = None
    if (
        fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER)
        and not manipulate_drawer_env
    ):
        if hasattr(env, "object_cfgs") and env.object_cfgs is not None:
            for cfg in env.object_cfgs:
                placement = cfg.get("placement", None)
                if placement is None:
                    continue
                fixture_id = placement.get("fixture", None)
                if fixture_id is None:
                    continue
                fixture = env.get_fixture(
                    id=fixture_id,
                    ref=placement.get("ref", None),
                    full_name_check=True if cfg["type"] == "fixture" else False,
                )
                if fixture_is_type(fixture, FixtureType.DINING_COUNTER):
                    sample_region_kwargs = placement.get("sample_region_kwargs", {})
                    ref_to_fixture = sample_region_kwargs.get("ref", None)
                    if ref_to_fixture is None:
                        continue
                    # in case ref_to_fixture is a string, get the corresponding fixture object
                    ref_to_fixture = env.get_fixture(ref_to_fixture)

    face_dir = 1  # 1 is facing front of fixture, -1 is facing south end of fixture
    if fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER) or stool_only:
        stool_rotations = get_current_layout_stool_rotations(env)

        # for dining counters, can face either north of south end of fixture
        if ref_object is not None:
            # choose the end that is closest to the ref object
            ref_point = env.object_placements[ref_object][0]
            categorized_stool_rotations = categorize_stool_rotations(
                stool_rotations, ground_fixture.rot
            )
        else:
            ### find the side closest to the ref fixture ###
            ref_point = ref_fixture.pos
            categorized_stool_rotations = None

        if (
            ref_to_fixture is not None
            and fixture_is_type(ref_to_fixture, FixtureType.STOOL)
            and ref_object is None
        ):
            face_dir = determine_face_dir(ground_fixture.rot, ref_to_fixture.rot)
        elif fixture_is_type(ref_fixture, FixtureType.STOOL) and ref_object is None:
            face_dir = determine_face_dir(ground_fixture.rot, ref_fixture.rot)
        else:
            island_group_counter_names = get_island_group_counter_names(env)
            if len(island_group_counter_names) > 1:
                abs_sites = get_combined_counters_2d_bbox_corners(
                    env, island_group_counter_names
                )
            else:
                abs_sites = ground_fixture.get_ext_sites(relative=False)
            abs_sites = np.vstack(abs_sites)
            rel_yaw = ground_fixture.rot + np.pi / 2

            cos_yaw = np.cos(rel_yaw)
            sin_yaw = np.sin(rel_yaw)
            rot_matrix = np.array(
                [
                    [cos_yaw, -sin_yaw],
                    [sin_yaw, cos_yaw],
                ]
            )

            # Rotate each point in abs_sites
            rotated_abs_sites = np.array([rot_matrix @ site[:2] for site in abs_sites])

            # Rotate ref_point (assuming it's a 3D point tuple or np.array)
            if isinstance(ref_point, tuple):
                ref_xy = np.array([ref_point[0], ref_point[1]])
            else:
                ref_xy = np.array(ref_point[:2])  # in case it's a full np.array

            rotated_ref_point = rot_matrix @ ref_xy

            dist1 = abs(rotated_ref_point[0] - rotated_abs_sites[0][0])
            dist2 = abs(rotated_ref_point[0] - rotated_abs_sites[2][0])
            dist3 = abs(rotated_ref_point[1] - rotated_abs_sites[1][1])
            dist4 = abs(rotated_ref_point[1] - rotated_abs_sites[0][1])

            if (
                fixture_is_type(ground_fixture, FixtureType.ISLAND)
                and not manipulate_drawer_env
            ):
                min_dist = min(dist1, dist2, dist3, dist4)
                if min_dist == dist1:
                    face_dir = 1
                elif min_dist == dist2:
                    face_dir = -1
                elif min_dist == dist3:
                    face_dir = 2
                else:
                    face_dir = -2
            else:
                if dist1 < dist2:
                    face_dir = 1
                else:
                    face_dir = -1

                # these dining counters only have 1 accesssible side for robot to spawn
                one_accessible_layout_ids = [11, 27, 30, 35, 49, 60]
                if env.layout_id in one_accessible_layout_ids:
                    stool_rotations = get_current_layout_stool_rotations(env)
                    categorized_stool_rotations = categorize_stool_rotations(
                        stool_rotations, ground_fixture.rot
                    )
                    face_dir = categorized_stool_rotations[0]

    fixture_ext_sites = ground_fixture.get_ext_sites(relative=True)
    fixture_to_robot_offset = np.zeros(3)

    # set x offset
    fixture_to_robot_offset[0] = ground_to_ref[0]

    # y direction it's facing from perspective of host fixture
    if face_dir == 1:  # north
        fixture_p = fixture_ext_sites[0]
        fixture_to_robot_offset[1] = fixture_p[1] - 0.20
    elif face_dir == -1:  # south
        fixture_p = fixture_ext_sites[2]
        fixture_to_robot_offset[1] = fixture_p[1] + 0.20
    elif face_dir == 2:  # west
        fixture_p = fixture_ext_sites[1]
        fixture_to_robot_offset[0] = fixture_p[0] + 0.20
    elif face_dir == -2:  # east
        fixture_p = fixture_ext_sites[0]
        fixture_to_robot_offset[0] = fixture_p[0] - 0.20

    if offset is not None:
        fixture_to_robot_offset[0] += offset[0]
        fixture_to_robot_offset[1] += offset[1]
    elif ref_object is not None:
        sampler = env.placement_initializer.samplers[f"{ref_object}_Sampler"]
        if face_dir == -1 or face_dir == 1:
            fixture_to_robot_offset[0] += np.mean(sampler.x_range)
        if face_dir == 2 or face_dir == -2:
            fixture_to_robot_offset[1] += np.mean(sampler.y_range)

    if (
        isinstance(ground_fixture, HousingCabinet)
        or isinstance(ground_fixture, Fridge)
        or "stack" in ground_fixture.name
    ):
        fixture_to_robot_offset[1] += face_dir * -0.10

    # move back a bit for the stools
    if fixture_is_type(ground_fixture, FixtureType.DINING_COUNTER):
        abs_sites = ground_fixture.get_ext_sites(relative=False)
        stool = ref_to_fixture or env.get_fixture(FixtureType.STOOL)

        stool_rotations = get_current_layout_stool_rotations(env)

        def rotation_matrix_z(theta):
            """
            Return the 3x3 rotation matrix that rotates a vector about the Z axis by theta radians.
            """
            c = np.cos(theta)
            s = np.sin(theta)
            return np.array(
                [
                    [c, -s, 0.0],
                    [s, c, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        if stool is not None:
            abs_sites = ground_fixture.get_ext_sites(relative=False)
            ref_sites = stool.get_ext_sites(relative=False)

            # Apply rotation to both sets of points
            fixture_Rz = rotation_matrix_z(stool.rot + np.pi)
            stool_Rz = rotation_matrix_z(stool.rot + np.pi)
            fixture_sites = [fixture_Rz @ p for p in abs_sites]
            stool_sites = [stool_Rz @ p for p in ref_sites]

            if ref_object is not None:
                # Determine if we should take min or max y based on stool orientation
                normalized_rot = (
                    (stool.rot + np.pi) % (2 * np.pi)
                ) - np.pi  # normalize
                angle = normalized_rot % (2 * np.pi)

                if np.isclose(angle, np.pi / 2, atol=0.2) or np.isclose(
                    angle, 3 * np.pi / 2, atol=0.2
                ):
                    stool_back_site = max(stool_sites, key=lambda p: p[1])
                else:
                    stool_back_site = min(stool_sites, key=lambda p: p[1])

                stool_y = stool_back_site[1]

                # Find fixture site closest in Y to stool back
                fixture_y_diffs = [abs(p[1] - stool_y) for p in fixture_sites]
                closest_fixture_site = fixture_sites[np.argmin(fixture_y_diffs)]
                fixture_y = closest_fixture_site[1]

                delta_y = abs(stool_y - fixture_y)

                if face_dir == 1 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[1] -= delta_y
                elif face_dir == -1 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[1] += delta_y
                elif face_dir == 2 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[0] += delta_y
                elif face_dir == -2 and face_dir in categorized_stool_rotations:
                    fixture_to_robot_offset[0] -= delta_y
            elif fixture_is_type(ref_to_fixture, FixtureType.STOOL):
                if face_dir == 1:
                    fixture_to_robot_offset[1] -= abs(
                        fixture_sites[0][1] - stool_sites[0][1]
                    )
                elif face_dir == -1:
                    fixture_to_robot_offset[1] += abs(
                        fixture_sites[2][1] - stool_sites[2][1]
                    )
                elif face_dir == 2:
                    fixture_to_robot_offset[0] += abs(
                        fixture_sites[1][1] - stool_sites[2][1]
                    )
                elif face_dir == -2:
                    fixture_to_robot_offset[0] -= abs(
                        fixture_sites[0][1] - stool_sites[2][1]
                    )

    # apply robot-specific offset relative to the base fixture for x,y dims
    robot_model = env.robots[0].robot_model
    robot_class_name = robot_model.__class__.__name__
    if robot_class_name in _ROBOT_POS_OFFSETS:
        for dimension in range(0, 2):
            if dimension == 1:
                fixture_to_robot_offset[dimension] += (
                    _ROBOT_POS_OFFSETS[robot_class_name][dimension] * face_dir
                )
            else:
                fixture_to_robot_offset[dimension] += _ROBOT_POS_OFFSETS[
                    robot_class_name
                ][dimension]

    # step 3: transform offset to global coordinates
    robot_base_pos = np.zeros(3)
    robot_base_pos[0:2] = OU.get_pos_after_rel_offset(
        ground_fixture, fixture_to_robot_offset
    )[0:2]

    # apply robot-specific absolutely for z dim
    if robot_class_name in _ROBOT_POS_OFFSETS:
        robot_base_pos[2] = _ROBOT_POS_OFFSETS[robot_class_name][2]
    robot_base_ori = np.array([0, 0, ground_fixture.rot + np.pi / 2])
    if face_dir == -1:
        robot_base_ori[2] += np.pi
    elif face_dir == -2:
        robot_base_ori[2] = ground_fixture.rot
    elif face_dir == 2:
        robot_base_ori[2] = ground_fixture.rot + np.pi

    return robot_base_pos, robot_base_ori


def _check_cfg_is_valid(cfg):
    """
    check a object / fixture config for correctness. called by _get_placement_initializer
    """

    for k in cfg:
        assert (
            k in VALID_CFG_KEYS
        ), f"got invaild key \"{k}\" in {cfg['type']} config {cfg['name']}"
    placement = cfg.get("placement", None)
    if placement is None:
        return
    for k in cfg["placement"]:
        assert (
            k in VALID_PLACEMENT_KEYS
        ), f"got invaild key \"{k}\" in placement config for {cfg['name']}"


def _get_placement_initializer(env, cfg_list, z_offset=0.005):
    """
    Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list.

    Args:
        cfg_list (list): list of object configurations
        z_offset (float): offset in z direction

    Returns:
        SequentialCompositeSampler: placement initializer
    """

    from lwlab.core.models.fixtures import FixtureType, fixture_is_type

    placement_initializer = SequentialCompositeSampler(name="SceneSampler", rng=env.rng)

    for (obj_i, cfg) in enumerate(cfg_list):
        _check_cfg_is_valid(cfg)

        if cfg["type"] == "fixture":
            mj_obj = env.fixtures[cfg["name"]]
        elif cfg["type"] == "object":
            mj_obj = env.objects[cfg["name"]]
        else:
            raise ValueError

        placement = cfg.get("placement", None)
        if placement is None:
            continue

        fixture_id = placement.get("fixture", None)
        reference_object = None
        rotation = placement.get("rotation", np.array([-np.pi / 2, np.pi / 2]))

        if hasattr(mj_obj, "mirror_placement") and mj_obj.mirror_placement:
            rotation = [-rotation[1], -rotation[0]]

        ensure_object_boundary_in_range = placement.get(
            "ensure_object_boundary_in_range", True
        )
        ensure_valid_placement = placement.get("ensure_valid_placement", True)
        rotation_axis = placement.get("rotation_axis", "z")
        sampler_kwargs = dict(
            name="{}_Sampler".format(cfg["name"]),
            mujoco_objects=mj_obj,
            rng=env.rng,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            rotation_axis=rotation_axis,
            rotation=rotation,
        )

        x_ranges = []
        y_ranges = []

        if fixture_id is None:
            target_size = placement.get("size", None)
            x_ranges.append(np.array([-target_size[0] / 2, target_size[0] / 2]))
            y_ranges.append(np.array([-target_size[1] / 2, target_size[1] / 2]))
            ref_pos = [0, 0, 0]
            ref_rot = 0.0
        else:
            fixture = env.get_fixture(
                id=fixture_id,
                ref=placement.get("ref", None),
                full_name_check=True if cfg["type"] == "fixture" else False,
            )
            sample_region_kwargs = placement.get("sample_region_kwargs", {})
            ref_fixture = sample_region_kwargs.get("ref", None)

            # this checks if the reference fixture and dining counter are facing different directions
            ref_dining_counter_mismatch = False
            if fixture_is_type(fixture, FixtureType.DINING_COUNTER) and fixture_is_type(
                ref_fixture, FixtureType.STOOL
            ):
                if abs(abs(ref_fixture.rot) - abs(fixture.rot)) > 0.01:
                    ref_dining_counter_mismatch = True

            ref_obj_name = placement.get("ref_obj", None)

            if ref_obj_name is not None and cfg["name"] != ref_obj_name:
                ref_obj_cfg = find_object_cfg_by_name(env, ref_obj_name)
                reset_region = ref_obj_cfg["reset_region"]
            else:
                if (
                    ensure_object_boundary_in_range
                    and ensure_valid_placement
                    and rotation_axis == "z"
                ):
                    sample_region_kwargs["min_size"] = mj_obj.size

                reset_regions = fixture.sample_reset_region(
                    env=env, **sample_region_kwargs
                )

                reference_object = fixture.name

            cfg["reset_regions"] = reset_regions
            for reset_region in cfg["reset_regions"]:
                outer_size = reset_region["size"]
                if fixture_is_type(fixture, FixtureType.TOASTER) or fixture_is_type(
                    fixture, FixtureType.BLENDER
                ):
                    default_margin = 0.0
                else:
                    default_margin = 0.04
                margin = placement.get("margin", default_margin)
                outer_size = (outer_size[0] - margin, outer_size[1] - margin)
                assert outer_size[0] > 0 and outer_size[1] > 0

                target_size = placement.get("size", None)
                offset = placement.get("offset", (0.0, 0.0))
                inner_xpos, inner_ypos = placement.get("pos", (None, None))

                if ref_dining_counter_mismatch:
                    rel_yaw = fixture.rot - ref_fixture.rot

                    target_size = T.rotate_2d_point(target_size, rot=rel_yaw)
                    target_size = (abs(target_size[0]), abs(target_size[1]))

                    # rotate the pos tuple
                    # treat "ref" as sentinel 5 (or -5) so it survives rotation
                    placeholder = 5.0
                    raw_pos = placement.get("pos", (None, None))

                    numeric_pos = []
                    for v in raw_pos:
                        if v == "ref":
                            numeric_pos.append(placeholder)
                        else:
                            numeric_pos.append(float(v))
                    rotated = T.rotate_2d_point(np.array(numeric_pos), rot=rel_yaw)

                    def unpack(v):
                        diff = abs(abs(v) - placeholder)
                        if abs(abs(v) - placeholder) < 1e-2:
                            return "ref"
                        return float(np.clip(v, -1.0, 1.0))

                    inner_xpos, inner_ypos = unpack(rotated[0]), unpack(rotated[1])

                stool_orientation = False

                # make sure the offset is relative to the reference fixture
                if fixture_is_type(fixture, FixtureType.DINING_COUNTER) and fixture_is_type(
                    ref_fixture, FixtureType.STOOL
                ):
                    rel_yaw = np.pi - (fixture.rot - ref_fixture.rot)
                    offset = T.rotate_2d_point(offset, rot=rel_yaw)
                    epsilon = 1e-2
                    off0 = 0.0 if abs(offset[0]) < epsilon else offset[0]
                    off1 = 0.0 if abs(offset[1]) < epsilon else offset[1]
                    offset = (float(off0), float(off1))

                    stool_orientation = True
                    inner_xpos_og = inner_xpos
                    inner_ypos_og = inner_ypos

                if target_size is not None:
                    target_size = deepcopy(list(target_size))
                    for size_dim in [0, 1]:
                        if target_size[size_dim] == "obj":
                            target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                        if target_size[size_dim] == "obj.x":
                            target_size[size_dim] = mj_obj.size[0] + 0.005
                        if target_size[size_dim] == "obj.y":
                            target_size[size_dim] = mj_obj.size[1] + 0.005
                    inner_size = np.min((outer_size, target_size), axis=0)
                else:
                    inner_size = outer_size

                # center inner region within outer region
                if inner_xpos == "ref":
                    # compute optimal placement of inner region to match up with the reference fixture
                    x_halfsize = outer_size[0] / 2 - inner_size[0] / 2
                    if x_halfsize == 0.0:
                        inner_xpos = 0.0
                    else:
                        ref_fixture = env.get_fixture(sample_region_kwargs["ref"])
                        ref_pos = ref_fixture.pos
                        fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                        outer_to_ref = fixture_to_ref - reset_region["offset"]
                        inner_xpos = outer_to_ref[0] / x_halfsize
                        inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
                elif inner_xpos is None:
                    inner_xpos = 0.0

                if inner_ypos == "ref":
                    # compute optimal placement of inner region to match up with the reference fixture
                    y_halfsize = outer_size[1] / 2 - inner_size[1] / 2
                    if y_halfsize == 0.0:
                        inner_ypos = 0.0
                    else:
                        ref_fixture = env.get_fixture(sample_region_kwargs["ref"])
                        ref_pos = ref_fixture.pos
                        fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                        outer_to_ref = fixture_to_ref - reset_region["offset"]
                        inner_ypos = outer_to_ref[1] / y_halfsize
                        inner_ypos = np.clip(inner_ypos, a_min=-1.0, a_max=1.0)
                elif inner_ypos is None:
                    inner_ypos = 0.0

                # make sure that the orientation is around stool reference
                if stool_orientation and not ref_dining_counter_mismatch:
                    # only skip if both coordinates are "ref"
                    if not (inner_xpos_og == "ref" and inner_ypos_og == "ref"):
                        rel_yaw = np.pi - (fixture.rot - ref_fixture.rot)
                        vec = np.array(
                            [
                                0.0 if inner_xpos_og == "ref" else inner_xpos_og,
                                0.0 if inner_ypos_og == "ref" else inner_ypos_og,
                            ]
                        )

                        cos_yaw = np.cos(rel_yaw)
                        sin_yaw = np.sin(rel_yaw)
                        rot_matrix = np.array(
                            [
                                [cos_yaw, -sin_yaw],
                                [sin_yaw, cos_yaw],
                            ]
                        )
                        rotated = rot_matrix @ vec

                        # Update only the non-"ref" values
                        if inner_xpos_og != "ref":
                            inner_xpos = float(np.clip(rotated[0], -1.0, 1.0))
                        if inner_ypos_og != "ref":
                            inner_ypos = float(np.clip(rotated[1], -1.0, 1.0))

                # offset for inner region
                intra_offset = (
                    (outer_size[0] / 2 - inner_size[0] / 2) * inner_xpos + offset[0],
                    (outer_size[1] / 2 - inner_size[1] / 2) * inner_ypos + offset[1],
                )

                # center surface point of entire region
                ref_pos = fixture.pos + [0, 0, reset_region["offset"][2]]
                ref_rot = fixture.rot

                # x, y, and rotational ranges for randomization
                x_ranges.append(
                    np.array([-inner_size[0] / 2, inner_size[0] / 2])
                    + reset_region["offset"][0]
                    + intra_offset[0]
                )
                y_ranges.append(
                    np.array([-inner_size[1] / 2, inner_size[1] / 2])
                    + reset_region["offset"][1]
                    + intra_offset[1]
                )

        placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                reference_object=reference_object,
                reference_pos=ref_pos,
                reference_rot=ref_rot,
                z_offset=z_offset,
                x_ranges=x_ranges,
                y_ranges=y_ranges,
                **sampler_kwargs,
            ),
            sample_args=placement.get("sample_args", None),
        )

    return placement_initializer


def init_robot_base_pose(env):
    """
    helper function to initialize robot base pose
    """
    # set robot position
    if env.init_robot_base_ref is not None:
        ref_fixture = env.get_fixture(env.init_robot_base_ref)
    else:
        fixtures = list(env.fixtures.values())
        valid_ref_fixture_classes = [
            "CoffeeMachine",
            "Toaster",
            "ToasterOven",
            "Stove",
            "Stovetop",
            "SingleCabinet",
            "HingeCabinet",
            "OpenCabinet",
            "Drawer",
            "Microwave",
            "Sink",
            "Hood",
            "Oven",
            "Fridge",
            "Dishwasher",
            "Wall_obj",
            "Floor_obj",
            "Book",
            "Carpet",
            "Cushion",
            "DecorativeVase",
            "SideTable",
            "Sofa",
            "TableLamp",
        ]
        while True:
            ref_fixture = env.rng.choice(fixtures)
            fxtr_class = type(ref_fixture).__name__
            if fxtr_class not in valid_ref_fixture_classes:
                continue
            break

    ref_object = None
    for cfg in env.object_cfgs:
        if cfg.get("init_robot_here", None) is True:
            ref_object = cfg.get("name")
            break

    robot_base_pos, robot_base_ori = compute_robot_base_placement_pose(
        env,
        ref_fixture=ref_fixture,
        ref_object=ref_object,
    )

    return robot_base_pos, robot_base_ori


def find_object_cfg_by_name(env, name):
    """
    Finds and returns the object configuration with the given name.

    Args:
        name (str): name of the object configuration to find

    Returns:
        dict: object configuration with the given name
    """
    for cfg in env.object_cfgs:
        if cfg["name"] == name:
            return cfg
    raise ValueError


def create_obj(env, cfg):
    """
    Helper function for creating objects.
    Called by _create_objects()
    """
    from lwlab.core.models.fixtures import (
        fixture_is_type,
        FixtureType,
    )  # NOTE: Not implemented in Isaaclab yet
    from lwlab.utils.place_utils.kitchen_object_utils import sample_kitchen_object

    object_cfgs = {}

    if "info" in cfg:
        """
        if cfg has "info" key in it, that means it is storing meta data already
        that indicates which object we should be using.
        set the obj_groups to this path to do deterministic playback
        """
        obj_path = cfg["info"]["obj_path"]
        obj_groups = obj_path
        exclude_obj_groups = None
    else:
        obj_groups = cfg.get("obj_groups", "all")
        exclude_obj_groups = cfg.get("exclude_obj_groups", None)

    if not isinstance(obj_groups, list) and isinstance(obj_groups, tuple):
        obj_groups = list(obj_groups)

    object_cfgs["task_name"] = cfg.get("name")
    object_cfgs["obj_groups"] = obj_groups
    object_cfgs["exclude_obj_groups"] = exclude_obj_groups
    object_cfgs["properties"] = {}
    object_properties = object_cfgs["properties"]

    for key in VALID_PROPERTY_KEYS:
        if cfg.get(key, None) is not None:
            object_properties[key] = cfg[key]

    if "placement" in cfg and "fixture" in cfg["placement"]:
        ref_fixture = cfg["placement"]["fixture"]
        if fixture_is_type(ref_fixture, FixtureType.SINK):
            object_properties["washable"] = True
        elif fixture_is_type(ref_fixture, FixtureType.DISHWASHER):
            object_properties["dishwashable"] = True
        elif fixture_is_type(ref_fixture, FixtureType.MICROWAVE):
            object_properties["microwavable"] = True
        elif fixture_is_type(ref_fixture, FixtureType.STOVE):
            if any(
                cat in obj_groups
                for cat in ["pan", "kettle_electric", "pot", "saucepan", "cookware"]
            ):
                object_properties["cookable"] = False
            else:
                object_properties["cookable"] = True
        elif fixture_is_type(ref_fixture, FixtureType.OVEN):
            # hack for cake, it is bakeable but not cookable.
            # we don't have a bakeable category, so using cookable category in place
            # however, cakes are not cookable in general, only bakable.
            # so we are making an exception here.
            if any(
                cat in obj_groups
                for cat in ["oven_tray", "pan", "pot", "saucepan", "cake"]
            ):
                object_properties["cookable"] = False
            else:
                object_properties["cookable"] = True
        elif fixture_is_type(ref_fixture, FixtureType.FRIDGE):
            object_properties["fridgable"] = True

    return sample_kitchen_object(
        object_cfgs,
        source=cfg.get("source", env.sources),
        max_size=cfg.get("max_size", (None, None, None)),
        scale_factor=cfg.get("scale_factor", None),
        rotate_upright=cfg.get("rotate_upright", False),
    )


@contextlib.contextmanager
def no_collision(sim):
    """
    A context manager that temporarily disables all collision interactions in the simulation.
    Args:
        sim (MjSim): The simulation object where collision interactions will be temporarily disabled.
    Yields:
        None: The function yields control back to the caller while collisions remain disabled.
    Upon exiting the context, the original collision settings are restored.
    """
    original_contype = sim.model.geom_contype.copy()
    original_conaffinity = sim.model.geom_conaffinity.copy()
    sim.model.geom_contype[:] = 0
    sim.model.geom_conaffinity[:] = 0
    try:
        yield
    finally:
        sim.model.geom_contype = original_contype
        sim.model.geom_conaffinity = original_conaffinity


def generate_random_robot_pos(anchor_pos, anchor_ori, pos_dev_x, pos_dev_y):
    local_deviation = np.random.uniform(
        low=(-pos_dev_x, -pos_dev_y),
        high=(pos_dev_x, pos_dev_y),
    )
    local_deviation = np.concatenate((local_deviation, [0.0]))
    global_deviation = np.matmul(
        T.euler2mat(anchor_ori + [0, 0, np.pi / 2]), -local_deviation
    )
    return anchor_pos + global_deviation


def set_robot_to_position(env, global_pos):
    local_pos = np.matmul(
        T.matrix_inverse(T.euler2mat(env.init_robot_base_ori_anchor)), global_pos
    )
    undo_pos = np.matmul(
        T.matrix_inverse(T.euler2mat(env.init_robot_base_ori_anchor)),
        [-10.0, -10.0, 0.0],
    )
    with no_collision(env.sim):
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_side")
        ] = (undo_pos[0] + local_pos[0])
        env.sim.data.qpos[
            env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_forward")
        ] = (undo_pos[1] + local_pos[1])

        env.sim.forward()
