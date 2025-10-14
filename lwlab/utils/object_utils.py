import re
import numpy as np
import torch
import lwlab.utils.math_utils.transform_utils.numpy_impl as T
import os
from isaaclab.utils.math import matrix_from_quat, euler_xyz_from_quat
from lwlab.utils.usd_utils import usd


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.

    Examples:
        [0, 1, 2] => "0 1 2"

    Args:
        array (n-array): Array to convert to a string

    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def obj_inside_of(env, obj_name, fixture_id, partial_check=False):
    """
    whether an object (another mujoco object) is inside of fixture. applies for most fixtures
    """

    obj = env.cfg.objects[obj_name]
    fixture = env.cfg.get_fixture(fixture_id)

    # step 1: calculate fxiture points
    fixtr_int_regions = fixture.get_int_sites(relative=False)
    check = []
    for i in range(env.cfg.scene.num_envs):
        for reset_region in fixtr_int_regions.values():
            inside_of = True
            fixtr_p0, fixtr_px, fixtr_py, fixtr_pz = [r + env.scene.env_origins[i].cpu().numpy() for r in reset_region]
            u = fixtr_px[i] - fixtr_p0[i]
            v = fixtr_py[i] - fixtr_p0[i]
            w = fixtr_pz[i] - fixtr_p0[i]

            # get the position and quaternion of object
            obj_pos = env.scene.rigid_objects[obj_name].data.body_com_pos_w[i, 0, :].cpu().numpy()
            obj_quat = T.convert_quat(
                env.scene.rigid_objects[obj_name].data.body_com_quat_w[i, 0, :].cpu().numpy(), to="xyzw"
            )

            if partial_check:
                obj_points_to_check = [obj_pos]
                th = 0.0
            else:
                # calculate 8 boundary points of object
                obj_points_to_check = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
                # threshold to mitigate false negatives: even if the bounding box point is out of bounds,
                th = 0.05

            for obj_p in obj_points_to_check:
                check1 = (
                    np.dot(u, fixtr_p0[i]) - th <= np.dot(u, obj_p) <= np.dot(u, fixtr_px[i]) + th
                )
                check2 = (
                    np.dot(v, fixtr_p0[i]) - th <= np.dot(v, obj_p) <= np.dot(v, fixtr_py[i]) + th
                )
                check3 = (
                    np.dot(w, fixtr_p0[i]) - th <= np.dot(w, obj_p) <= np.dot(w, fixtr_pz[i]) + th
                )

                if not (check1 and check2 and check3):
                    inside_of = False
                    break

            if inside_of:
                check.append(True)
                break

        if not inside_of:
            check.append(False)

    return torch.tensor(check, dtype=torch.bool, device=env.device)


# used for cabinets, cabinet panels, counters, etc.
def set_geom_dimensions(sizes, positions, geoms, rotated=False):
    """
    set the dimensions of geoms in a fixture

    Args:
        sizes (dict): dictionary of sizes for each side

        positions (dict): dictionary of positions for each side

        geoms (dict): dictionary of geoms for each side

        rotated (bool): whether the fixture is rotated. Fixture may be rotated to make texture appear uniform
                        due to mujoco texture conventions
    """
    if rotated:
        # rotation trick to make texture appear uniform
        # see .xml file
        for side in sizes.keys():
            # not the best detection method, TODO: check euler
            if "door" in side or "trim" in side:
                sizes[side] = [sizes[side][i] for i in [0, 2, 1]]

    # set sizes and positions of all geoms
    for side in positions.keys():
        for geom in geoms[side]:
            if geom is None:
                raise ValueError("Did not find geom:", side)
            geom.set("pos", array_to_string(positions[side]))
            geom.set("size", array_to_string(sizes[side]))


def get_rel_transform(fixture_A, fixture_B):
    """
    Gets fixture_B's position and rotation relative to fixture_A's frame
    """
    A_trans = np.array(fixture_A.pos)
    B_trans = np.array(fixture_B.pos)

    A_rot = np.array([0, 0, fixture_A.rot])
    B_rot = np.array([0, 0, fixture_B.rot])

    A_mat = T.euler2mat(A_rot)
    B_mat = T.euler2mat(B_rot)

    T_WA = np.vstack((np.hstack((A_mat, A_trans[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_trans[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def transform_global_to_local(global_x, global_y, rot):
    """
    Transforms a global movement vector from a global frame (with rotation `rot`)
    into local coordinates (assumed to be rotation 0).
    Args:
        global_x (float): Movement along local x-axis
        global_y (float): Movement along local y-axis
        rot (float): Rotation of the local frame (in radians)
    Returns:
        (float, float): Transformed (local_x, local_y)
    """
    cos_yaw = torch.cos(rot)
    sin_yaw = torch.sin(rot)

    local_x = cos_yaw * global_x - sin_yaw * global_y
    local_y = sin_yaw * global_x + cos_yaw * global_y

    return local_x, local_y


def compute_rel_transform(A_pos, A_mat, B_pos, B_mat):
    """
    Gets B's position and rotation relative to A's frame
    """
    T_WA = np.vstack((np.hstack((A_mat, A_pos[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_pos[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def get_fixture_to_point_rel_offset(fixture, point, rot=None):
    """
    get offset relative to fixture's frame, given a global point
    """
    global_offset = point - fixture.pos
    if rot is None:
        rot = fixture.rot
    T_WF = T.euler2mat([0, 0, rot])
    rel_offset = np.matmul(np.linalg.inv(T_WF), global_offset)
    return rel_offset


def get_pos_after_rel_offset(fixture, offset):
    """
    Get the global position after applying an offset relative to the center of the fixture.
    Supports offset of shape (3,) or (N, 3).
    """
    fixture_rot = np.array([0, 0, fixture.rot])
    fixture_mat = T.euler2mat(fixture_rot)

    offset = np.asarray(offset)
    if offset.ndim == 1:
        # (3,)
        return fixture.pos + np.dot(fixture_mat, offset)
    elif offset.ndim == 2 and offset.shape[1] == 3:
        # (N, 3)
        return fixture.pos + np.dot(offset, fixture_mat.T)
    else:
        raise ValueError("offset must have shape (3,) or (N, 3)")


def project_point_to_line(P, A, B):
    """
    logic copied from here: https://stackoverflow.com/a/61342198
    """
    AP = P - A
    AB = B - A
    result = A + np.dot(AP, AB) / np.dot(AB, AB) * AB

    return result


def point_in_fixture(point, fixture, only_2d=False):
    """
    check if point is inside of the exterior bounding boxes of the fixture

    Args:
        point (np.array): point to check

        fixture (Fixture): fixture object

        only_2d (bool): whether to check only in 2D
    """
    p0, px, py, pz = fixture.get_ext_sites(relative=False)
    th = 0.00
    u = px - p0
    v = py - p0
    w = pz - p0
    check1 = np.dot(u, p0) - th <= np.dot(u, point) <= np.dot(u, px) + th
    check2 = np.dot(v, p0) - th <= np.dot(v, point) <= np.dot(v, py) + th
    check3 = np.dot(w, p0) - th <= np.dot(w, point) <= np.dot(w, pz) + th

    if only_2d:
        return check1 and check2
    else:
        return check1 and check2 and check3


from lwlab.utils.place_utils.usd_object import USDObject


def obj_in_region(
    obj,
    obj_pos,
    obj_quat,
    p0,
    px,
    py,
    pz=None,
):
    """
    check if object is in the region defined by the points.
    Uses either the objects bounding box or the object's horizontal radius
    """
    from lwlab.core.models.fixtures import Fixture

    if isinstance(obj, USDObject) or isinstance(obj, Fixture):
        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
    else:
        radius = obj.horizontal_radius
        obj_points = obj_pos + np.array(
            [
                [radius, 0, 0],
                [-radius, 0, 0],
                [0, radius, 0],
                [0, -radius, 0],
            ]
        )

    u = px - p0
    v = py - p0
    w = pz - p0 if pz is not None else None

    for point in obj_points:
        check1 = np.dot(u, p0) <= np.dot(u, point) <= np.dot(u, px)
        check2 = np.dot(v, p0) <= np.dot(v, point) <= np.dot(v, py)

        if not check1 or not check2:
            return False

        if w is not None:
            check3 = np.dot(w, p0) <= np.dot(w, point) <= np.dot(w, pz)
            if not check3:
                return False

    return True


def fixture_pairwise_dist(f1, f2):
    """
    Gets the distance between two fixtures by finding the minimum distance between their exterior bounding box points
    """
    f1_points = f1.get_ext_sites(all_points=True, relative=False)
    f2_points = f2.get_ext_sites(all_points=True, relative=False)

    all_dists = [np.linalg.norm(p1 - p2) for p1 in f1_points for p2 in f2_points]
    return np.min(all_dists)


def objs_intersect(
    obj,
    obj_pos,
    obj_quat,
    other_obj,
    other_obj_pos,
    other_obj_quat,
):
    """
    check if two objects intersect using Separating Axis Theorem (SAT)
    """
    obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
    other_obj_points = other_obj.get_bbox_points(
        trans=other_obj_pos, rot=other_obj_quat
    )

    face_normals = [
        obj_points[1] - obj_points[0],
        obj_points[2] - obj_points[0],
        obj_points[3] - obj_points[0],
        other_obj_points[1] - other_obj_points[0],
        other_obj_points[2] - other_obj_points[0],
        other_obj_points[3] - other_obj_points[0],
    ]

    intersect = True

    # noramlize length of normals
    for normal in face_normals:
        normal = np.array(normal) / np.linalg.norm(normal)

        obj_projs = [np.dot(p, normal) for p in obj_points]
        other_obj_projs = [np.dot(p, normal) for p in other_obj_points]

        # see if gap detected
        if np.min(other_obj_projs) > np.max(obj_projs) or np.min(
            obj_projs
        ) > np.max(other_obj_projs):
            intersect = False
            break
    return intersect


def normalize_joint_value(raw, joint_min, joint_max):
    """
    normalize raw value to be between 0 and 1
    """
    return (raw - joint_min) / (joint_max - joint_min)


def check_obj_in_receptacle(env, obj_name, receptacle_name, th=None):
    """
    check if object is in receptacle object based on threshold
    """
    obj = env.cfg.objects[obj_name]
    recep = env.cfg.objects[receptacle_name]
    obj_contact_path = env.scene.sensors[f"{obj_name}_contact"].contact_physx_view.sensor_paths
    recep_contact_path = env.scene.sensors[f"{receptacle_name}_contact"].contact_physx_view.sensor_paths
    if env.common_step_counter > 1:
        contact_views = [env.cfg.contact_queues[env_id].pop() for env_id in range(env.num_envs)]
        is_contact = torch.tensor(
            [max(abs(view.get_contact_data(env.physics_dt)[0])) > 0 for view in contact_views],
            device=env.device,
        )  # (env_num, )
    else:
        for env_id in range(env.scene.num_envs):
            env.cfg.contact_queues[env_id].add(
                env.sim.physics_sim_view.create_rigid_contact_view(
                    obj_contact_path[env_id],
                    [recep_contact_path[env_id]],
                    max_contact_data_count=200,
                )
            )
        is_contact = torch.tensor([False], device=env.device).repeat(env.scene.num_envs)  # (env_num, )

    obj_pos = torch.mean(env.scene.rigid_objects[obj_name].data.body_com_pos_w, dim=1)  # (env_num, 3)
    recep_pos = torch.mean(env.scene.rigid_objects[receptacle_name].data.body_com_pos_w, dim=1)  # (env_num, 3)
    if th is None:
        th = recep.horizontal_radius
    is_closed = torch.norm(obj_pos[:, :2] - recep_pos[:, :2], dim=-1) < th  # (env_num, )
    return is_contact & is_closed


def check_obj_in_receptacle_no_contact(env, obj_name, receptacle_name, th=None):
    """
    check if object is in receptacle object based on threshold
    """
    recep = env.cfg.objects[receptacle_name]

    obj_pos = torch.mean(env.scene.rigid_objects[obj_name].data.body_com_pos_w, dim=1)  # (env_num, 3)
    recep_pos = torch.mean(env.scene.rigid_objects[receptacle_name].data.body_com_pos_w, dim=1)  # (env_num, 3)
    if th is None:
        th = recep.horizontal_radius * 0.7
    is_closed = torch.norm(obj_pos[:, :2] - recep_pos[:, :2], dim=-1) < th  # (env_num, )
    return is_closed


def check_obj_fixture_contact(env, obj_name, fixture_name) -> torch.Tensor:
    """
    check if object is in contact with fixture
    """
    obj = env.cfg.objects[obj_name]
    fixture = env.cfg.get_fixture(fixture_name)
    return env.cfg.check_contact(obj, fixture)


def check_obj_any_counter_contact(env, kit_env, obj_name):  # use example: check_obj_any_counter_contact(self.env,self,obj_name)
    """
    check if the object is in contact with any counter fixture in the environment.
    """
    from lwlab.core.models.fixtures import Counter
    for fixture in kit_env.fixtures.values():
        if isinstance(fixture, Counter):
            if check_obj_fixture_contact(env, obj_name, fixture).any():
                return torch.tensor([True], device=env.device, dtype=torch.bool).repeat(env.num_envs)
    return torch.tensor([False], device=env.device, dtype=torch.bool).repeat(env.num_envs)


def check_fixture_in_receptacle(env, fixture_name, fixture_object, receptacle_name, th=None):
    """
    check if fixture is in receptacle object based on threshold
    """
    is_contact = check_obj_fixture_contact(env, receptacle_name, fixture_name)

    recep = env.cfg.objects[receptacle_name]

    fix_pos = torch.tensor(env.scene.articulations[fixture_object].data.root_link_pos_w[..., 0:2])
    recep_pos = torch.mean(env.scene.rigid_objects[receptacle_name].data.body_com_pos_w, dim=1)  # (env_num, 3)
    if th is None:
        th = recep.horizontal_radius * 0.7
    is_closed = torch.norm(fix_pos[:, :2] - recep_pos[:, :2], dim=-1) < th  # (env_num, )
    return is_contact & is_closed


def gripper_obj_far(env, obj_name="obj", th=0.25, eef_name=None) -> torch.Tensor:
    """
    check if gripper is far from object based on distance defined by threshold
    """
    if obj_name in env.cfg.objects:
        obj_pos = env.scene.rigid_objects[obj_name].data.body_com_pos_w[:, 0:1, :]  # (num_envs, 1, 3)
    else:
        if not isinstance(obj_name, str):
            obj_name = obj_name.name
        obj_pos = env.scene.state['articulation'][obj_name]['root_pose'][:, :3].unsqueeze(1)  # (num_envs, 1, 3)

    if eef_name is None:
        gripper_site_pos = env.scene["ee_frame"].data.target_pos_w  # (num_envs, num_eefs, 3)
    else:
        eef_frame_data = env.scene['ee_frame'].data
        eef_index = eef_frame_data.target_frame_names.index(eef_name)
        gripper_site_pos = eef_frame_data.target_pos_w[:, eef_index: eef_index + 1, :]  # (num_envs, 1, 3)
    gripper_obj_far = torch.norm(gripper_site_pos - obj_pos, dim=-1) > th  # (num_envs, num_eefs)
    return torch.all(gripper_obj_far, dim=-1)  # (num_envs, )


def gripper_fixture_far(env, fixture_object, th=0.2) -> torch.Tensor:
    """
    check if gripper is far from fixture based on distance defined by threshold

    Args:
        env (Env): environment
        fixture_object: fixture object
        th (float): threshold for distance

    Returns:
        torch.Tensor: (num_envs,) - True if gripper is far from fixture for each environment
    """
    # Get fixture position for all environments
    fix_pos = torch.tensor(env.scene.articulations[fixture_object].data.root_link_pos_w[..., 0:2])  # (num_envs, 2)

    # Get gripper position for all environments
    gripper_site_pos = env.scene["ee_frame"].data.target_pos_w  # (num_envs, num_eefs, 3)

    # Calculate distance between gripper and fixture for all environments
    gripper_fixture_dist = torch.norm(gripper_site_pos[:, :, :2] - fix_pos.unsqueeze(1), dim=-1)  # (num_envs, num_eefs)

    # Check if gripper is far from fixture for all end effectors
    gripper_fixture_far = gripper_fixture_dist > th  # (num_envs, num_eefs)

    # Return True only if ALL end effectors are far from fixture
    return torch.all(gripper_fixture_far, dim=-1)  # (num_envs,)


def obj_cos(env, obj_name="obj", ref=(0, 0, 1)):
    def cos(u, v):
        return np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-10)

    obj_quat = torch.mean(env.scene.rigid_objects[obj_name].data.body_com_quat_w, dim=1)[0]
    obj_mat = T.quat2mat(obj_quat)

    return cos(obj_mat[:, 2], np.array(ref))


def get_obj_lang(env, obj_name="obj", get_preposition=False):
    """
    gets a formatted language string for the object (replaces underscores with spaces)

    Args:
        obj_name (str): name of object
        get_preposition (bool): if True, also returns preposition for object

    Returns:
        str: language string for object
    """
    obj_cfg = None
    for cfg in env.object_cfgs:
        if cfg["name"] == obj_name:
            obj_cfg = cfg
            break
    lang = obj_cfg["info"]["category"]

    # replace some phrases
    if lang == "kettle electric":
        lang = "electric kettle"
    elif lang == "kettle non electric":
        lang = "kettle"
    elif lang == "bread_flat":
        lang = "bread"

    if not get_preposition:
        return lang

    if lang in ["bowl", "pot", "pan"]:
        preposition = "in"
    elif lang in ["plate"]:
        preposition = "on"
    else:
        raise ValueError("obj category not in bowl / pot / pan / plate")

    return lang, preposition


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (torch.Tensor): (N,4) tensor of quaternions in (w,x,y,z) format

    Returns:
        torch.Tensor: (N,3,3) tensor of rotation matrices
    """
    q = quaternion.to(dtype=torch.float32)

    n = torch.sum(q * q, dim=-1, keepdim=True)
    mask = (n < 1e-7).squeeze(-1)

    q = torch.where(mask.unsqueeze(-1), torch.tensor([1., 0., 0., 0.], device=q.device), q)
    n = torch.where(mask.unsqueeze(-1), torch.ones_like(n), n)

    q *= torch.sqrt(2.0 / n)
    q2 = q.unsqueeze(-1) @ q.unsqueeze(-2)

    mat = torch.stack([
        torch.stack([1.0 - q2[..., 2, 2] - q2[..., 3, 3], q2[..., 1, 2] - q2[..., 3, 0], q2[..., 1, 3] + q2[..., 2, 0]], dim=-1),
        torch.stack([q2[..., 1, 2] + q2[..., 3, 0], 1.0 - q2[..., 1, 1] - q2[..., 3, 3], q2[..., 2, 3] - q2[..., 1, 0]], dim=-1),
        torch.stack([q2[..., 1, 3] - q2[..., 2, 0], q2[..., 2, 3] + q2[..., 1, 0], 1.0 - q2[..., 1, 1] - q2[..., 2, 2]], dim=-1)
    ], dim=-2)

    return torch.where(mask.unsqueeze(-1).unsqueeze(-1),
                       torch.eye(3, dtype=torch.float32, device=q.device).expand(q.shape[0], -1, -1),
                       mat)


def construct_full_env_path(env_regex_ns_template: str, env_index: int, fixture_name: str) -> str:

    return os.path.join(f"{env_regex_ns_template.rstrip('.*')}{env_index}", "Scene", fixture_name)


def check_object_stable(env, obj: str, threshold=0.5):
    '''
    check if the object is stable
    '''
    obj_vel = env.scene.rigid_objects[obj].data.body_com_vel_w[:, 0, :]
    obj_vel_norm = torch.norm(obj_vel, dim=-1)
    return obj_vel_norm < threshold


def project_point_to_segment(point, seg_start, seg_end):
    """
    Projects a point onto a line segment, and clamps it to the segment bounds.

    Args:
        point (np.ndarray): The point to project, shape (2,) or (3,)
        seg_start (np.ndarray): Start point of the segment, same shape as point
        seg_end (np.ndarray): End point of the segment, same shape as point

    Returns:
        np.ndarray: The closest point on the segment to the input point
        float: The distance to the segment

    From ChatGPT
    """
    # Convert to numpy arrays in case they aren't already
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)

    seg_vec = seg_end - seg_start
    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq == 0.0:
        # Segment is a point
        return seg_start, 0.0

    # Compute t: projection factor along the segment
    t = np.dot(point - seg_start, seg_vec) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    # Compute the closest point
    closest_point = seg_start + t_clamped * seg_vec
    dist = np.linalg.norm(point - closest_point)

    return closest_point, dist


def check_place_obj1_on_obj2(env, obj1, obj2, th_z_axis_cos=0.8, th_xy_dist=0.25, th_xyz_vel=0.5, gipper_th=0.25):
    """
    check if obj1 is placed on obj2
    obj1 and obj2 must be a fixture moveable or a object

    Args:
        env (Env): environment
        obj1 : name of object 1 or a moveable fixture
        obj2 : name of object 2 or a moveable fixture
        th_z_axis_cos (float): threshold for z-axis cosine similarity
        th_xy_dist (float): threshold for xy distance
        th_xyz_vel (float): threshold for xyz velocity
        gipper_th (float): threshold for gripper distance

    Returns:
        dict: success state of the task with tensor values for multi-env support
        - gripper_far: check if gripper is far from obj1 and obj2 (tensor)
        - obj1_is_standing: check if obj1 is standing (tensor)
        - obj1_in_obj2: check if obj1 is in obj2 (tensor)
        - obj1_stable: check if obj1 is stable (tensor)
        - obj1_contact_with_obj2: check if obj1 is in contact with obj2 (tensor)

    """
    import torch

    # Get object positions - use torch.mean for multi-body objects like check_obj_in_receptacle
    if obj1 in env.cfg.objects:
        obj1_obj = env.cfg.objects[obj1]
        obj1_pos = torch.mean(env.scene.rigid_objects[obj1].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        obj1_vel = torch.mean(env.scene.rigid_objects[obj1].data.body_com_vel_w, dim=1)  # (num_envs, 3)
        obj1_quat = torch.mean(env.scene.rigid_objects[obj1].data.body_com_quat_w, dim=1)  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
    else:
        obj1_name = obj1 if isinstance(obj1, str) else obj1.name
        obj1_pos = env.scene.state['articulation'][obj1_name]['root_pose'][:, :3]  # (num_envs, 3)
        obj1_quat = env.scene.state['articulation'][obj1_name]['root_pose'][:, 3:]  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
        obj1_vel = env.scene.state['articulation'][obj1_name]['root_velocity'][:, :3]  # (num_envs, 3)

    if obj2 in env.cfg.objects:
        obj2_obj = env.cfg.objects[obj2]
        obj2_pos = torch.mean(env.scene.rigid_objects[obj2].data.body_com_pos_w, dim=1)  # (num_envs, 3)
    else:
        obj2_obj = obj2
        obj2_name = obj2 if isinstance(obj2, str) else obj2.name
        obj2_pos = env.scene.state['articulation'][obj2_name]['root_pose'][:, :3]  # (num_envs, 3)

    # Calculate z-axis cosine similarity for all environments
    obj1_z_axis = obj1_rot_mat[:, :, 2]  # (num_envs, 3)
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=obj1_z_axis.dtype)
    z_axis_cos = torch.abs(torch.sum(obj1_z_axis * world_z_axis, dim=-1))  # (num_envs,)

    # Calculate xy distance for all environments - similar to check_obj_in_receptacle
    xy_dist = torch.norm(obj1_pos[:, :2] - obj2_pos[:, :2], dim=-1)  # (num_envs,)
    obj1_obj2_size_xy_min = min(obj2_obj.size[:2])

    # Calculate velocity magnitude for all environments
    xyz_vel = torch.norm(obj1_vel, dim=-1)  # (num_envs,)

    # Check gripper distance for all environments
    gripper_far_obj1 = gripper_obj_far(env, obj1, th=gipper_th)  # (num_envs,)
    gripper_far_obj2 = gripper_obj_far(env, obj2, th=gipper_th)  # (num_envs,)
    gripper_far = gripper_far_obj1 & gripper_far_obj2  # (num_envs,)

    # Calculate success conditions for all environments
    obj1_is_standing = z_axis_cos > th_z_axis_cos  # (num_envs,)
    obj1_in_obj2 = xy_dist < obj1_obj2_size_xy_min * th_xy_dist  # (num_envs,)
    obj1_stable = xyz_vel < th_xyz_vel  # (num_envs,)
    print(f"gripper_far: {gripper_far}, obj1_is_standing: {obj1_is_standing}, obj1_in_obj2: {obj1_in_obj2}, obj1_stable: {obj1_stable}")
    return gripper_far & obj1_is_standing & obj1_in_obj2 & obj1_stable


def get_object_pos(env, name):
    return env.scene.rigid_objects[name].data.body_com_pos_w[0, 0, :]


def check_near(pos1, pos2, th=0.2):
    dis = torch.sqrt(torch.sum((pos1 - pos2) ** 2))
    return dis < th


def check_place_obj1_side_by_obj2(env, obj1: str, obj2: str, check_states: dict, gipper_th=0.25):
    """
    check if obj1 is placed side by side with obj2

    Args:
        env (Env): environment
        obj1 (str): name of object 1
        obj2 (str): name of object 2
        check_states (dict): check states
        check_states = {
            "gripper_far": True,   # obj1 and obj2 should be far from the gripper
            "contact": False,   # obj1 should not be in contact with obj2
            "side": 'right',    # relative position of obj1 to obj2, left, right, front, back
            "side_threshold": 0.25,    # threshold for distance between obj1 and obj2 in other directions, 0.25*(min(obj2_obj.size[:2]) + min(obj1_obj.size[:2]))/ 2
            "margin_threshold": [0.001, 0.1],    # threshold for distance between obj1 and obj2, 0.001 for min distance, 0.1 for max distance
            "parallel": [0, 1, 1],    # whether obj1 is parallel to the world axis, can be multiple, here means parallel to y and z axis
            "parallel_threshold": 0.95,    # threshold for parallel to world axis
            "stable_threshold": 0.5,    # threshold for stable, velocity vector length less than 0.5
        }

    Returns:
        torch.Tensor: success state of the task for all environments (num_envs,)

    """
    import torch

    # Get object positions - use torch.mean for multi-body objects like check_place_obj1_on_obj2
    if obj1 in env.cfg.objects:
        obj1_obj = env.cfg.objects[obj1]
        obj1_pos = torch.mean(env.scene.rigid_objects[obj1].data.body_com_pos_w, dim=1)  # (num_envs, 3)
        obj1_vel = torch.mean(env.scene.rigid_objects[obj1].data.body_com_vel_w, dim=1)  # (num_envs, 3)
        obj1_quat = torch.mean(env.scene.rigid_objects[obj1].data.body_com_quat_w, dim=1)  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
    else:
        obj1_obj = obj1
        obj1_name = obj1 if isinstance(obj1, str) else obj1.name
        obj1_pos = env.scene.state['articulation'][obj1_name]['root_pose'][:, :3]  # (num_envs, 3)
        obj1_quat = env.scene.state['articulation'][obj1_name]['root_pose'][:, 3:]  # (num_envs, 4)
        obj1_rot_mat = matrix_from_quat(obj1_quat)  # (num_envs, 3, 3)
        obj1_vel = env.scene.state['articulation'][obj1_name]['root_velocity'][:, :3]  # (num_envs, 3)

    if obj2 in env.cfg.objects:
        obj2_obj = env.cfg.objects[obj2]
        obj2_pos = torch.mean(env.scene.rigid_objects[obj2].data.body_com_pos_w, dim=1)  # (num_envs, 3)
    else:
        obj2_obj = obj2
        obj2_name = obj2 if isinstance(obj2, str) else obj2.name
        obj2_pos = env.scene.state['articulation'][obj2_name]['root_pose'][:, :3]  # (num_envs, 3)

    # Initialize all conditions as True for environments
    all_conditions = []

    # Check gripper distance for all environments
    if 'gripper_far' in check_states:
        gripper_far_obj1 = gripper_obj_far(env, obj1, th=gipper_th)  # (num_envs,)
        gripper_far_obj2 = gripper_obj_far(env, obj2, th=gipper_th)  # (num_envs,)
        gripper_far = gripper_far_obj1 & gripper_far_obj2  # (num_envs,)
        all_conditions.append(gripper_far)

    # Check contact for all environments
    if 'contact' in check_states:
        need_contact = check_states['contact']
        if hasattr(env.cfg, 'check_contact'):
            contact_result = env.cfg.check_contact(obj1_obj, obj2_obj)  # (num_envs,)
            if need_contact:
                all_conditions.append(contact_result)
            else:
                all_conditions.append(~contact_result)

    # Check stability for all environments
    if 'stable_threshold' in check_states:
        xyz_vel = torch.norm(obj1_vel, dim=-1)  # (num_envs,)
        stable = xyz_vel < check_states['stable_threshold']  # (num_envs,)
        all_conditions.append(stable)

    # Check parallel alignment for all environments
    if 'parallel' in check_states:
        parallel_conditions = []
        for index, axis in enumerate(check_states['parallel']):
            if axis:
                # Calculate parallel alignment for all environments
                obj_axis = obj1_rot_mat[:, :, index]  # (num_envs, 3)
                world_axis = [0, 0, 0]
                world_axis[index] = 1
                world_axis_tensor = torch.tensor(world_axis, device=env.device, dtype=obj_axis.dtype)
                axis_cos = torch.abs(torch.sum(obj_axis * world_axis_tensor, dim=-1))  # (num_envs,)
                parallel = axis_cos > check_states.get('parallel_threshold', 0.9)  # (num_envs,)
                parallel_conditions.append(parallel)
        if parallel_conditions:
            # All parallel conditions must be satisfied
            all_parallel = torch.stack(parallel_conditions, dim=0)
            all_conditions.append(torch.all(all_parallel, dim=0))

    # Check side positioning for all environments
    if 'side' in check_states:
        side = check_states['side']
        obj1_obj2_size_xy_min = (min(obj2_obj.size[:2]) + min(obj1_obj.size[:2])) / 2
        obj1_obj2_center_xydist = torch.norm(obj1_pos[:, :2] - obj2_pos[:, :2], dim=-1)  # (num_envs,)
        obj1_obj2_edge_xydist = obj1_obj2_center_xydist - obj1_obj2_size_xy_min  # (num_envs,)

        margin_threshold = check_states.get('margin_threshold', [0, 0])
        margin_threshold_min = margin_threshold[0]
        margin_threshold_max = margin_threshold[1]
        space_success = (obj1_obj2_edge_xydist > margin_threshold_min) & (obj1_obj2_edge_xydist < margin_threshold_max)  # (num_envs,)

        if side == "left":
            y_dist = torch.abs(obj1_pos[:, 1] - obj2_pos[:, 1])  # (num_envs,)
            side_condition = (obj1_pos[:, 0] > obj2_pos[:, 0]) & (y_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)
        elif side == "right":
            y_dist = torch.abs(obj1_pos[:, 1] - obj2_pos[:, 1])  # (num_envs,)
            side_condition = (obj1_pos[:, 0] < obj2_pos[:, 0]) & (y_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)
        elif side == "back":
            x_dist = torch.abs(obj1_pos[:, 0] - obj2_pos[:, 0])  # (num_envs,)
            side_condition = (obj1_pos[:, 1] > obj2_pos[:, 1]) & (x_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)
        elif side == "front":
            x_dist = torch.abs(obj1_pos[:, 0] - obj2_pos[:, 0])  # (num_envs,)
            side_condition = (obj1_pos[:, 1] < obj2_pos[:, 1]) & (x_dist < obj1_obj2_size_xy_min * check_states.get('side_threshold', 0.25))  # (num_envs,)
            all_conditions.append(space_success & side_condition)

    # Combine all conditions with AND operation
    if all_conditions:
        return torch.all(torch.stack(all_conditions, dim=0), dim=0)  # (num_envs,)
    else:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def check_obj_location_on_stove(env, stove, obj_name, threshold=0.08, need_knob_on=True):
    """
    Check if the object is on the stove and close to a burner and the knob is on (optional).
    Returns the location of the burner if the object is on the stove, close to a burner, and the burner is on (optional).
    None otherwise.
    """

    knobs_state = stove.get_knobs_state(env=env)
    obj_pos = env.scene.rigid_objects[obj_name].data.body_com_pos_w[..., 0, :]
    obj_on_stove = check_obj_fixture_contact(env, obj_name, stove)
    stove_pos = torch.tensor(stove.pos, device=env.device)
    stove_rot = T.euler2mat(torch.tensor([0.0, 0.0, stove.rot], device=env.device)).to(dtype=torch.float32)
    locations = []
    for env_id in range(len(obj_on_stove)):
        found_location = False
        if obj_on_stove[env_id]:
            for location, site in stove.burner_sites.items():
                if site[env_id] is not None:
                    burner_pos = stove_rot @ torch.tensor(site[env_id].GetAttribute("xformOp:translate").Get(), device=env.device) + stove_pos + env.scene.env_origins[env_id]
                    dist = torch.norm(burner_pos[:2] - obj_pos[env_id][:2])
                    obj_on_site = dist < threshold
                    knob_on = (
                        (0.35 <= torch.abs(knobs_state[location]) <= 2 * torch.pi - 0.35)
                        if location in knobs_state
                        else False
                    )
                    check_result = obj_on_site if not need_knob_on else obj_on_site and knob_on
                    if check_result:
                        found_location = True
                        locations.append(location)
                        break
        if not found_location:
            locations.append(None)
    return locations


def is_obj_z_up(env, obj_name="obj", th=5.0):
    obj_quat = env.scene[obj_name].data.root_quat_w
    r, p, y = euler_xyz_from_quat(obj_quat)
    th = th * torch.pi / 180.0
    return torch.logical_and(torch.abs(r) < th, torch.abs(p) < th)


def grasp_obj(env, obj_name='obj'):
    close_to_obj = torch.logical_not(gripper_obj_far(env, obj_name, th=0.15, eef_name="tool_left_arm"))
    left_hand_action_closed = env.action_manager.action[:, -2] > 0.5
    return torch.logical_and(close_to_obj, left_hand_action_closed)


def put_obj_to_coffee_machine(env, obj_name="obj", judge_obj_in_coffee_machine=None):
    obj_height = env.scene.rigid_objects[obj_name].data.body_com_pos_w[:, 0, 2]
    default_obj_height = env.scene.rigid_objects[obj_name].data.default_root_state[:, 2]
    higher_than_default = obj_height > default_obj_height

    close_to_obj = torch.logical_not(gripper_obj_far(env, obj_name, th=0.15, eef_name="tool_left_arm"))
    left_hand_action_open = env.action_manager.action[:, -2] < 0.5
    try:
        obj_in_coffee_machine = judge_obj_in_coffee_machine(env, obj_name, xy_thresh=0.09)
    except Exception:
        obj_in_coffee_machine = torch.tensor([False] * env.num_envs, dtype=torch.bool, device=env.device)

    return torch.logical_and(close_to_obj, left_hand_action_open) & obj_in_coffee_machine & higher_than_default


def obj_fixture_bbox_min_dist(env, obj_name, fixture):
    """
    Gets the minimum distance between a fixture and an object by computing the minimal axis-aligned bounding separation.
    """
    fix_pts = fixture.get_ext_sites(all_points=True, relative=False)
    fix_coords = np.array(fix_pts)
    fix_min = fix_coords.min(axis=0)
    fix_max = fix_coords.max(axis=0)

    all_sep_distances = []

    for i in range(env.cfg.scene.num_envs):
        obj_pos = env.scene.rigid_objects[obj_name].data.body_com_pos_w[i, 0, :].cpu().numpy()
        obj_quat = T.convert_quat(
            env.scene.rigid_objects[obj_name].data.body_com_quat_w[i, 0, :].cpu().numpy(), to="xyzw"
        )

        obj = env.cfg.objects[obj_name]
        obj_pts = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
        obj_coords = np.array(obj_pts)
        obj_min = obj_coords.min(axis=0)
        obj_max = obj_coords.max(axis=0)

        sep = np.zeros(3)
        for j, axis in enumerate(["x", "y", "z"]):
            if fix_max[j] < obj_min[j]:
                sep[j] = obj_min[j] - fix_max[j]
            elif obj_max[j] < fix_min[j]:
                sep[j] = fix_min[j] - obj_max[j]
            else:
                sep[j] = 0.0

        sep_distance = np.linalg.norm(sep)
        all_sep_distances.append(sep_distance)

    return torch.tensor(all_sep_distances, dtype=torch.float32, device=env.device)


def check_contact(env, geoms_1, geoms_2) -> torch.Tensor:
    """
    check if the two geoms are in contact
    """
    if env.common_step_counter == 1:
        if isinstance(geoms_1, str):
            geoms_1_sensor_path = f"{geoms_1}_contact"
        else:
            geoms_1_sensor_path = f"{geoms_1.task_name}_contact"

        if isinstance(geoms_2, str):
            geoms_2_sensor_path = geoms_2
        else:
            geoms_2_sensor_path = []
            geoms_2_prims = usd.get_prim_by_name(env.scene.stage.GetPseudoRoot(), geoms_2.name)
            for prim in geoms_2_prims:
                geoms_2_sensor_path.append([str(cp.GetPrimPath()) for cp in usd.get_prim_by_types(prim, ["Mesh", "Cube", "Cylinder"])])

        geoms_1_contact_paths = env.scene.sensors[geoms_1_sensor_path].contact_physx_view.sensor_paths

        for env_id in range(env.num_envs):
            if isinstance(geoms_2, str):
                filter_prim_paths_expr = [re.sub(r'env_\d+', f'env_{env_id}', geoms_2_sensor_path)]
            else:
                filter_prim_paths_expr = geoms_2_sensor_path[env_id]
            env.cfg.contact_queues[env_id].add(
                env.sim.physics_sim_view.create_rigid_contact_view(
                    geoms_1_contact_paths[env_id],
                    filter_patterns=filter_prim_paths_expr,
                    max_contact_data_count=200
                )
            )
    elif env.common_step_counter:
        contact_views = [env.cfg.contact_queues[env_id].pop() for env_id in range(env.num_envs)]
        return torch.tensor(
            [max(abs(view.get_contact_data(env.physics_dt)[0])) > 0 for view in contact_views],
            device=env.device,
        )
    return torch.tensor([False], device=env.device).repeat(env.num_envs)


def calculate_contact_force(env, geom) -> torch.Tensor:
    """
    calculate the contact force on the geom
    """
    if f"{geom}_contact" in env.scene.sensors:
        return torch.max(env.scene.sensors[f"{geom}_contact"].data.net_forces_w, dim=-1).values
    else:
        return torch.tensor([0.0], device=env.device).repeat(env.num_envs)
