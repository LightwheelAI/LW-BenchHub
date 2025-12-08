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

"""
Utility functions for matrix and vector transformations.

NOTE: Quaternion convention is (x, y, z, w)
"""

import math
import torch

from .consts import _NEXT_AXIS, _AXES2TUPLE

PI = torch.pi
EPS = torch.finfo(torch.float32).eps * 4.0


def convert_quat(q, to="xyzw"):
    """
    Convert quaternion from one convention to another (torch version).
    If to == 'xyzw', input is 'wxyz', and vice versa.

    Args:
        q (torch.Tensor): 4D quaternion tensor
        to (str): 'xyzw' or 'wxyz'

    Returns:
        torch.Tensor: Converted quaternion
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat_torch: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0) (torch version).

    Args:
        quaternion1 (torch.Tensor): (x, y, z, w) quaternion
        quaternion0 (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        torch.Tensor: (x, y, z, w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return torch.stack((
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
    ))


def quat_conjugate(quaternion):
    """
    Return conjugate of quaternion (torch version).

    Args:
        quaternion (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        torch.Tensor: (x, y, z, w) conjugate quaternion
    """
    return torch.stack((
        -quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]
    ))


def quat_inverse(quaternion):
    """
    Return inverse of quaternion (torch version).

    Args:
        quaternion (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        torch.Tensor: (x, y, z, w) inverse quaternion
    """
    return quat_conjugate(quaternion) / torch.dot(quaternion, quaternion)


def quat_distance(quaternion1, quaternion0):
    """
    Return distance between two quaternions, such that distance * quaternion0 = quaternion1 (torch version).

    Args:
        quaternion1 (torch.Tensor): (x, y, z, w) quaternion
        quaternion0 (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        torch.Tensor: (x, y, z, w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions (torch version).

    Args:
        quat0 (torch.Tensor): (x, y, z, w) start quaternion
        quat1 (torch.Tensor): (x, y, z, w) end quaternion
        fraction (float): interpolation fraction
        shortestpath (bool): whether to use shortest path

    Returns:
        torch.Tensor: (x, y, z, w) interpolated quaternion
    """
    def unit_vector(data):
        norm = torch.norm(data)
        if norm == 0:
            return data
        return data / norm

    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = torch.dot(q0, q1)
    if torch.abs(torch.abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        d = -d
        q1 = -q1
    angle = torch.acos(torch.clamp(d, -1, 1))
    if torch.abs(angle) < EPS:
        return q0
    isin = 1.0 / torch.sin(angle)
    q0 = q0 * (torch.sin((1.0 - fraction) * angle) * isin)
    q1 = q1 * (torch.sin(fraction * angle) * isin)
    q0 = q0 + q1
    return q0


def random_quat(rand=None, device=None):
    """
    Return uniform random unit quaternion (torch version).

    Args:
        rand (torch.Tensor or None): If specified, must be 3 random variables in [0,1]
        device: torch device

    Returns:
        torch.Tensor: (x, y, z, w) random quaternion
    """
    if rand is None:
        rand = torch.rand(3, device=device)
    else:
        assert len(rand) == 3
    r1 = torch.sqrt(1.0 - rand[0])
    r2 = torch.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return torch.stack((
        torch.sin(t1) * r1,
        torch.cos(t1) * r1,
        torch.sin(t2) * r2,
        torch.cos(t2) * r2,
    )).to(dtype=torch.float32)


def random_axis_angle(angle_limit=None, generator=None, device=None):
    """
    Sample an axis-angle rotation (torch version).

    Args:
        angle_limit (float or None): angle limit
        generator (torch.Generator or None): random generator
        device: torch device

    Returns:
        tuple: (axis, angle)
    """
    if angle_limit is None:
        angle_limit = 2.0 * math.pi
    randn = torch.randn(3, generator=generator, device=device)
    axis = randn / torch.norm(randn)
    angle = torch.empty(1, device=device).uniform_(0.0, angle_limit).item()
    return axis, angle


def vec(values, device=None):
    """
    Convert value tuple to torch vector.

    Args:
        values (array-like): numbers
        device: torch device

    Returns:
        torch.Tensor: vector
    """
    return torch.tensor(values, dtype=torch.float32, device=device)


def mat4(array, device=None):
    """
    Convert array to 4x4 matrix (torch version).

    Args:
        array (array-like): input array
        device: torch device

    Returns:
        torch.Tensor: 4x4 matrix
    """
    return torch.tensor(array, dtype=torch.float32, device=device).reshape(4, 4)


def mat2pose(hmat):
    """
    Convert homogeneous 4x4 matrix to pose (torch version).

    Args:
        hmat (torch.Tensor): 4x4 homogeneous matrix

    Returns:
        tuple: (position, quaternion)
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


def mat2quat(rmat):
    """
    Convert rotation matrix to quaternion (torch version).

    Args:
        rmat (torch.Tensor): 3x3 rotation matrix

    Returns:
        torch.Tensor: (x, y, z, w) quaternion
    """
    M = rmat[:3, :3].float()
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    K = torch.tensor([
        [m00 - m11 - m22, 0.0, 0.0, 0.0],
        [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
        [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
        [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
    ], dtype=torch.float32)
    K /= 3.0
    w, V = torch.linalg.eigh(K)
    inds = torch.tensor([3, 0, 1, 2])
    q1 = V[:, torch.argmax(w)][inds]
    if q1[0] < 0.0:
        q1 = -q1
    inds2 = torch.tensor([1, 2, 3, 0])
    return q1[inds2]


def euler2mat(euler):
    """
    Convert euler angles to rotation matrix (torch version).

    Args:
        euler (torch.Tensor): (r, p, y) euler angles

    Returns:
        torch.Tensor: 3x3 rotation matrix
    """
    euler = euler.float()
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)
    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = torch.empty(euler.shape[:-1] + (3, 3), dtype=torch.float32, device=euler.device)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def mat2euler(rmat, axes="sxyz"):
    """
    Convert rotation matrix to euler angles in radians (torch version).

    Args:
        rmat (torch.Tensor): 3x3 rotation matrix
        axes (str): axis sequence

    Returns:
        torch.Tensor: (r, p, y) euler angles
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = rmat[:3, :3].float()
    if repetition:
        sy = torch.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = torch.atan2(M[i, j], M[i, k])
            ay = torch.atan2(sy, M[i, i])
            az = torch.atan2(M[j, i], -M[k, i])
        else:
            ax = torch.atan2(-M[j, k], M[j, j])
            ay = torch.atan2(sy, M[i, i])
            az = torch.tensor(0.0, device=M.device)
    else:
        cy = torch.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = torch.atan2(M[k, j], M[k, k])
            ay = torch.atan2(-M[k, i], cy)
            az = torch.atan2(M[j, i], M[i, i])
        else:
            ax = torch.atan2(-M[j, k], M[j, j])
            ay = torch.atan2(-M[k, i], cy)
            az = torch.tensor(0.0, device=M.device)

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return torch.stack((ax, ay, az)).to(dtype=torch.float32)


def pose2mat(pose):
    """
    Convert pose to homogeneous matrix (torch version).

    Args:
        pose (tuple): (position, quaternion)

    Returns:
        torch.Tensor: 4x4 homogeneous matrix
    """
    homo_pose_mat = torch.zeros((4, 4), dtype=torch.float32, device=pose[0].device)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = pose[0].float()
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quat2mat(quaternion):
    """
    Convert quaternion to rotation matrix (torch version).

    Args:
        quaternion (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        torch.Tensor: 3x3 rotation matrix
    """
    inds = torch.tensor([3, 0, 1, 2], dtype=torch.long, device=quaternion.device)
    q = quaternion.clone().float()[inds]

    n = torch.dot(q, q)
    if n < EPS:
        return torch.eye(3, dtype=torch.float32, device=quaternion.device)
    q = q * torch.sqrt(torch.tensor(2.0, device=q.device) / n)
    q2 = torch.ger(q, q)
    return torch.stack([
        torch.stack([1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]]),
        torch.stack([q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]]),
        torch.stack([q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]])
    ])


def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle format (torch version).
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (torch.Tensor): (x, y, z, w) quaternion

    Returns:
        torch.Tensor: (ax, ay, az) axis-angle exponential coordinates
    """
    if not torch.is_tensor(quat):
        quat = torch.tensor(quat, dtype=torch.float32)
    w = quat[3].clamp(-1.0, 1.0)
    xyz = quat[:3]
    den = torch.sqrt(1.0 - w * w)
    if torch.isclose(den, torch.tensor(0.0, dtype=quat.dtype)):
        return torch.zeros(3, dtype=quat.dtype, device=quat.device)
    angle = 2.0 * torch.acos(w)
    return xyz * angle / den


def axisangle2quat(vec):
    """
    Convert axis-angle to quaternion (torch version).

    Args:
        vec (torch.Tensor): (ax, ay, az) axis-angle

    Returns:
        torch.Tensor: (x, y, z, w) quaternion
    """
    angle = torch.norm(vec)
    if torch.isclose(angle, torch.tensor(0.0, dtype=vec.dtype, device=vec.device)):
        return torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=vec.dtype, device=vec.device)
    axis = vec / angle
    q = torch.zeros(4, dtype=vec.dtype, device=vec.device)
    q[3] = torch.cos(angle / 2.0)
    q[:3] = axis * torch.sin(angle / 2.0)
    return q


def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Transform pose of C in frame A to frame B (torch version).

    Args:
        pose_A (torch.Tensor): 4x4 pose of C in A
        pose_A_in_B (torch.Tensor): 4x4 pose of A in B

    Returns:
        torch.Tensor: 4x4 pose of C in B
    """
    return torch.matmul(pose_A_in_B, pose_A)


def pose_inv(pose):
    """
    Compute inverse of homogeneous pose matrix (torch version).

    Args:
        pose (torch.Tensor): 4x4 pose matrix

    Returns:
        torch.Tensor: 4x4 inverse pose matrix
    """
    pose_inv = torch.zeros((4, 4), dtype=pose.dtype, device=pose.device)
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -torch.matmul(pose_inv[:3, :3], pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def _skew_symmetric_translation(pos_A_in_B):
    """
    Get skew symmetric translation matrix for frame conversion (torch version).

    Args:
        pos_A_in_B (torch.Tensor): (x, y, z) position

    Returns:
        torch.Tensor: 3x3 skew symmetric matrix
    """
    return torch.tensor([
        0.0, -pos_A_in_B[2], pos_A_in_B[1],
        pos_A_in_B[2], 0.0, -pos_A_in_B[0],
        -pos_A_in_B[1], pos_A_in_B[0], 0.0
    ], dtype=pos_A_in_B.dtype, device=pos_A_in_B.device).reshape(3, 3)


def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Convert linear and angular velocity from frame A to frame B (torch version).

    Args:
        vel_A (torch.Tensor): (vx, vy, vz) linear velocity in A
        ang_vel_A (torch.Tensor): (wx, wy, wz) angular velocity in A
        pose_A_in_B (torch.Tensor): 4x4 pose of A in B

    Returns:
        tuple: (linear velocity, angular velocity) in B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = torch.matmul(rot_A_in_B, vel_A) + torch.matmul(skew_symm, torch.matmul(rot_A_in_B, ang_vel_A))
    ang_vel_B = torch.matmul(rot_A_in_B, ang_vel_A)
    return vel_B, ang_vel_B


def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Convert force and torque from frame A to frame B (torch version).

    Args:
        force_A (torch.Tensor): (fx, fy, fz) force in A
        torque_A (torch.Tensor): (tx, ty, tz) torque in A
        pose_A_in_B (torch.Tensor): 4x4 pose of A in B

    Returns:
        tuple: (force, torque) in B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = torch.matmul(rot_A_in_B.T, force_A)
    torque_B = -torch.matmul(rot_A_in_B.T, torch.matmul(skew_symm, force_A)) + torch.matmul(rot_A_in_B.T, torque_A)
    return force_B, torque_B


def rotation_matrix(angle, direction, point=None):
    """
    Return 4x4 rotation matrix about axis and point (torch version).

    Args:
        angle (float): rotation angle
        direction (torch.Tensor): (ax, ay, az) axis
        point (torch.Tensor or None): (x, y, z) point

    Returns:
        torch.Tensor: 4x4 homogeneous rotation matrix
    """
    sina = torch.sin(torch.tensor(angle, dtype=direction.dtype, device=direction.device))
    cosa = torch.cos(torch.tensor(angle, dtype=direction.dtype, device=direction.device))
    direction = unit_vector(direction[:3])
    R = torch.eye(3, dtype=direction.dtype, device=direction.device) * cosa
    R += torch.ger(direction, direction) * (1.0 - cosa)
    direction_sina = direction * sina
    R += torch.stack([
        torch.stack([0.0, -direction_sina[2], direction_sina[1]]),
        torch.stack([direction_sina[2], 0.0, -direction_sina[0]]),
        torch.stack([-direction_sina[1], direction_sina[0], 0.0])
    ])
    M = torch.eye(4, dtype=direction.dtype, device=direction.device)
    M[:3, :3] = R
    if point is not None:
        point = point[:3].float()
        M[:3, 3] = point - torch.matmul(R, point)
    return M


def clip_translation(dpos, limit):
    """
    Limit translation vector norm (torch version).

    Args:
        dpos (torch.Tensor): translation vector
        limit (float): limit value

    Returns:
        tuple: (clipped vector, clipped flag)
    """
    input_norm = torch.norm(dpos)
    if input_norm > limit:
        return dpos * limit / input_norm, True
    else:
        return dpos, False


def clip_rotation(quat, limit):
    """
    Limit rotation quaternion angle (torch version).

    Args:
        quat (torch.Tensor): (x, y, z, w) quaternion
        limit (float): angle limit (radians)

    Returns:
        tuple: (clipped quaternion, clipped flag)
    """
    clipped = False
    quat = quat / torch.norm(quat)
    den = torch.sqrt(torch.clamp(1 - quat[3] * quat[3], min=0))
    if den == 0:
        return quat, clipped
    else:
        x = quat[0] / den
        y = quat[1] / den
        z = quat[2] / den
        a = 2 * torch.acos(quat[3])
    if torch.abs(a) > limit:
        a = limit * torch.sign(a) / 2
        sa = torch.sin(a)
        ca = torch.cos(a)
        quat = torch.stack([x * sa, y * sa, z * sa, ca])
        clipped = True
    return quat, clipped


def make_pose(translation, rotation):
    """
    Make homogeneous pose matrix from translation and rotation (torch version).

    Args:
        translation (torch.Tensor): (x, y, z) translation
        rotation (torch.Tensor): 3x3 rotation matrix

    Returns:
        torch.Tensor: 4x4 pose matrix
    """
    pose = torch.zeros((4, 4), dtype=rotation.dtype, device=rotation.device)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def unit_vector(data, axis=None, out=None):
    """
    Return unit vector (torch version).

    Args:
        data (torch.Tensor): input data
        axis (int or None): normalization axis
        out (torch.Tensor or None): output

    Returns:
        torch.Tensor: unit vector
    """
    if out is None:
        data = data.clone().float()
        if data.ndim == 1:
            data /= torch.sqrt(torch.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = data
        data = out
    length = torch.sum(data * data, dim=axis, keepdim=True)
    length = torch.sqrt(length)
    data /= length
    if out is None:
        return data


def get_orientation_error(target_orn, current_orn):
    """
    Return orientation error between two quaternions as 3D vector (torch version).

    Args:
        target_orn (torch.Tensor): (x, y, z, w) target quaternion
        current_orn (torch.Tensor): (x, y, z, w) current quaternion

    Returns:
        torch.Tensor: (ax, ay, az) orientation error
    """
    current_orn = torch.stack([current_orn[3], current_orn[0], current_orn[1], current_orn[2]])
    target_orn = torch.stack([target_orn[3], target_orn[0], target_orn[1], target_orn[2]])
    pinv = torch.zeros((3, 4), dtype=current_orn.dtype, device=current_orn.device)
    pinv[0, :] = torch.tensor([-current_orn[1], current_orn[0], -current_orn[3], current_orn[2]], dtype=current_orn.dtype, device=current_orn.device)
    pinv[1, :] = torch.tensor([-current_orn[2], current_orn[3], current_orn[0], -current_orn[1]], dtype=current_orn.dtype, device=current_orn.device)
    pinv[2, :] = torch.tensor([-current_orn[3], -current_orn[2], current_orn[1], current_orn[0]], dtype=current_orn.dtype, device=current_orn.device)
    orn_error = 2.0 * torch.matmul(pinv, target_orn)
    return orn_error


def get_pose_error(target_pose, current_pose):
    """
    Compute the error between target pose and current pose as a 6D vector (torch version).

    Args:
        target_pose (torch.Tensor): 4x4 target pose
        current_pose (torch.Tensor): 4x4 current pose

    Returns:
        torch.Tensor: 6D error vector
    """
    error = torch.zeros(6, dtype=target_pose.dtype, device=target_pose.device)
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (torch.cross(r1, r1d) + torch.cross(r2, r2d) + torch.cross(r3, r3d))
    error[:3] = pos_err
    error[3:] = rot_err
    return error


def rotate_2d_point(input, rot):
    """
    rotate a 2d vector counterclockwise (torch version)

    Args:
        input (torch.Tensor): 1d-array representing 2d vector
        rot (float): rotation value

    Returns:
        torch.Tensor: rotated 1d-array
    """
    input_x, input_y = input
    x = input_x * torch.cos(rot) - input_y * torch.sin(rot)
    y = input_x * torch.sin(rot) + input_y * torch.cos(rot)

    return torch.stack([x, y])
