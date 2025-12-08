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
import numpy as np

PI = np.pi
EPS = np.finfo(np.float32).eps * 4.0

from .consts import _NEXT_AXIS, _AXES2TUPLE


def convert_quat(q, to="xyzw"):
    """
    Convert quaternion from one convention to another.
    The 'to' parameter specifies the target convention.
    If to == 'xyzw', input is in 'wxyz' format, and vice versa.

    Args:
        q (np.array): 4D quaternion array
        to (str): 'xyzw' or 'wxyz', target convention

    Returns:
        np.array: Converted quaternion
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception("convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat_multiply(quaternion1, quaternion0):
    """
    Return the product of two quaternions (q1 * q0).

    Args:
        quaternion1 (np.array): (x, y, z, w) quaternion
        quaternion0 (np.array): (x, y, z, w) quaternion

    Returns:
        np.array: (x, y, z, w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def quat_conjugate(quaternion):
    """
    Return the conjugate of a quaternion.

    Args:
        quaternion (np.array): (x, y, z, w) quaternion

    Returns:
        np.array: (x, y, z, w) conjugate quaternion
    """
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )


def quat_inverse(quaternion):
    """
    Return the inverse of a quaternion.

    Args:
        quaternion (np.array): (x, y, z, w) quaternion

    Returns:
        np.array: (x, y, z, w) inverse quaternion
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def quat_distance(quaternion1, quaternion0):
    """
    Return the distance between two quaternions, such that distance * quaternion0 = quaternion1.

    Args:
        quaternion1 (np.array): (x, y, z, w) quaternion
        quaternion0 (np.array): (x, y, z, w) quaternion

    Returns:
        np.array: (x, y, z, w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))


def quat_slerp(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.

    Args:
        quat0 (np.array): (x, y, z, w) start quaternion
        quat1 (np.array): (x, y, z, w) end quaternion
        fraction (float): interpolation fraction
        shortestpath (bool): whether to use shortest path

    Returns:
        np.array: (x, y, z, w) interpolated quaternion
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        d = -d
        q1 *= -1.0
    angle = math.acos(np.clip(d, -1, 1))
    if abs(angle) < EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0


def random_quat(rand=None):
    """
    Return a uniformly distributed random unit quaternion.

    Args:
        rand (3-array or None): If specified, must be 3 random variables in [0, 1]

    Returns:
        np.array: (x, y, z, w) random quaternion
    """
    if rand is None:
        rand = np.random.rand(3)
    else:
        assert len(rand) == 3
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2, np.cos(t2) * r2),
        dtype=np.float32,
    )


def random_axis_angle(angle_limit=None, random_state=None):
    """
    Sample an axis-angle rotation by first sampling a random axis and then a random angle.

    Args:
        angle_limit (None or float): Limit for the angle range
        random_state (None or RandomState): Random number generator

    Returns:
        tuple: (axis, angle)
    """
    if angle_limit is None:
        angle_limit = 2.0 * np.pi

    if random_state is not None:
        assert isinstance(random_state, np.random.RandomState)
        npr = random_state
    else:
        npr = np.random

    random_axis = npr.randn(3)
    random_axis /= np.linalg.norm(random_axis)
    random_angle = npr.uniform(low=0.0, high=angle_limit)
    return random_axis, random_angle


def vec(values):
    """
    Convert a tuple of values to a numpy vector.

    Args:
        values (n-array): tuple of numbers

    Returns:
        np.array: vector
    """
    return np.array(values, dtype=np.float32)


def mat4(array):
    """
    Convert array to 4x4 matrix.

    Args:
        array (n-array): input array

    Returns:
        np.array: 4x4 matrix
    """
    return np.array(array, dtype=np.float32).reshape((4, 4))


def mat2pose(hmat):
    """
    Convert a 4x4 homogeneous matrix to pose.

    Args:
        hmat (np.array): 4x4 homogeneous matrix

    Returns:
        tuple: (position, quaternion)
    """
    pos = hmat[:3, 3]
    orn = mat2quat(hmat[:3, :3])
    return pos, orn


def mat2quat(rmat):
    """
    Convert rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x, y, z, w) quaternion
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]


def euler2mat(euler):
    """
    Convert euler angles to rotation matrix.

    Args:
        euler (np.array): (r, p, y) euler angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
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
    Convert rotation matrix to euler angles (radians).

    Args:
        rmat (np.array): 3x3 rotation matrix
        axes (str): axis sequence

    Returns:
        np.array: (r, p, y) euler angles
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.asarray(rmat, dtype=np.float32)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return vec((ax, ay, az))


def pose2mat(pose):
    """
    Convert pose to homogeneous matrix.

    Args:
        pose (2-tuple): (position, quaternion)

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=np.float32)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=np.float32)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat


def quat2mat(quaternion):
    """
    Convert quaternion to rotation matrix.

    Args:
        quaternion (np.array): (x, y, z, w) quaternion

    Returns:
        np.array: 3x3 rotation matrix
    """
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def quat2axisangle(quat):
    """
    Convert quaternion to axis-angle format, returns a unit vector scaled by the rotation angle.

    Args:
        quat (np.array): (x, y, z, w) quaternion

    Returns:
        np.array: (ax, ay, az) axis-angle
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def axisangle2quat(vec):
    """
    Convert axis-angle format to quaternion.

    Args:
        vec (np.array): (ax, ay, az) axis-angle

    Returns:
        np.array: (x, y, z, w) quaternion
    """
    angle = np.linalg.norm(vec)
    if math.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = vec / angle
    q = np.zeros(4)
    q[3] = np.cos(angle / 2.0)
    q[:3] = axis * np.sin(angle / 2.0)
    return q


def pose_in_A_to_pose_in_B(pose_A, pose_A_in_B):
    """
    Transform the pose of C in frame A to frame B.

    Args:
        pose_A (np.array): 4x4 pose of C in A
        pose_A_in_B (np.array): 4x4 pose of A in B

    Returns:
        np.array: 4x4 pose of C in B
    """
    return pose_A_in_B.dot(pose_A)


def pose_inv(pose):
    """
    Compute the inverse of a homogeneous pose matrix.

    Args:
        pose (np.array): 4x4 pose matrix

    Returns:
        np.array: 4x4 inverse pose matrix
    """
    pose_inv = np.zeros((4, 4))
    pose_inv[:3, :3] = pose[:3, :3].T
    pose_inv[:3, 3] = -pose_inv[:3, :3].dot(pose[:3, 3])
    pose_inv[3, 3] = 1.0
    return pose_inv


def _skew_symmetric_translation(pos_A_in_B):
    """
    Get the skew-symmetric matrix for coordinate frame transformation.

    Args:
        pos_A_in_B (np.array): (x, y, z) position

    Returns:
        np.array: 3x3 skew-symmetric matrix
    """
    return np.array(
        [
            0.0,
            -pos_A_in_B[2],
            pos_A_in_B[1],
            pos_A_in_B[2],
            0.0,
            -pos_A_in_B[0],
            -pos_A_in_B[1],
            pos_A_in_B[0],
            0.0,
        ]
    ).reshape((3, 3))


def vel_in_A_to_vel_in_B(vel_A, ang_vel_A, pose_A_in_B):
    """
    Transform linear and angular velocity from frame A to frame B.

    Args:
        vel_A (np.array): (vx, vy, vz) linear velocity in A
        ang_vel_A (np.array): (wx, wy, wz) angular velocity in A
        pose_A_in_B (np.array): 4x4 pose of A in B

    Returns:
        tuple: (linear velocity, angular velocity) in B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    vel_B = rot_A_in_B.dot(vel_A) + skew_symm.dot(rot_A_in_B.dot(ang_vel_A))
    ang_vel_B = rot_A_in_B.dot(ang_vel_A)
    return vel_B, ang_vel_B


def force_in_A_to_force_in_B(force_A, torque_A, pose_A_in_B):
    """
    Transform force and torque from frame A to frame B.

    Args:
        force_A (np.array): (fx, fy, fz) force in A
        torque_A (np.array): (tx, ty, tz) torque in A
        pose_A_in_B (np.array): 4x4 pose of A in B

    Returns:
        tuple: (force, torque) in B
    """
    pos_A_in_B = pose_A_in_B[:3, 3]
    rot_A_in_B = pose_A_in_B[:3, :3]
    skew_symm = _skew_symmetric_translation(pos_A_in_B)
    force_B = rot_A_in_B.T.dot(force_A)
    torque_B = -rot_A_in_B.T.dot(skew_symm.dot(force_A)) + rot_A_in_B.T.dot(torque_A)
    return force_B, torque_B


def rotation_matrix(angle, direction, point=None):
    """
    Return a 4x4 rotation matrix about a given axis and point.

    Args:
        angle (float): rotation angle
        direction (np.array): (ax, ay, az) rotation axis
        point (None or np.array): rotation point

    Returns:
        np.array: 4x4 homogeneous rotation matrix
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    R = np.array(((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        point = np.asarray(point[:3], dtype=np.float32)
        M[:3, 3] = point - np.dot(R, point)
    return M


def clip_translation(dpos, limit):
    """
    Limit the norm of a translation vector.

    Args:
        dpos (n-array): translation vector
        limit (float): limit value

    Returns:
        tuple: (clipped vector, clipped flag)
    """
    input_norm = np.linalg.norm(dpos)
    return (dpos * limit / input_norm, True) if input_norm > limit else (dpos, False)


def clip_rotation(quat, limit):
    """
    Limit the rotation angle of a quaternion.

    Args:
        quat (np.array): (x, y, z, w) quaternion
        limit (float): angle limit (radians)

    Returns:
        tuple: (clipped quaternion, clipped flag)
    """
    clipped = False
    quat = quat / np.linalg.norm(quat)
    den = np.sqrt(max(1 - quat[3] * quat[3], 0))
    if den == 0:
        return quat, clipped
    else:
        x = quat[0] / den
        y = quat[1] / den
        z = quat[2] / den
        a = 2 * math.acos(quat[3])
    if abs(a) > limit:
        a = limit * np.sign(a) / 2
        sa = math.sin(a)
        ca = math.cos(a)
        quat = np.array([x * sa, y * sa, z * sa, ca])
        clipped = True
    return quat, clipped


def make_pose(translation, rotation):
    """
    Create a homogeneous pose matrix from translation and rotation.

    Args:
        translation (np.array): (x, y, z) translation
        rotation (np.array): 3x3 rotation matrix

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    pose = np.zeros((4, 4))
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    pose[3, 3] = 1.0
    return pose


def unit_vector(data, axis=None, out=None):
    """
    Return the unit vector.

    Args:
        data (np.array): input data
        axis (None or int): normalization axis
        out (None or np.array): output

    Returns:
        np.array: unit vector
    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.asarray(data)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def get_orientation_error(target_orn, current_orn):
    """
    Return the orientation error between two quaternions as a 3D vector.

    Args:
        target_orn (np.array): (x, y, z, w) target quaternion
        current_orn (np.array): (x, y, z, w) current quaternion

    Returns:
        np.array: (ax, ay, az) orientation error
    """
    current_orn = np.array([current_orn[3], current_orn[0], current_orn[1], current_orn[2]])
    target_orn = np.array([target_orn[3], target_orn[0], target_orn[1], target_orn[2]])

    pinv = np.zeros((3, 4))
    pinv[0, :] = [-current_orn[1], current_orn[0], -current_orn[3], current_orn[2]]
    pinv[1, :] = [-current_orn[2], current_orn[3], current_orn[0], -current_orn[1]]
    pinv[2, :] = [-current_orn[3], -current_orn[2], current_orn[1], current_orn[0]]
    orn_error = 2.0 * pinv.dot(np.array(target_orn))
    return orn_error


def get_pose_error(target_pose, current_pose):
    """
    Compute the error between target pose and current pose as a 6D vector.

    Args:
        target_pose (np.array): 4x4 target pose
        current_pose (np.array): 4x4 current pose

    Returns:
        np.array: 6D error vector
    """
    error = np.zeros(6)
    target_pos = target_pose[:3, 3]
    current_pos = current_pose[:3, 3]
    pos_err = target_pos - current_pos
    r1 = current_pose[:3, 0]
    r2 = current_pose[:3, 1]
    r3 = current_pose[:3, 2]
    r1d = target_pose[:3, 0]
    r2d = target_pose[:3, 1]
    r3d = target_pose[:3, 2]
    rot_err = 0.5 * (np.cross(r1, r1d) + np.cross(r2, r2d) + np.cross(r3, r3d))
    error[:3] = pos_err
    error[3:] = rot_err
    return error


def rotate_2d_point(input, rot):
    """
    rotate a 2d vector counterclockwise

    Args:
        input (np.array): 1d-array representing 2d vector
        rot (float): rotation value

    Returns:
        np.array: rotated 1d-array
    """
    input_x, input_y = input
    x = input_x * np.cos(rot) - input_y * np.sin(rot)
    y = input_x * np.sin(rot) + input_y * np.cos(rot)

    return np.array([x, y])
