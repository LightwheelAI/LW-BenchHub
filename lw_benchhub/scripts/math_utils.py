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
import torch


# -------------------- SO3/SE3 utils --------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_conj(q):  # wxyz
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([w, -x, -y, -z], axis=-1)


def quat_mul(q1, q2):  # both wxyz
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_to_R(q):  # wxyz -> (...,3,3)
    q = quat_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def R_to_axis_angle(R):  # (...,3,3) -> (...,3), ω=θ·u
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    eps = 1e-6
    omega = np.zeros(R.shape[:-2] + (3,), dtype=np.float64)
    RmRt = np.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], axis=-1)
    small = theta < eps
    if np.any(~small):
        axis = RmRt / (2.0 * np.sin(theta)[..., None])
        omega[~small] = axis[~small] * theta[..., None][~small]
    if np.any(small):
        omega[small] = 0.5 * RmRt[small]
    return omega


def se3_compose(p1, q1, p2, q2):  # (p1,q1)∘(p2,q2)
    R1 = quat_to_R(q1)
    p = p1 + np.einsum('...ij,...j->...i', R1, p2)
    q = quat_mul(q1, q2)
    return p, quat_normalize(q)


def se3_inverse(p, q):  # (p,q)^{-1}
    R = quat_to_R(q)
    Rt = np.swapaxes(R, -1, -2)
    p_inv = -np.einsum('...ij,...j->...i', Rt, p)
    q_inv = quat_conj(quat_normalize(q))
    return p_inv, q_inv


def quat_to_axis_angle(q_wxyz):  # (...,4) -> (...,3)
    # Convert quaternion to axis-angle vector ω=θ·u (in radians), taking the shortest arc, stable for small angles
    q = quat_normalize(q_wxyz)
    sign = np.where(q[..., 0:1] < 0.0, -1.0, 1.0)
    q = q * sign
    w = q[..., 0:1]
    v = q[..., 1:4]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    eps = 1e-8
    theta = 2.0 * np.arctan2(s, np.clip(np.abs(w), eps, None))
    axis = np.where(s > eps, v / s, np.zeros_like(v))
    omega = axis * theta
    small = (s < 1e-6).squeeze(-1)
    if np.any(small):
        omega[small] = 2.0 * v[small]
    return omega


def quat_to_r6d(q_wxyz):  # (...,4) -> (...,6)
    R = quat_to_R(q_wxyz)  # (...,3,3)
    # [col0, col1] = [R[:, 0], R[:, 1]]
    rotation_6d = R[..., :, :2].reshape(*R.shape[:-2], 6)  # (...,6)
    return rotation_6d


def r6d_to_quat(r6d):  # (...,6) -> (...,4) wxyz
    r6d = np.asarray(r6d, dtype=np.float64)
    # (..., 3, 2)
    if r6d.ndim == 1:
        r6d = r6d.reshape(3, 2)
        single = True
    else:
        r6d = r6d.reshape(*r6d.shape[:-1], 3, 2)
        single = False

    col0 = r6d[..., :, 0]  # (..., 3)
    col1 = r6d[..., :, 1]  # (..., 3)

    col0_norm = col0 / (np.linalg.norm(col0, axis=-1, keepdims=True) + 1e-8)

    # Gram-Schmidt orthogonalization: col1 = col1 - (col1·col0) * col0
    dot = np.sum(col1 * col0_norm, axis=-1, keepdims=True)
    col1_ortho = col1 - dot * col0_norm
    col1_norm = col1_ortho / (np.linalg.norm(col1_ortho, axis=-1, keepdims=True) + 1e-8)

    # col2_norm = col0_norm × col1_norm
    col2_norm = np.cross(col0_norm, col1_norm)

    # build rotation matrix
    R = np.stack([col0_norm, col1_norm, col2_norm], axis=-1)  # (..., 3, 3)

    # quaternion from rotation matrix
    trace = np.trace(R, axis1=-2, axis2=-1)
    w = np.sqrt(np.maximum(1 + trace, 0)) / 2
    x = np.sign(R[..., 2, 1] - R[..., 1, 2]) * np.sqrt(np.maximum(1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], 0)) / 2
    y = np.sign(R[..., 0, 2] - R[..., 2, 0]) * np.sqrt(np.maximum(1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], 0)) / 2
    z = np.sign(R[..., 1, 0] - R[..., 0, 1]) * np.sqrt(np.maximum(1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], 0)) / 2

    q = np.stack([w, x, y, z], axis=-1)
    q = quat_normalize(q)

    if single:
        return q[0]
    return q


def pose_left_multiply(a, b):
    dp_rel = b[:, :3]      # (T,3)
    dq_rel = b[:, 3:]      # (T,4) wxyz
    p_tgt = a[:, :3]      # (T,3)
    q_tgt = a[:, 3:]      # (T,4) wxyz
    # T_curr =(ΔT)^{-1} ∘ T_target
    dp_rel_inv, dq_rel_inv = se3_inverse(dp_rel, dq_rel)
    p_curr, q_curr = se3_compose(p_tgt, q_tgt, dp_rel_inv, dq_rel_inv)  # (T,3),(T,4)
    return np.concatenate([p_curr, q_curr], axis=-1)


def pose_right_multiply(a, b):
    p_cur = a[:, :3]
    q_cur = a[:, 3:]
    dp = b[:, :3]
    dq = b[:, 3:]
    # T_tgt = ΔT ∘ T_cur
    p_tgt, q_tgt = se3_compose(p_cur, q_cur, dp, dq)
    return np.concatenate([p_tgt, q_tgt], axis=-1)


def compute_delta_pose(cur, tgt):
    p_cur = cur[:, :3]
    q_cur = cur[:, 3:]
    p_tgt = tgt[:, :3]
    q_tgt = tgt[:, 3:]
    #  (cur)^{-1}
    p_cur_inv, q_cur_inv = se3_inverse(p_cur, q_cur)
    #  delta = tgt ∘ (cur)^{-1}
    dp, dq = se3_compose(p_cur_inv, q_cur_inv, p_tgt, q_tgt)
    return np.concatenate([dp, dq], axis=-1)
# ----


def quat_to_axis_angle_torch(q):
    """
    Input: q (..., 4) = [w, x, y, z]
    Output: omega (..., 3) = [wx, wy, wz]
    """

    q = q / torch.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    theta = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

    sin_half = torch.sin(theta / 2)
    axis = torch.stack([x, y, z], dim=-1)

    mask = torch.abs(sin_half) > 1e-8
    omega = torch.where(
        mask.unsqueeze(-1),
        axis / sin_half.unsqueeze(-1) * theta.unsqueeze(-1),
        axis * 2
    )
    return omega


def quat_normalize_torch(q: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(q.dtype).eps
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
    return q / norm


def quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = torch.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_to_R_torch(q: torch.Tensor) -> torch.Tensor:
    q = quat_normalize_torch(q)
    w, x, y, z = torch.moveaxis(q, -1, 0)
    R = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_to_r6d_torch(q_wxyz: torch.Tensor) -> torch.Tensor:  # (...,4) -> (...,6)

    R = quat_to_R_torch(q_wxyz)
    rotation_6d = R[..., :, :2].reshape(*R.shape[:-2], 6)  # (...,6)
    return rotation_6d


def axis_angle_to_quat_torch(omega: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    denom = torch.where(theta > 1e-8, theta, torch.ones_like(theta))
    axis = omega / denom
    axis = torch.where(theta > 1e-8, axis, torch.zeros_like(axis))
    half = 0.5 * theta
    qw = torch.cos(half)
    qv = axis * torch.sin(half)
    return torch.cat([qw, qv], dim=-1)


def r6d_to_quat_torch(r6d: torch.Tensor) -> torch.Tensor:  # (...,6) -> (...,4) wxyz

    if r6d.ndim == 1:
        r6d = r6d.reshape(3, 2)
        single = True
    else:
        r6d = r6d.reshape(*r6d.shape[:-1], 3, 2)
        single = False

    col0 = r6d[..., :, 0]  # (..., 3)
    col1 = r6d[..., :, 1]  # (..., 3)

    col0_norm = col0 / (torch.linalg.norm(col0, dim=-1, keepdim=True) + 1e-8)

    # Gram-Schmidt orthogonalization: col1 = col1 - (col1·col0) * col0
    dot = torch.sum(col1 * col0_norm, dim=-1, keepdim=True)
    col1_ortho = col1 - dot * col0_norm
    col1_norm = col1_ortho / (torch.linalg.norm(col1_ortho, dim=-1, keepdim=True) + 1e-8)

    # col2_norm = col0_norm × col1_norm
    col2_norm = torch.cross(col0_norm, col1_norm, dim=-1)

    # build rotation matrix
    R = torch.stack([col0_norm, col1_norm, col2_norm], dim=-1)  # (..., 3, 3)

    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    w = torch.sqrt(torch.clamp(1 + trace, min=0)) / 2

    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    x = torch.sign(R21 - R12) * torch.sqrt(torch.clamp(1 + R00 - R11 - R22, min=0)) / 2
    y = torch.sign(R02 - R20) * torch.sqrt(torch.clamp(1 - R00 + R11 - R22, min=0)) / 2
    z = torch.sign(R10 - R01) * torch.sqrt(torch.clamp(1 - R00 - R11 + R22, min=0)) / 2

    q = torch.stack([w, x, y, z], dim=-1)
    q = quat_normalize_torch(q)

    if single:
        return q[0]
    return q


def se3_compose_torch(p1: torch.Tensor, q1: torch.Tensor, p2: torch.Tensor, q2: torch.Tensor):
    R1 = quat_to_R_torch(q1)
    p = p1 + torch.einsum('...ij,...j->...i', R1, p2)
    q = quat_mul_torch(q1, q2)
    return p, quat_normalize_torch(q)


def pose_right_multiply_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    p_cur = a[:, :3]
    q_cur = a[:, 3:]
    dp = b[:, :3]
    dq = b[:, 3:]
    p_tgt, q_tgt = se3_compose_torch(p_cur, q_cur, dp, dq)
    action_eef_abs = torch.cat([p_tgt, q_tgt], dim=-1).to(dtype=torch.float32)
    return action_eef_abs
