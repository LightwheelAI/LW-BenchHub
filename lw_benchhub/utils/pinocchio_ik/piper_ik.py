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

import os
from typing import Optional, Tuple

import numpy as np

try:
    import pinocchio as pin
    from pinocchio.robot_wrapper import RobotWrapper
    from pinocchio import casadi as cpin
    import casadi
except Exception as _e:  # pragma: no cover
    pin = None
    RobotWrapper = None
    cpin = None
    casadi = None


def _default_urdf_path() -> Tuple[str, list[str]]:
    """Resolve the default Piper URDF path and package dirs based on this file location."""
    # This file: lw_benchhub/utils/pinocchio_ik/piper_ik.py
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lw_benchhub_dir = os.path.dirname(os.path.dirname(cur_dir))
    # utils/piper_description/urdf/piper_description.urdf
    urdf_dir = os.path.join(lw_benchhub_dir, "utils", "piper_description", "urdf")
    urdf_path = os.path.join(urdf_dir, "piper_description.urdf")
    # For package://piper_description/... resolution, provide the parent that contains 'piper_description'
    package_dirs = [os.path.join(lw_benchhub_dir, "utils")]
    return urdf_path, package_dirs


class PiperPinocchioIK:
    """Lightweight DLS IK solver for Piper using Pinocchio.

    - Locks gripper joints (joint7, joint8) in a reduced model.
    - Targets an EE frame resolved from URDF (hand_link/gripper_base/link6/etc.).
      TCP offset should be handled by caller if needed.
    - Provides batched solve with warm-start.
    """

    def __init__(self, urdf_path: Optional[str] = None, package_dirs: Optional[list[str]] = None,
                 dls_lambda: float = 1e-2, max_iters: int = 50, tol: float = 1e-4, step_gain: float = 1.0):
        if pin is None or RobotWrapper is None:
            raise ImportError("pinocchio is required for PiperPinocchioIK. Please install pinocchio>=3.6.")
        if cpin is None or casadi is None:
            raise ImportError("casadi and pinocchio.casadi are required to match piper_pinocchio.py logic.")

        if urdf_path is None or package_dirs is None:
            urdf_path, package_dirs = _default_urdf_path()

        # Load full robot then build reduced (lock gripper joints)
        self._full_robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs)
        model = self._full_robot.model

        # Common Piper naming: lock finger joints if present
        joints_to_lock = []
        for jname in ("joint7", "joint8", "joint4"):
            try:
                jid = model.getJointId(jname)
                if jid > 0:
                    joints_to_lock.append(jname)
            except Exception:
                continue

        ref_q = np.zeros(model.nq)
        self._robot = self._full_robot.buildReducedRobot(list_of_joints_to_lock=joints_to_lock,
                                                         reference_configuration=ref_q)
        self._model = self._robot.model
        self._data = self._model.createData()

        # Add an EE frame 'ee' to joint6 with identity transform (align with piper_pinocchio.py)
        try:
            j6 = self._model.getJointId('joint6')
        except Exception as e:
            raise ValueError("Required joint 'joint6' not found in Piper URDF.") from e
        self._model.addFrame(
            pin.Frame('ee', j6, pin.SE3(pin.Quaternion(1.0, 0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])), pin.FrameType.OP_FRAME)
        )
        self._frame_id = self._model.getFrameId('ee')
        self._frame_name = 'ee'

        # Params
        self._lambda = float(dls_lambda)
        self._max_iters = int(max_iters)
        self._tol = float(tol)
        self._gain = float(step_gain)

        # Runtime warm-start cache
        self._last_q: Optional[np.ndarray] = None  # shape: (batch, nq)

        # Joint limits
        self._q_lower = np.array(self._model.lowerPositionLimit)
        self._q_upper = np.array(self._model.upperPositionLimit)

        # Build collision model using full robot for checks
        urdf_path_resolved, package_dirs_resolved = (urdf_path, package_dirs) if urdf_path is not None else _default_urdf_path()
        self._geom_model = pin.buildGeomFromUrdf(self._full_robot.model, urdf_path_resolved, pin.GeometryType.COLLISION, package_dirs_resolved)
        for i in range(4, 9):
            for j in range(0, 3):
                try:
                    self._geom_model.addCollisionPair(pin.CollisionPair(i, j))
                except Exception:
                    pass
        self._geom_data = pin.GeometryData(self._geom_model)

        # Joint regularization weights (prefer certain joints to move less)
        self._joint_weights = np.ones(self._model.nq, dtype=np.float64)
        # for jname in ["joint1"]:
        #     try:
        #         jid = self._model.getJointId(jname)
        #         if jid > 0:
        #             idx_q = self._model.joints[jid].idx_q
        #             nq_j = self._model.joints[jid].nq
        #             self._joint_weights[idx_q: idx_q + nq_j] *= 40
        #     except Exception:
        #         pass
        # Precompute sqrt-weight matrix for Casadi objective
        self._sqrt_w = casadi.DM(np.sqrt(self._joint_weights))

        # Casadi symbolic model and optimizer
        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()
        self._cq = casadi.SX.sym("q", self._model.nq, 1)
        self._cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self._cmodel, self._cdata, self._cq)
        self._gripper_id = self._model.getFrameId("ee")
        self._error_fun = casadi.Function(
            "error",
            [self._cq, self._cTf],
            [casadi.vertcat(cpin.log6(self._cdata.oMf[self._gripper_id].inverse() * cpin.SE3(self._cTf)).vector)],
        )

        self._opti = casadi.Opti()
        self._var_q = self._opti.variable(self._model.nq)
        self._param_tf = self._opti.parameter(4, 4)
        self._qref_param = self._opti.parameter(self._model.nq)
        totalcost = casadi.sumsqr(self._error_fun(self._var_q, self._param_tf))
        # Weighted regularization on delta-q to keep preferred joints steadier
        delta_q = self._var_q - self._qref_param
        regularization = casadi.sumsqr(casadi.diag(self._sqrt_w) @ delta_q)
        self._opti.subject_to(self._opti.bounded(self._q_lower, self._var_q, self._q_upper))
        self._opti.minimize(20 * totalcost + 0.01 * regularization)
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self._opti.solver("ipopt", opts)

        # History for warm-start
        self._init_data = np.zeros(self._model.nq)
        self._history_data = np.zeros(self._model.nq)
        # Output-side protection: error gating and delta limiting (to reduce jitter on solver issues)
        self._err_gate_thresh: float = 0.4  # gate if pose error larger than this
        self._max_delta_per_step: float = 0.4  # max per-joint change per step (rad)

    def _resolve_target_frame(self) -> tuple[int, str]:
        """Pick a sensible EE frame present in the URDF.

        Preference order: hand_link -> gripper_base -> link6 -> link7 -> last frame.
        """
        candidate_names = [
            "hand_link",
            "gripper_base",
            "link6",
            "link7",
        ]
        for name in candidate_names:
            try:
                fid = self._model.getFrameId(name)
                # Ensure id is valid
                if 0 <= fid < len(self._model.frames):
                    return fid, name
            except Exception:
                continue
        # Fallback: use the last frame (typically last link)
        if len(self._model.frames) == 0:
            raise ValueError("No frames found in Piper URDF model.")
        return len(self._model.frames) - 1, self._model.frames[-1].name

    @staticmethod
    def _homogeneous_from_pose(pos_wxyz: np.ndarray) -> np.ndarray:
        """Construct 4x4 homogeneous matrix from pos(3) + quat(wxyz,4) as float64."""
        vec = np.asarray(pos_wxyz, dtype=np.float64)
        t = np.eye(4, dtype=np.float64)
        t[0:3, 3] = vec[0:3]
        w, x, y, z = vec[3:7]
        # Ensure double precision for pinocchio Quaternion
        R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        t[0:3, 0:3] = R
        return t

    def _ensure_batch(self, batch_size: int):
        if self._last_q is None or self._last_q.shape[0] != batch_size:
            self._last_q = np.zeros((batch_size, self._model.nq), dtype=np.float64)

    def _ik_single(self, target_T: np.ndarray, warm_q: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool, bool, np.ndarray]:
        """Solve single-target IK using Casadi Opti to mirror piper_pinocchio.py."""
        if warm_q is not None:
            self._init_data = warm_q.astype(np.float64)
        self._opti.set_initial(self._var_q, self._init_data)
        self._opti.set_value(self._param_tf, target_T)
        # reference configuration for weighted regularization
        try:
            qref = self._init_data if warm_q is None else warm_q
            self._opti.set_value(self._qref_param, qref)
        except Exception:
            self._opti.set_value(self._qref_param, self._init_data)
        try:
            sol = self._opti.solve_limited()
            sol_q = np.asarray(self._opti.value(self._var_q)).reshape(-1)
            # Warm-start policy
            if self._init_data is not None:
                max_diff = float(np.max(np.abs(self._history_data - sol_q)))
                self._init_data = sol_q
                if max_diff > 30.0 / 180.0 * np.pi:
                    self._init_data = np.zeros(self._model.nq)
            else:
                self._init_data = sol_q
            self._history_data = sol_q

            # Feedforward torque (RNEA) with zero velocities/accelerations
            v = np.zeros_like(sol_q)
            tau_ff = pin.rnea(self._model, self._data, sol_q, v, np.zeros(self._model.nv))
            is_collision = self.check_self_collision(sol_q)
            return sol_q, True, bool(is_collision), np.asarray(tau_ff)
        except Exception:
            return np.zeros(self._model.nq), False, False, np.zeros(self._model.nv)

    def solve_pose_to_joints(self, targets_pos_wxyz: np.ndarray,
                             warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve batched IK.

        Args:
            targets_pos_wxyz: shape (B, 7) as [x,y,z,w,x,y,z] (wxyz quaternion).
            warm_start: optional (B, nq) array as starting q.

        Returns:
            q_solutions: (B, nq) joint positions.
            success: (B,) bool mask.
        """
        B = int(targets_pos_wxyz.shape[0])
        self._ensure_batch(B)

        q_out = np.zeros((B, self._model.nq), dtype=np.float64)
        success = np.zeros((B,), dtype=bool)
        T_targets = [self._homogeneous_from_pose(targets_pos_wxyz[i]) for i in range(B)]
        for i in range(B):
            warm_q = None if warm_start is None else warm_start[i]
            sol_q, converged, _, _ = self._ik_single(T_targets[i], warm_q)

            # Compute pose error norm for gating
            err_norm = float('inf')
            data_chk = self._model.createData()
            pin.forwardKinematics(self._model, data_chk, sol_q)
            pin.updateFramePlacements(self._model, data_chk)
            cur = data_chk.oMf[self._frame_id]
            Rt = np.asarray(T_targets[i], dtype=np.float64)
            target_se3 = pin.SE3(Rt[0:3, 0:3], Rt[0:3, 3])
            err6 = pin.log6(cur.inverse() * target_se3).vector
            err_norm = float(np.linalg.norm(np.asarray(err6)))

            prev_q = None
            if self._last_q is not None and self._last_q.shape == (B, self._model.nq):
                prev_q = self._last_q[i]

            # Gate on failure or excessive error
            use_prev = (not converged) or (err_norm > self._err_gate_thresh)
            chosen_q = prev_q if (use_prev and prev_q is not None) else sol_q

            # Delta limiting to avoid jitter
            if prev_q is not None:
                delta = chosen_q - prev_q
                delta = np.clip(delta, -self._max_delta_per_step, self._max_delta_per_step)
                chosen_q = prev_q + delta

            q_out[i] = chosen_q
            success[i] = converged and (err_norm <= self._err_gate_thresh)

        self._last_q = q_out.copy()
        return q_out, success

    def check_self_collision(self, q: np.ndarray, gripper: np.ndarray = np.array([0.0, 0.0])) -> bool:
        """Self-collision check using full model geometry, following script's approach."""
        try:
            # Combine with gripper values if matches full nq
            full_q = q
            if full_q.shape[0] + gripper.shape[0] == self._full_robot.model.nq:
                full_q = np.concatenate([q, gripper], axis=0)
            pin.forwardKinematics(self._full_robot.model, self._full_robot.data, full_q)
            pin.updateGeometryPlacements(self._full_robot.model, self._full_robot.data, self._geom_model, self._geom_data)
            collision = pin.computeCollisions(self._geom_model, self._geom_data, False)
            return bool(collision)
        except Exception:
            return False

    def reset(self):
        self._last_q = None
        self._init_data = np.zeros(self._model.nq)
        self._history_data = np.zeros(self._model.nq)

    # ------------add FK function--------------

    def fk_ee(self, q: np.ndarray) -> np.ndarray:
        """
        Forward kinematics for the IK end-effector frame 'ee'.

        Args:
            q: Joint configuration(s) for the *reduced* model
               - shape (nq,)  or (B, nq)

        Returns:
            ee_poses:
                - if input is (nq,),  returns shape (7,)   [x, y, z, w, x, y, z]
                - if input is (B,nq), returns shape (B,7)  [x, y, z, w, x, y, z] per batch
            All poses are **w.r.t. Pinocchio world frame**, 即 ee@world.
        """
        if pin is None:
            raise ImportError("pinocchio is required for fk_ee.")

        q_arr = np.asarray(q, dtype=np.float64)

        single_input = False
        if q_arr.ndim == 1:
            q_arr = q_arr.reshape(1, -1)
            single_input = True

        B, nq = q_arr.shape
        if nq != self._model.nq:
            raise ValueError(f"fk_ee expected q dimension {self._model.nq}, got {nq}.")

        ee_poses = np.zeros((B, 7), dtype=np.float64)

        for i in range(B):
            qi = q_arr[i]
            data = self._model.createData()
            pin.forwardKinematics(self._model, data, qi)
            pin.updateFramePlacements(self._model, data)

            oMf_ee = data.oMf[self._frame_id]  # SE3

            p = oMf_ee.translation  # (3,)
            R = oMf_ee.rotation     # (3,3)

            # Rotation -> quaternion in wxyz
            quat = pin.Quaternion(R)
            quat.normalize()
            w = float(quat.w)
            x = float(quat.x)
            y = float(quat.y)
            z = float(quat.z)

            ee_poses[i, 0:3] = p
            ee_poses[i, 3:7] = np.array([w, x, y, z], dtype=np.float64)  # [x,y,z,w,x,y,z]

        if single_input:
            return ee_poses[0]
        return ee_poses
