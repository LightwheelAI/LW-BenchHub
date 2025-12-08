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
    """Resolve the default X7S URDF path and package dirs based on this file location.

    Returns absolute path to the URDF and package directories list to resolve
    package://X7S/... mesh resources. The package root is the directory that
    contains the "X7S" folder (typically lw_benchhub/utils).
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lw_benchhub_dir = os.path.dirname(os.path.dirname(cur_dir))
    urdf_dir = os.path.join(lw_benchhub_dir, "utils", "X7S", "urdf")
    urdf_path = os.path.join(urdf_dir, "X7S.urdf")
    package_dirs = [os.path.join(lw_benchhub_dir, "utils")]
    return urdf_path, package_dirs


class X7SPinocchioIK:
    """Lightweight IK solver for X7S arm using Pinocchio.

    - Builds a reduced model that locks all joints except a specified set (default joint5..joint11 for left arm).
    - Adds an operational frame 'ee' on a specified joint with identity transform (default joint11).
    - Uses Casadi Opti with log6 pose error and joint regularization, mirroring Piper.
    - Provides batched solve with warm-start.
    """

    def __init__(self, urdf_path: Optional[str] = None, package_dirs: Optional[list[str]] = None,
                 dls_lambda: float = 1e-2, max_iters: int = 50, tol: float = 1e-4, step_gain: float = 1.0,
                 joints_to_keep: Optional[list[str]] = None, ee_joint_name: Optional[str] = None):
        if pin is None or RobotWrapper is None:
            raise ImportError("pinocchio is required for X7SPinocchioIK. Please install pinocchio>=3.6.")
        if cpin is None or casadi is None:
            raise ImportError("casadi and pinocchio.casadi are required for X7SPinocchioIK.")

        if urdf_path is None or package_dirs is None:
            urdf_path, package_dirs = _default_urdf_path()

        self._full_robot = RobotWrapper.BuildFromURDF(urdf_path, package_dirs)
        full_model = self._full_robot.model

        # Configure which joints are kept (others are locked)
        if joints_to_keep is None:
            joints_to_keep = ["joint5", "joint6", "joint7", "joint8", "joint9", "joint10", "joint11"]
        keep_joint_names = set(joints_to_keep)
        joints_to_lock: list[str] = []
        for jid in range(1, full_model.njoints):
            jname = full_model.names[jid]
            if jname not in keep_joint_names:
                joints_to_lock.append(jname)

        ref_q_full = np.zeros(full_model.nq)
        self._robot = self._full_robot.buildReducedRobot(list_of_joints_to_lock=joints_to_lock,
                                                         reference_configuration=ref_q_full)
        self._model = self._robot.model

        # Add EE frame at the specified joint (default: joint11 for left arm)
        if ee_joint_name is None:
            ee_joint_name = 'joint11'
        try:
            jee = self._model.getJointId(ee_joint_name)
        except Exception as e:
            raise ValueError(f"Required joint '{ee_joint_name}' not found in X7S URDF (reduced model).") from e
        self._model.addFrame(
            pin.Frame('ee', jee, pin.SE3(pin.Quaternion(1.0, 0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0])), pin.FrameType.OP_FRAME)
        )
        self._frame_id = self._model.getFrameId('ee')
        self._frame_name = 'ee'

        self._data = self._model.createData()

        self._lambda = float(dls_lambda)
        self._max_iters = int(max_iters)
        self._tol = float(tol)
        self._gain = float(step_gain)

        self._last_q: Optional[np.ndarray] = None

        self._q_lower = np.array(self._model.lowerPositionLimit)
        self._q_upper = np.array(self._model.upperPositionLimit)

        urdf_path_resolved, package_dirs_resolved = (urdf_path, package_dirs) if urdf_path is not None else _default_urdf_path()
        self._geom_model = pin.buildGeomFromUrdf(self._full_robot.model, urdf_path_resolved, pin.GeometryType.COLLISION, package_dirs_resolved)
        self._geom_data = pin.GeometryData(self._geom_model)

        self._joint_weights = np.ones(self._model.nq, dtype=np.float64)
        self._sqrt_w = casadi.DM(np.sqrt(self._joint_weights))

        self._cmodel = cpin.Model(self._model)
        self._cdata = self._cmodel.createData()
        self._cq = casadi.SX.sym("q", self._model.nq, 1)
        self._cTf = casadi.SX.sym("tf", 4, 4)
        cpin.framesForwardKinematics(self._cmodel, self._cdata, self._cq)
        self._ee_id = self._model.getFrameId("ee")
        self._error_fun = casadi.Function(
            "error",
            [self._cq, self._cTf],
            [casadi.vertcat(cpin.log6(self._cdata.oMf[self._ee_id].inverse() * cpin.SE3(self._cTf)).vector)],
        )

        self._opti = casadi.Opti()
        self._var_q = self._opti.variable(self._model.nq)
        self._param_tf = self._opti.parameter(4, 4)
        self._qref_param = self._opti.parameter(self._model.nq)
        totalcost = casadi.sumsqr(self._error_fun(self._var_q, self._param_tf))
        delta_q = self._var_q - self._qref_param
        regularization = casadi.sumsqr(casadi.diag(self._sqrt_w) @ delta_q)
        self._opti.subject_to(self._opti.bounded(self._q_lower, self._var_q, self._q_upper))
        self._opti.minimize(20 * totalcost + 0.01 * regularization)
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': int(self._max_iters),
                'tol': float(self._tol),
            },
            'print_time': False
        }
        self._opti.solver("ipopt", opts)

        self._init_data = np.zeros(self._model.nq)
        self._history_data = np.zeros(self._model.nq)
        # Output-side protection: error gating and delta limiting (to reduce jitter on solver issues)
        self._err_gate_thresh: float = 0.15  # gate if pose error larger than this
        self._max_delta_per_step: float = 0.15  # max per-joint change per step (rad)

    @staticmethod
    def _homogeneous_from_pose(pos_wxyz: np.ndarray) -> np.ndarray:
        vec = np.asarray(pos_wxyz, dtype=np.float64)
        t = np.eye(4, dtype=np.float64)
        t[0:3, 3] = vec[0:3]
        w, x, y, z = vec[3:7]
        R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        t[0:3, 0:3] = R
        return t

    def _ensure_batch(self, batch_size: int):
        if self._last_q is None or self._last_q.shape[0] != batch_size:
            self._last_q = np.zeros((batch_size, self._model.nq), dtype=np.float64)

    def _ik_single(self, target_T: np.ndarray, warm_q: Optional[np.ndarray] = None) -> tuple[np.ndarray, bool, bool, np.ndarray]:
        if warm_q is not None:
            self._init_data = warm_q.astype(np.float64)
        self._opti.set_initial(self._var_q, self._init_data)
        self._opti.set_value(self._param_tf, target_T)
        try:
            qref = self._init_data if warm_q is None else warm_q
            self._opti.set_value(self._qref_param, qref)
        except Exception:
            self._opti.set_value(self._qref_param, self._init_data)

        try:
            _ = self._opti.solve_limited()
            sol_q = np.asarray(self._opti.value(self._var_q)).reshape(-1)
            if self._init_data is not None:
                max_diff = float(np.max(np.abs(self._history_data - sol_q)))
                self._init_data = sol_q
                if max_diff > 30.0 / 180.0 * np.pi:
                    self._init_data = np.zeros(self._model.nq)
            else:
                self._init_data = sol_q
            self._history_data = sol_q

            v = np.zeros_like(sol_q)
            tau_ff = pin.rnea(self._model, self._data, sol_q, v, np.zeros(self._model.nv))
            is_collision = self.check_self_collision(sol_q)
            return sol_q, True, bool(is_collision), np.asarray(tau_ff)
        except Exception:
            return np.zeros(self._model.nq), False, False, np.zeros(self._model.nv)

    def solve_pose_to_joints(self, targets_pos_wxyz: np.ndarray,
                             warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        B = int(targets_pos_wxyz.shape[0])
        self._ensure_batch(B)

        q_out = np.zeros((B, self._model.nq), dtype=np.float64)
        success = np.zeros((B,), dtype=bool)
        T_targets = [self._homogeneous_from_pose(targets_pos_wxyz[i]) for i in range(B)]
        for i in range(B):
            warm_q = None if warm_start is None else warm_start[i]
            sol_q, converged, _, _ = self._ik_single(T_targets[i], warm_q)

            # Compute pose error norm for gating
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
        try:
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
        """Reset the internal state of the IK solver (warm-start, history, etc)."""
        self._last_q = None
        self._init_data = np.zeros(self._model.nq)
        self._history_data = np.zeros(self._model.nq)


class X7SBimanualIK:
    """Two-arm IK wrapper for X7S.

    - Left arm: joints 5..11, ee on joint11
    - Right arm: joints 14..20, ee on joint20
    """

    def __init__(self, urdf_path: Optional[str] = None, package_dirs: Optional[list[str]] = None,
                 dls_lambda: float = 1e-2, max_iters: int = 50, tol: float = 1e-4):
        left_keep = ["joint5", "joint6", "joint7", "joint8", "joint9", "joint10", "joint11"]
        right_keep = ["joint14", "joint15", "joint16", "joint17", "joint18", "joint19", "joint20"]

        self.left = X7SPinocchioIK(
            urdf_path=urdf_path,
            package_dirs=package_dirs,
            dls_lambda=dls_lambda,
            max_iters=max_iters,
            tol=tol,
            joints_to_keep=left_keep,
            ee_joint_name='joint11',
        )
        self.right = X7SPinocchioIK(
            urdf_path=urdf_path,
            package_dirs=package_dirs,
            dls_lambda=dls_lambda,
            max_iters=max_iters,
            tol=tol,
            joints_to_keep=right_keep,
            ee_joint_name='joint20',
        )

    def solve_pose_to_joints(self,
                             left_targets_pos_wxyz: np.ndarray,
                             right_targets_pos_wxyz: np.ndarray,
                             left_warm_start: Optional[np.ndarray] = None,
                             right_warm_start: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        l_q, l_succ = self.left.solve_pose_to_joints(left_targets_pos_wxyz, warm_start=left_warm_start)
        r_q, r_succ = self.right.solve_pose_to_joints(right_targets_pos_wxyz, warm_start=right_warm_start)
        return l_q, l_succ, r_q, r_succ
