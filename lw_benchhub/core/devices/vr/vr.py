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

"""VR controller for SE(3) control."""

from collections.abc import Callable

import numpy as np
import requests
import torch
from pynput.keyboard import Key, Listener
from termcolor import colored

from isaaclab.devices.device_base import DeviceBase

import lw_benchhub.utils.math_utils.transform_utils.numpy_impl as T
from lw_benchhub.utils.log_utils import get_vr_logger
from lw_benchhub.utils.opentelevision import OpenTeleVision

from . import consts


def mat_update(prev_mat, mat, name="UNK"):
    if np.linalg.det(mat) == 0:
        return prev_mat
    else:
        return mat


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


class Device(DeviceBase):
    def __init__(self, env):
        """
        Args:
            env (RobotEnv): The environment which contains the robot(s) to control
                            using this device.
        """
        self.env = env

        # isaaclab robot
        self.robot = env.scene.articulations['robot']
        self.actuators = self.robot.actuators

        # dummy robot arm
        if hasattr(env.action_manager.cfg, 'arm_action'):
            self.arm_count = 1
            self.all_robot_arms = [['arm']]
        elif hasattr(env.action_manager.cfg, 'arms_action') or \
            (hasattr(env.action_manager.cfg, 'left_arm_action') and
             hasattr(env.action_manager.cfg, 'right_arm_action')):
            self.arm_count = 2
            self.all_robot_arms = [['left_arm', 'right_arm']]
        else:
            raise ValueError("No arm action found in the environment")

        self.active_robot = 0

        self.num_robots = 1
        self._display_controls()

    def refresh_env(self, env):
        self.env = env
        self.robot = env.scene.articulations['robot']
        # dummy robot arm
        if hasattr(env.action_manager.cfg, 'arm_action'):
            self.arm_count = 1
            self.all_robot_arms = [['arm']]
        elif hasattr(env.action_manager.cfg, 'arms_action') or \
            (hasattr(env.action_manager.cfg, 'left_arm_action') and
             hasattr(env.action_manager.cfg, 'right_arm_action')):
            self.arm_count = 2
            self.all_robot_arms = [['left_arm', 'right_arm']]
        else:
            raise ValueError("No arm action found in the environment")
        self._reset_internal_state()

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control
        """
        self.grasp_states = [[False] * self.arm_count for i in range(self.num_robots)]
        self.active_arm_indices = [0] * self.num_robots
        self.base_modes = [False] * self.num_robots
        if hasattr(self.env.cfg.isaaclab_arena_env.embodiment, 'reset_robot_cfg_state'):
            self.env.cfg.isaaclab_arena_env.embodiment.reset_robot_cfg_state()

    def _display_controls(self):
        pass

    @staticmethod
    def print_command(char, info):
        char += " " * (30 - len(char))
        print("{}\t{}".format(char, info))

    @property
    def active_arm(self):
        return self.all_robot_arms[self.active_robot][self.active_arm_index]

    @property
    def grasp(self):
        return self.grasp_states[self.active_robot][self.active_arm_index]

    @property
    def active_arm_index(self):
        return self.active_arm_indices[self.active_robot]

    @property
    def base_mode(self):
        return self.base_modes[self.active_robot]

    @active_arm_index.setter
    def active_arm_index(self, value):
        self.active_arm_indices[self.active_robot] = value

    def get_controller_state(self):
        raise NotImplementedError

    def input2action(self):
        raise NotImplementedError

    def set_checkpoint_frame_idx(self, frame_index):
        raise NotImplementedError

    def _save_action_dict_to_hdf5(self, action: dict) -> None:
        """Save complete action dictionary structure to HDF5 for full replay capability.

        Args:
            action: Action dictionary containing various action components
        """
        try:
            # Save each action component separately to preserve structure
            for key, value in action.items():
                if isinstance(value, torch.Tensor):
                    # Save tensor directly
                    self.env.recorder_manager.add_to_episodes(f"obs/raw_action/{key}", value)
                elif isinstance(value, np.ndarray):
                    # Convert numpy array to tensor and save
                    tensor_value = torch.tensor(value, device=self.env.device, dtype=torch.float32)
                    self.env.recorder_manager.add_to_episodes(f"obs/raw_action/{key}", tensor_value)
                elif isinstance(value, (int, float, bool)):
                    # Convert scalar to tensor and save
                    tensor_value = torch.tensor([value], device=self.env.device, dtype=torch.float32)
                    self.env.recorder_manager.add_to_episodes(f"obs/raw_action/{key}", tensor_value)
                elif isinstance(value, list):
                    # Convert list to tensor and save
                    tensor_value = torch.tensor(value, device=self.env.device, dtype=torch.float32)
                    self.env.recorder_manager.add_to_episodes(f"obs/raw_action/{key}", tensor_value)
                else:
                    # Skip non-serializable values
                    print(f"Warning: Skipping non-serializable action component '{key}' of type {type(value)}")

        except Exception as e:
            print(f"Error saving action dict to HDF5: {e}")

    def _save_raw_input_to_hdf5(self, head_mat, abs_left_wrist_mat, abs_right_wrist_mat,
                                rel_left_wrist_mat, rel_right_wrist_mat,
                                left_controller_state, right_controller_state) -> None:
        """Save raw input data to HDF5 for complete replay capability.

        Args:
            head_mat: Head transformation matrix
            abs_left_wrist_mat: Absolute left wrist transformation matrix
            abs_right_wrist_mat: Absolute right wrist transformation matrix
            rel_left_wrist_mat: Relative left wrist transformation matrix
            rel_right_wrist_mat: Relative right wrist transformation matrix
            left_controller_state: Left controller state dictionary
            right_controller_state: Right controller state dictionary
        """
        try:
            # Save transformation matrices
            self._save_matrix_to_hdf5("obs/raw_input/head_mat", head_mat)
            self._save_matrix_to_hdf5("obs/raw_input/abs_left_wrist_mat", abs_left_wrist_mat)
            self._save_matrix_to_hdf5("obs/raw_input/abs_right_wrist_mat", abs_right_wrist_mat)
            self._save_matrix_to_hdf5("obs/raw_input/rel_left_wrist_mat", rel_left_wrist_mat)
            self._save_matrix_to_hdf5("obs/raw_input/rel_right_wrist_mat", rel_right_wrist_mat)

            # Save controller states
            self._save_controller_state_to_hdf5("obs/raw_input/left_controller_state", left_controller_state)
            self._save_controller_state_to_hdf5("obs/raw_input/right_controller_state", right_controller_state)

            # Save internal state information
            self._save_internal_state_to_hdf5()

        except Exception as e:
            print(f"Error saving raw input to HDF5: {e}")

    def _save_matrix_to_hdf5(self, key_prefix: str, matrix: np.ndarray) -> None:
        """Save a transformation matrix to HDF5.

        Args:
            key_prefix: Key prefix for the matrix data
            matrix: 4x4 transformation matrix
        """
        if matrix is not None:
            tensor_matrix = torch.tensor(matrix, device=self.env.device, dtype=torch.float32).unsqueeze(0)
            if self.env.num_envs > 1:
                tensor_matrix = torch.repeat_interleave(tensor_matrix, self.env.num_envs, dim=0)
            self.env.recorder_manager.add_to_episodes(key_prefix, tensor_matrix)

    def _save_controller_state_to_hdf5(self, key_prefix: str, controller_state: dict) -> None:
        """Save controller state dictionary to HDF5.

        Args:
            key_prefix: Key prefix for the controller state data
            controller_state: Controller state dictionary
        """
        for key, value in controller_state.items():
            if isinstance(value, (int, float, bool)):
                tensor_value = torch.tensor([value], device=self.env.device, dtype=torch.float32).unsqueeze(0)
                if self.env.num_envs > 1:
                    tensor_value = torch.repeat_interleave(tensor_value, self.env.num_envs, dim=0)
                self.env.recorder_manager.add_to_episodes(f"{key_prefix}/{key}", tensor_value)
            elif isinstance(value, (list, tuple)):
                tensor_value = torch.tensor(value, device=self.env.device, dtype=torch.float32).unsqueeze(0)
                if self.env.num_envs > 1:
                    tensor_value = torch.repeat_interleave(tensor_value, self.env.num_envs, dim=0)
                self.env.recorder_manager.add_to_episodes(f"{key_prefix}/{key}", tensor_value)
            else:
                print(f"Warning: Skipping non-serializable controller state '{key}' of type {type(value)}")

    def _save_internal_state_to_hdf5(self) -> None:
        """Save internal VR device state to HDF5."""
        try:
            # Save important internal state variables
            internal_state = {
                "has_started": getattr(self, 'has_started', False),
                "started": getattr(self, 'started', False),
                "base_mode_flag": getattr(self, 'base_mode_flag', 1),
                "last_thumbstick_state": getattr(self, 'last_thumbstick_state', 0),
                "last_x_button_state": getattr(self, 'last_x_button_state', 0),
                "last_y_button_state": getattr(self, 'last_y_button_state', 0),
                "last_start_state": getattr(self, 'last_start_state', False),
                "is_body_moving": getattr(self, 'is_body_moving', False),
                "is_body_moving_last_frame": getattr(self, 'is_body_moving_last_frame', False),
                "has_keep": getattr(self, 'has_keep', False),
                "rollback_keep": getattr(self, 'rollback_keep', False),
                "last_checkpoint_frame_idx": getattr(self, 'last_checkpoint_frame_idx', -1),
            }

            for key, value in internal_state.items():
                if isinstance(value, (int, float, bool)):
                    tensor_value = torch.tensor([value], device=self.env.device, dtype=torch.float32).unsqueeze(0)
                else:
                    tensor_value = torch.tensor(value, device=self.env.device, dtype=torch.float32).unsqueeze(0)
                if self.env.num_envs > 1:
                    tensor_value = torch.repeat_interleave(tensor_value, self.env.num_envs, dim=0)
                self.env.recorder_manager.add_to_episodes(f"obs/raw_input/internal_state/{key}", tensor_value)

        except Exception as e:
            print(f"Error saving internal state to HDF5: {e}")

    @staticmethod
    def load_raw_input_from_hdf5(episode_data, step_index: int) -> dict | None:
        """Load complete raw input data from HDF5 for replay.

        Args:
            episode_data: EpisodeData object containing the recorded data
            step_index: Index of the step to load

        Returns:
            Complete raw input dictionary, or None if loading fails
        """
        try:
            raw_input_dict = {}

            # Check if raw_input data exists
            if "obs" not in episode_data._data or "raw_input" not in episode_data._data["obs"]:
                print("Warning: No raw_input data found in episode")
                return None

            raw_input_data = episode_data._data["obs"]["raw_input"]

            # Load transformation matrices
            matrix_keys = ["head_mat", "abs_left_wrist_mat", "abs_right_wrist_mat",
                           "rel_left_wrist_mat", "rel_right_wrist_mat"]
            for key in matrix_keys:
                if key in raw_input_data and isinstance(raw_input_data[key], torch.Tensor):
                    if step_index < len(raw_input_data[key]):
                        matrix = raw_input_data[key][step_index].squeeze(0).cpu().numpy()
                        raw_input_dict[key] = matrix
                    else:
                        print(f"Warning: Step index {step_index} out of range for matrix '{key}'")

            # Load controller states
            controller_keys = ["left_controller_state", "right_controller_state"]
            for controller_key in controller_keys:
                if controller_key in raw_input_data:
                    controller_state = {}
                    controller_data = raw_input_data[controller_key]
                    for state_key, tensor_data in controller_data.items():
                        if isinstance(tensor_data, torch.Tensor) and step_index < len(tensor_data):
                            value = tensor_data[step_index].squeeze(0)
                            if value.numel() == 1:
                                controller_state[state_key] = value.item()
                            else:
                                controller_state[state_key] = value.cpu().numpy()
                    raw_input_dict[controller_key] = controller_state

            # Load internal state
            if "internal_state" in raw_input_data:
                internal_state = {}
                internal_data = raw_input_data["internal_state"]
                for state_key, tensor_data in internal_data.items():
                    if isinstance(tensor_data, torch.Tensor) and step_index < len(tensor_data):
                        value = tensor_data[step_index].squeeze(0).item()
                        internal_state[state_key] = value
                raw_input_dict["internal_state"] = internal_state

            return raw_input_dict if raw_input_dict else None

        except Exception as e:
            print(f"Error loading raw input from HDF5: {e}")
            return None

    def advance(self):
        action = self.input2action()
        if action is None:
            return self.env.action_manager.action
        if action['reset']:
            return None
        if not action['started']:
            return False
        for key, value in action.items():
            action[key] = torch.tensor(value, device=self.env.device, dtype=torch.float32)
        processed_action = self.env.cfg.isaaclab_arena_env.embodiment.preprocess_device_action(action, self)

        # Repeat action to (num_envs, action_dim)
        for key in action.keys():
            action[key] = action[key].unsqueeze(0)
            if self.env.num_envs > 1:
                action[key] = torch.repeat_interleave(action[key], self.env.num_envs, dim=0)
        if processed_action.shape[0] != self.env.num_envs:
            processed_action = processed_action.repeat(self.env.num_envs, 1)

        # Save raw action data to HDF5 through recorder_manager
        if hasattr(self.env, 'recorder_manager') and len(self.env.recorder_manager.active_terms) > 0:
            # Save complete action dictionary structure for full replay capability
            self._save_action_dict_to_hdf5(action)

        return processed_action


class VRDevice(Device):
    """
    A minimalistic driver class for VR with OpenTeleVision library.

    Args:
        env (RobotEnv): The environment which contains the robot(s) to control
                        using this device.
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """
    tv_device_type: str

    def __init__(
        self,
        env,
        img_shape,
        shm_name,
        relative_control=False,
        robot_mode="arm",
    ):
        super().__init__(env)
        self._additional_callbacks = {}
        self.robot_mode = robot_mode
        self.relative_control = relative_control
        self.tv = OpenTeleVision(img_shape, shm_name, device_type=self.tv_device_type)
        self._init_preprocessor()
        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.started = False
        self.last_start_state = False

        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self._pos_step = 0.05
        self.pos_sensitivity = 1.0
        self.rot_sensitivity = 1.0
        self._cumulative_base = np.array([0, 0, 0, 0], dtype=np.float64)

        # also add a keyboard for aux controls
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

        self.last_thumbstick_state = 0  # record the last thumbstick state
        self.base_mode_flag = 1         # switch mode flag

        self.last_x_button_state = 0
        self.last_y_button_state = 0

    def _init_preprocessor(self):
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 1.1],
                                       [0, 0, 1, -0.0],
                                       [0, 0, 0, 1]])
        self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0],  # -y
                                              [0, 1, 0, 0],  # z
                                              [0, 0, 1, 0],  # -x
                                              [0, 0, 0, 1]])
        self.vuer_left_wrist_mat = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])

        config = self.env.cfg.isaaclab_arena_env.embodiment.offset_config
        self.vuer_head_mat = config.get("vuer_head_mat", self.vuer_head_mat)
        self.vuer_right_wrist_mat = config.get("vuer_right_wrist_mat", self.vuer_right_wrist_mat)
        self.vuer_left_wrist_mat = config.get("vuer_left_wrist_mat", self.vuer_left_wrist_mat)

        self.first_left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.first_right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.last_left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.prev_left_arm_abs = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.last_right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.prev_right_arm_abs = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)

        self.rel_keep_mat_left = np.eye(4)
        self.rel_keep_mat_right = np.eye(4)
        self.rel_checkpoint_keep_mat_left = np.eye(4)
        self.rel_checkpoint_keep_mat_right = np.eye(4)
        self.has_started = False
        self.has_keep = False
        self.rollback_keep = False
        self.is_body_moving = False
        self.is_body_moving_last_frame = True
        self.checkpoint_frame_origin_left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.checkpoint_frame_origin_right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.checkpoint_frame_first_left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.checkpoint_frame_first_right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.last_checkpoint_frame_idx = -1

        self.left_offset = config.get("left_offset", np.array([0, 0, 0]))
        self.right_offset = config.get("right_offset", np.array([0, 0, 0]))
        self.left2arm_transform = config.get("left2arm_transform", np.eye(4))
        self.right2arm_transform = config.get("right2arm_transform", np.eye(4))
        self.left2finger_transform = config.get("left2finger_transform", np.eye(4))
        self.right2finger_transform = config.get("right2finger_transform", np.eye(4))
        self.robot_arm_length = config.get("robot_arm_length", 1.4)

    def get_x_rotation(self, offset, homogeneous=False):
        if not homogeneous:
            return np.array([[1, 0, 0], [0, np.cos(offset), -np.sin(offset)], [0, np.sin(offset), np.cos(offset)]])
        else:
            return np.array([[1, 0, 0, 0], [0, np.cos(offset), -np.sin(offset), 0], [0, np.sin(offset), np.cos(offset), 0], [0, 0, 0, 1]])

    def get_y_rotation(self, offset, homogeneous=False):
        if not homogeneous:
            return np.array([[np.cos(offset), 0, np.sin(offset)], [0, 1, 0], [-np.sin(offset), 0, np.cos(offset)]])
        else:
            return np.array([[np.cos(offset), 0, np.sin(offset), 0], [0, 1, 0, 0], [-np.sin(offset), 0, np.cos(offset), 0], [0, 0, 0, 1]])

    def get_z_rotation(self, offset, homogeneous=False):
        if not homogeneous:
            return np.array([[np.cos(offset), -np.sin(offset), 0], [np.sin(offset), np.cos(offset), 0], [0, 0, 1]])
        else:
            return np.array([[np.cos(offset), -np.sin(offset), 0, 0], [np.sin(offset), np.cos(offset), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # NOTE: Interface to IsaacLab

    def add_callback(self, key: str, func: Callable):

        self._additional_callbacks[key] = func

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        super()._reset_internal_state()

        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        # Reset 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        # Reset control
        self._control = np.zeros(7)

        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self._pos_step = 0.05
        self.pos_sensitivity = 1.0
        self.rot_sensitivity = 1.0

        # Reset grasp
        self.single_click_and_hold = False
        self.started = False
        self.last_start_state = False
        # Reset Mode Switching
        self.last_thumbstick_state = 0
        self.base_mode_flag = 1

        self.last_x_button_state = 0
        self.last_y_button_state = 0
        self.last_checkpoint_frame_idx = -1

    # NOTE: Interface to robosuite
    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def pose2action(self, pose):
        # input: 4x4 SE(3)
        # output: [x, y, z, axisangle-x, axisangle-y, axisangle-z]
        ac = np.zeros(6)
        ac[:3] = pose[:3, 3]
        ac[3:] = T.quat2axisangle(T.mat2quat(pose[:3, :3]))
        return ac

    def pose2action_xyzw(self, pose):
        # input: 4x4 SE(3)
        # output: [x, y, z, quat-x, quat-y, quat-z, quat-w]
        ac = np.zeros(7)
        ac[:3] = pose[:3, 3]  # pos xyz
        quat = T.mat2quat(pose[:3, :3])
        ac[3:] = quat  # xyz
        return ac

    def get_controller_state(self):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, self.tv.head_matrix.copy(), "head")
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, self.tv.left_hand.copy(), "left wrist")
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, self.tv.right_hand.copy(), "right wrist")
        # change of basis
        head_mat = consts.grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(consts.grd_yup2grd_zup)
        left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(consts.grd_yup2grd_zup)
        right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(consts.grd_yup2grd_zup)

        if not self.has_started and not self.started:
            self.first_right_wrist_mat = right_wrist_mat.copy()
            self.first_left_wrist_mat = left_wrist_mat.copy()
            self.last_left_wrist_mat = left_wrist_mat.copy()
            self.last_right_wrist_mat = right_wrist_mat.copy()

        self.has_started = self.started

        origin_left_wrist_mat = left_wrist_mat.copy()
        origin_right_wrist_mat = right_wrist_mat.copy()
        left_wrist_mat[0:3, 3] = left_wrist_mat[0:3, 3] - self.first_left_wrist_mat[0:3, 3]
        right_wrist_mat[0:3, 3] = right_wrist_mat[0:3, 3] - self.first_right_wrist_mat[0:3, 3]
        left_wrist_mat[:3, :3] = (left_wrist_mat @ fast_mat_inv(self.first_left_wrist_mat))[:3, :3]
        right_wrist_mat[:3, :3] = (right_wrist_mat @ fast_mat_inv(self.first_right_wrist_mat))[:3, :3]
        left_wrist_mat = self.adjust_pose(left_wrist_mat, human_arm_length=0.5, robot_arm_length=self.robot_arm_length)
        right_wrist_mat = self.adjust_pose(right_wrist_mat, human_arm_length=0.5, robot_arm_length=self.robot_arm_length)
        abs_left_wrist_mat = left_wrist_mat @ self.left2arm_transform
        abs_right_wrist_mat = right_wrist_mat @ self.right2arm_transform
        abs_left_wrist_mat[0:3, 3] += self.left_offset
        abs_right_wrist_mat[0:3, 3] += self.right_offset

        if not self.has_started and not self.started:
            self.last_left_wrist_mat = abs_left_wrist_mat.copy()
            self.last_right_wrist_mat = abs_right_wrist_mat.copy()
            self.prev_left_arm_abs = abs_left_wrist_mat.copy()
            self.prev_right_arm_abs = abs_right_wrist_mat.copy()

        self.has_started = self.started

        return head_mat, origin_left_wrist_mat, origin_right_wrist_mat, abs_left_wrist_mat, abs_right_wrist_mat

    def get_controller_state_hamer(self):
        try:
            response = requests.get("http://127.0.0.1:5001/results", timeout=0.1)
            if response.ok:
                response_json = response.json()
                if "message" not in response_json:
                    return response_json.get("lh_bones_kps"), response_json.get("rh_bones_kps"), \
                        response_json.get("lh_global_orient"), response_json.get("rh_global_orient")
        except requests.exceptions.RequestException as e:
            return None, None, None, None
        return None, None, None, None

    def adjust_pose(self, pose, human_arm_length=0.5, robot_arm_length=1.4, ratios: list[float] = [1.0, 0.6, 1.0]):
        scale_factor = robot_arm_length / human_arm_length
        robot_pose = pose.copy()
        robot_pose[:3, 3] *= scale_factor
        robot_pose[:3, 3] *= ratios
        return robot_pose

    @property
    def control(self):
        """
        Grabs current pose of Spacemouse

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    def on_press(self, key):
        """
        Key handler for key presses.
        Args:
            key (str): key that was pressed
        """
        try:
            # controls for moving position
            if key == Key.up:
                self.pos[0] += self._pos_step * self.pos_sensitivity  # dec x
            elif key == Key.down:
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # inc x
            elif key == Key.left:
                self.pos[1] += self._pos_step * self.pos_sensitivity  # dec y
            elif key == Key.right:
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # inc y
            elif key.char == "p":
                drot = T.rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity
            elif key.char == "o":
                drot = T.rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            elif key.char == "n":
                self.base_mode_flag = 1 - self.base_mode_flag
        except AttributeError as e:
            pass

    def on_release(self, key):
        """
        Key handler for key releases.
        Args:
            key (str): key that was pressed
        """
        try:
            # controls for mobile base (only applicable if mobile base present)
            if key.char == "b":
                self.started = True
                self._reset_state = 0
                if "B" in self._additional_callbacks:
                    self._additional_callbacks["B"]()
            elif key.char == "s":
                self.active_arm_index = (self.active_arm_index + 1) % len(self.all_robot_arms[self.active_robot])
            elif key.char == "=":
                self.active_robot = (self.active_robot + 1) % self.num_robots
            elif key.char == "r":
                self.started = False
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                self._cumulative_base = np.array([0, 0, 0, 0], dtype=np.float64)
                self._additional_callbacks["R"]()
            elif key.char == "m":
                if "M" in self._additional_callbacks:
                    self._additional_callbacks["M"]()
            elif key.char == "n":
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
            elif key.char == "t":
                if "T" in self._additional_callbacks:
                    self._additional_callbacks["T"]()
            elif key.char == "x":
                self.started = False
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                self._cumulative_base = np.array([0, 0, 0, 0], dtype=np.float64)
                if "X" in self._additional_callbacks:
                    self._additional_callbacks["X"]()
            elif key.char == "y":
                self.started = False
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                self._cumulative_base = np.array([0, 0, 0, 0], dtype=np.float64)
                if "Y" in self._additional_callbacks:
                    self._additional_callbacks["Y"]()
            elif key.char == "u":
                self.started = False
                self._reset_state = 1
                self._enabled = False
                self._reset_internal_state()
                self._cumulative_base = np.array([0, 0, 0, 0], dtype=np.float64)
                if "U" in self._additional_callbacks:
                    self._additional_callbacks["U"]()
        except AttributeError as e:
            pass

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 50
        dpos = dpos * 125

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation

    def is_empty_input(self, action_dict):
        return not action_dict["started"]

    def close(self):
        self.listener.stop()
        self.listener.join()
        self.tv.close()

    def set_checkpoint_frame_idx(self, frame_index):
        self.last_checkpoint_frame_idx = frame_index


class VRController(VRDevice):
    tv_device_type = "controller"

    def get_controller_state(self):
        head_mat, origin_left_wrist_mat, origin_right_wrist_mat, abs_left_wrist_mat, abs_right_wrist_mat = super().get_controller_state()

        if self.has_started and not self.has_keep and self.tv.right_controller_state["a_button"]:
            get_vr_logger().info("before a button, abs_right_wrist_mat:\n%s", abs_right_wrist_mat)
            self.before_a_button_abs_right_wrist_mat = abs_right_wrist_mat.copy()
            self.before_a_button_abs_left_wrist_mat = abs_left_wrist_mat.copy()
            self.rel_keep_mat_left[:3, 3] = self.first_left_wrist_mat[:3, 3] - origin_left_wrist_mat[:3, 3]
            self.rel_keep_mat_left[:3, :3] = (self.first_left_wrist_mat @ fast_mat_inv(origin_left_wrist_mat))[:3, :3]
            self.rel_keep_mat_right[:3, 3] = self.first_right_wrist_mat[:3, 3] - origin_right_wrist_mat[:3, 3]
            self.rel_keep_mat_right[:3, :3] = (self.first_right_wrist_mat @ fast_mat_inv(origin_right_wrist_mat))[:3, :3]
        if self.has_started and self.has_keep and not self.tv.right_controller_state["a_button"]:
            self.first_left_wrist_mat[:3, 3] = origin_left_wrist_mat[:3, 3] + self.rel_keep_mat_left[:3, 3]
            self.first_left_wrist_mat[:3, :3] = self.rel_keep_mat_left[:3, :3] @ origin_left_wrist_mat[:3, :3]
            self.first_right_wrist_mat[:3, 3] = origin_right_wrist_mat[:3, 3] + self.rel_keep_mat_right[:3, 3]
            self.first_right_wrist_mat[:3, :3] = self.rel_keep_mat_right[:3, :3] @ origin_right_wrist_mat[:3, :3]
            left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(consts.grd_yup2grd_zup)
            right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(consts.grd_yup2grd_zup)
            left_wrist_mat[0:3, 3] = origin_left_wrist_mat[0:3, 3] - self.first_left_wrist_mat[0:3, 3]
            right_wrist_mat[0:3, 3] = origin_right_wrist_mat[0:3, 3] - self.first_right_wrist_mat[0:3, 3]
            left_wrist_mat[:3, :3] = (left_wrist_mat @ fast_mat_inv(self.first_left_wrist_mat))[:3, :3]
            right_wrist_mat[:3, :3] = (origin_right_wrist_mat @ fast_mat_inv(self.first_right_wrist_mat))[:3, :3]
            left_wrist_mat = self.adjust_pose(left_wrist_mat, human_arm_length=0.5, robot_arm_length=self.robot_arm_length)
            right_wrist_mat = self.adjust_pose(right_wrist_mat, human_arm_length=0.5, robot_arm_length=self.robot_arm_length)
            abs_left_wrist_mat = left_wrist_mat @ self.left2arm_transform
            abs_right_wrist_mat = right_wrist_mat @ self.right2arm_transform
            abs_left_wrist_mat[0:3, 3] += self.left_offset
            abs_right_wrist_mat[0:3, 3] += self.right_offset
            get_vr_logger().info("new abs_right_wrist_mat:\n%s", abs_right_wrist_mat)
        self.has_keep = self.tv.right_controller_state["a_button"]

        if self.has_started and not self.rollback_keep and self.tv.left_controller_state["b_button"]:
            get_vr_logger().info("before rollback, abs_right_wrist_mat:\n%s", abs_right_wrist_mat)
            self.rel_checkpoint_keep_mat_left[:3, 3] = self.checkpoint_frame_first_left_wrist_mat[:3, 3] - self.checkpoint_frame_origin_left_wrist_mat[:3, 3]
            self.rel_checkpoint_keep_mat_left[:3, :3] = (self.checkpoint_frame_first_left_wrist_mat @ fast_mat_inv(self.checkpoint_frame_origin_left_wrist_mat))[:3, :3]
            self.rel_checkpoint_keep_mat_right[:3, 3] = self.checkpoint_frame_first_right_wrist_mat[:3, 3] - self.checkpoint_frame_origin_right_wrist_mat[:3, 3]
            self.rel_checkpoint_keep_mat_right[:3, :3] = (self.checkpoint_frame_first_right_wrist_mat @ fast_mat_inv(self.checkpoint_frame_origin_right_wrist_mat))[:3, :3]
        if self.has_started and self.rollback_keep and not self.tv.left_controller_state["b_button"]:
            self.first_left_wrist_mat[:3, 3] = origin_left_wrist_mat[:3, 3] + self.rel_checkpoint_keep_mat_left[:3, 3]
            self.first_left_wrist_mat[:3, :3] = self.rel_checkpoint_keep_mat_left[:3, :3] @ origin_left_wrist_mat[:3, :3]
            self.first_right_wrist_mat[:3, 3] = origin_right_wrist_mat[:3, 3] + self.rel_checkpoint_keep_mat_right[:3, 3]
            self.first_right_wrist_mat[:3, :3] = self.rel_checkpoint_keep_mat_right[:3, :3] @ origin_right_wrist_mat[:3, :3]
            left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(consts.grd_yup2grd_zup)
            right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(consts.grd_yup2grd_zup)
            left_wrist_mat[0:3, 3] = origin_left_wrist_mat[0:3, 3] - self.first_left_wrist_mat[0:3, 3]
            right_wrist_mat[0:3, 3] = origin_right_wrist_mat[0:3, 3] - self.first_right_wrist_mat[0:3, 3]
            left_wrist_mat[:3, :3] = (left_wrist_mat @ fast_mat_inv(self.first_left_wrist_mat))[:3, :3]
            right_wrist_mat[:3, :3] = (origin_right_wrist_mat @ fast_mat_inv(self.first_right_wrist_mat))[:3, :3]
            left_wrist_mat = self.adjust_pose(left_wrist_mat, human_arm_length=0.5, robot_arm_length=self.robot_arm_length)
            right_wrist_mat = self.adjust_pose(right_wrist_mat, human_arm_length=0.5, robot_arm_length=self.robot_arm_length)
            abs_left_wrist_mat = left_wrist_mat @ self.left2arm_transform
            abs_right_wrist_mat = right_wrist_mat @ self.right2arm_transform
            abs_left_wrist_mat[0:3, 3] += self.left_offset
            abs_right_wrist_mat[0:3, 3] += self.right_offset
            get_vr_logger().info("new rollback abs_right_wrist_mat:\n%s", abs_right_wrist_mat)
        self.rollback_keep = self.tv.left_controller_state["b_button"]

        if not self.relative_control:
            rel_use_left_wrist_mat = origin_left_wrist_mat.copy()
            rel_use_right_wrist_mat = origin_right_wrist_mat.copy()
        else:
            rel_use_left_wrist_mat = abs_left_wrist_mat.copy()
            rel_use_right_wrist_mat = abs_right_wrist_mat.copy()
        rel_left_wrist_mat = np.eye(4)
        rel_left_wrist_mat[:3, :3] = (rel_use_left_wrist_mat[:3, :3] @ np.linalg.inv(self.last_left_wrist_mat[:3, :3]))
        rel_left_wrist_mat[:3, 3] = (rel_use_left_wrist_mat[:3, 3] - self.last_left_wrist_mat[:3, 3])
        self.last_left_wrist_mat = rel_use_left_wrist_mat.copy()
        rel_right_wrist_mat = np.eye(4)
        rel_right_wrist_mat[:3, :3] = (rel_use_right_wrist_mat[:3, :3] @ np.linalg.inv(self.last_right_wrist_mat[:3, :3]))
        rel_right_wrist_mat[:3, 3] = (rel_use_right_wrist_mat[:3, 3] - self.last_right_wrist_mat[:3, 3])
        self.last_right_wrist_mat = rel_use_right_wrist_mat.copy()

        return head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, self.tv.left_controller_state, self.tv.right_controller_state

    def input2action(self):
        state = {}
        reset = state["reset"] = bool(self._reset_state)
        if reset:
            self._reset_state = False
            return state

        head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()

        # Save raw input data to HDF5 for complete replay capability
        # Only record when started is True
        if hasattr(self.env, 'recorder_manager') and len(self.env.recorder_manager.active_terms) > 0:
            if self.started:
                self._save_raw_input_to_hdf5(head_mat, abs_left_wrist_mat, abs_right_wrist_mat,
                                             rel_left_wrist_mat, rel_right_wrist_mat,
                                             left_controller_state, right_controller_state)
        if right_controller_state["a_button"]:
            if not self.relative_control:
                if self.is_body_moving_last_frame:
                    self.is_body_moving = False
                    joint_vel = self.env.scene.articulations["robot"].data.joint_vel
                    for i in range(joint_vel.shape[1]):
                        get_vr_logger().info("joint %s vel: %s", self.env.scene.articulations["robot"].joint_names[i], joint_vel[0, i])
                        if abs(joint_vel[0, i]) > 0.01:  # Check joint velocity threshold
                            self.is_body_moving = True
                            get_vr_logger().info("joint %s has non zero vel: %s", self.env.scene.articulations["robot"].joint_names[i], joint_vel[0, i])
                        else:
                            # Use write_joint_velocity_to_sim to set individual joint velocity to 0
                            # For individual joint, we need to pass a tensor with shape (1, 1) for the specific joint
                            zero_velocity_single = torch.zeros((1, 1), device=self.env.device)
                            self.env.scene.articulations["robot"].write_joint_velocity_to_sim(zero_velocity_single, joint_ids=[i])
                            get_vr_logger().info("joint %s set zero vel using write_joint_velocity_to_sim: %s", self.env.scene.articulations["robot"].joint_names[i], joint_vel[0, i])
                self.is_body_moving_last_frame = self.is_body_moving

                if self.is_body_moving:
                    get_vr_logger().info("body is moving, using before a button abs right wrist mat")
                    abs_right_wrist_mat = self.before_a_button_abs_right_wrist_mat.copy()
                    abs_left_wrist_mat = self.before_a_button_abs_left_wrist_mat.copy()
                else:
                    # if robot body is not moving, wait for a button release in the loop
                    while True:
                        head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()
                        if not right_controller_state["a_button"]:
                            break
            else:
                while True:
                    head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()
                    if not right_controller_state["a_button"]:
                        self.prev_left_arm_abs = abs_left_wrist_mat.copy()
                        self.prev_right_arm_abs = abs_right_wrist_mat.copy()
                        break
        else:
            self.is_body_moving_last_frame = True
            self.is_body_moving = False

        state["lpose_abs"] = abs_left_wrist_mat
        state["rpose_abs"] = abs_right_wrist_mat
        state["prev_lpose_abs"] = self.prev_left_arm_abs.copy()
        state["prev_rpose_abs"] = self.prev_right_arm_abs.copy()
        state["lpose_delta"] = rel_left_wrist_mat
        state["rpose_delta"] = rel_right_wrist_mat
        # TODO: get gripper and base from self.tv.right_controller
        state["lgrasp"] = left_controller_state["trigger"] * 2 - 1.0
        state["rgrasp"] = right_controller_state["trigger"] * 2 - 1.0
        state["lbase"] = np.array([left_controller_state["thumbstick_x"], left_controller_state["thumbstick_y"]])
        state["rbase"] = np.array([right_controller_state["thumbstick_x"], right_controller_state["thumbstick_y"]])
        state["lsqueeze"] = left_controller_state["squeeze"]
        state["rsqueeze"] = right_controller_state["squeeze"]
        state["rbase_button"] = right_controller_state["thumbstick"]
        state["lbase_button"] = left_controller_state["thumbstick"]

        state["x_button"] = left_controller_state["b_button"]
        state["y_button"] = left_controller_state["a_button"]

        if left_controller_state["thumbstick"] == 1 and self.last_thumbstick_state == 0:
            self.base_mode_flag = 1 - self.base_mode_flag
            print(f"base_mode_flag switched to: {self.base_mode_flag}")
        self.last_thumbstick_state = left_controller_state["thumbstick"]
        state["base_mode"] = self.base_mode_flag

        if left_controller_state["b_button"] == 1 and self.last_x_button_state == 0:
            print(colored("x button pressed", "blue"))
            state["x_button_pressed"] = True
        else:
            state["x_button_pressed"] = False
        self.last_x_button_state = left_controller_state["b_button"]

        if left_controller_state["a_button"] == 1 and self.last_y_button_state == 0:
            print(colored("y button pressed", "blue"))
            state["y_button_pressed"] = True
        else:
            state["y_button_pressed"] = False
        self.last_y_button_state = left_controller_state["a_button"]

        b_pressed = right_controller_state["b_button"]
        if not b_pressed and self.last_start_state:
            state["reset"] = self.started
            self.started = not self.started
        if state["reset"]:
            return None
        self.last_start_state = b_pressed

        state["started"] = self.started
        if self.arm_count == 1:
            state[f"{self.active_arm}_abs"] = self.pose2action_xyzw(state["rpose_abs"])
            state[f"{self.active_arm}_delta"] = self.pose2action(state["rpose_delta"])
            state[f"{self.active_arm}_gripper"] = state["rgrasp"]
        else:
            state["left_arm_abs"] = self.pose2action_xyzw(state["lpose_abs"])
            state["right_arm_abs"] = self.pose2action_xyzw(state["rpose_abs"])

            # Convert 4x4 matrices to 6D vectors (position + axis-angle)
            state["left_arm_delta"] = self.pose2action(state['lpose_delta'])
            state["right_arm_delta"] = self.pose2action(state['rpose_delta'])
            state["prev_left_arm_abs_val"] = self.pose2action(state["prev_lpose_abs"])
            state["prev_right_arm_abs_val"] = self.pose2action(state["prev_rpose_abs"])
            self.prev_left_arm_abs = state["lpose_abs"].copy()
            self.prev_right_arm_abs = state["rpose_abs"].copy()

            state["left_gripper"] = state["lgrasp"]
            state["right_gripper"] = state["rgrasp"]

        if state.get("x_button_pressed", False):
            if "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()

        if state.get("y_button_pressed", False):
            self.checkpoint_frame_origin_left_wrist_mat = self.last_left_wrist_mat.copy()
            self.checkpoint_frame_origin_right_wrist_mat = self.last_right_wrist_mat.copy()
            self.checkpoint_frame_first_left_wrist_mat = self.first_left_wrist_mat.copy()
            self.checkpoint_frame_first_right_wrist_mat = self.first_right_wrist_mat.copy()
            if "M" in self._additional_callbacks:
                self._additional_callbacks["M"]()

        return state

    def get_rollback_action(self):
        if self.last_checkpoint_frame_idx == -1:
            return None

        episode_data = self.env.recorder_manager._episodes[0]
        if len(episode_data._data['actions']) <= self.last_checkpoint_frame_idx:
            saved_action = None
        else:
            action = episode_data._data['actions'][self.last_checkpoint_frame_idx]
            saved_action = action.reshape(self.env.num_envs, -1)

        head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()
        if left_controller_state["b_button"]:
            if self.is_body_moving_last_frame:
                self.is_body_moving = False
                joint_vel = self.env.scene.articulations["robot"].data.joint_vel
                for i in range(joint_vel.shape[1]):
                    get_vr_logger().info("joint %s vel: %s", self.env.scene.articulations["robot"].joint_names[i], joint_vel[0, i])
                    if abs(joint_vel[0, i]) > 0.01:  # Check joint velocity threshold
                        self.is_body_moving = True
                        get_vr_logger().info("joint %s has non zero vel: %s", self.env.scene.articulations["robot"].joint_names[i], joint_vel[0, i])
            self.is_body_moving_last_frame = self.is_body_moving

            if self.is_body_moving:
                return saved_action
            else:
                # if robot body is not moving, wait for a button release in the loop
                while True:
                    head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()
                    if not right_controller_state["a_button"]:
                        break
        else:
            self.is_body_moving_last_frame = True
            self.is_body_moving = False

    def saved_input_to_action(self, raw_input):
        state = {}

        # Check if we have internal state to determine if recording was started
        if "internal_state" in raw_input:
            internal_state = raw_input["internal_state"]
            started = internal_state.get("started", False)
            if not started:
                print("Warning: Processing raw_input but recording was not started")
                return None

        head_mat = raw_input['head_mat']
        abs_left_wrist_mat = raw_input['abs_left_wrist_mat']
        abs_right_wrist_mat = raw_input['abs_right_wrist_mat']
        rel_left_wrist_mat = raw_input['rel_left_wrist_mat']
        rel_right_wrist_mat = raw_input['rel_right_wrist_mat']
        left_controller_state = raw_input['left_controller_state']
        right_controller_state = raw_input['right_controller_state']

        if right_controller_state["a_button"]:
            if self.is_body_moving_last_frame:
                self.is_body_moving = False
                body_lin_vel_w = self.env.scene.articulations["robot"].data.body_lin_vel_w
                for i in range(body_lin_vel_w.shape[1]):
                    if abs(np.linalg.norm(body_lin_vel_w[0, i, :])) > 0.00001:  # TODO: 0.00001 is a magic number, need to be tuned, 0.0001 is tested to be unsuitable
                        self.is_body_moving = True
                        get_vr_logger().info("body %s has non zero vel: %s", self.env.scene.articulations["robot"].body_names[i], body_lin_vel_w[0, i, :])
            self.is_body_moving_last_frame = self.is_body_moving

            if self.is_body_moving:
                abs_right_wrist_mat = self.before_a_button_abs_right_wrist_mat.copy()
                abs_left_wrist_mat = self.before_a_button_abs_left_wrist_mat.copy()
            else:
                # if robot body is not moving, wait for a button release in the loop
                while True:
                    head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()
                    if not right_controller_state["a_button"]:
                        break
        else:
            self.is_body_moving_last_frame = True
            self.is_body_moving = False

        state["lpose_abs"] = abs_left_wrist_mat
        state["rpose_abs"] = abs_right_wrist_mat
        state["lpose_delta"] = rel_left_wrist_mat
        state["rpose_delta"] = rel_right_wrist_mat
        # TODO: get gripper and base from self.tv.right_controller
        state["lgrasp"] = left_controller_state["trigger"] * 2 - 1.0
        state["rgrasp"] = right_controller_state["trigger"] * 2 - 1.0
        state["lbase"] = np.array([left_controller_state["thumbstick_x"], left_controller_state["thumbstick_y"]])
        state["rbase"] = np.array([right_controller_state["thumbstick_x"], right_controller_state["thumbstick_y"]])
        state["lsqueeze"] = left_controller_state["squeeze"]
        state["rsqueeze"] = right_controller_state["squeeze"]
        state["rbase_button"] = right_controller_state["thumbstick"]
        state["lbase_button"] = left_controller_state["thumbstick"]

        state["x_button"] = left_controller_state["b_button"]
        state["y_button"] = left_controller_state["a_button"]

        if left_controller_state["thumbstick"] == 1 and self.last_thumbstick_state == 0:
            self.base_mode_flag = 1 - self.base_mode_flag
            print(f"base_mode_flag switched to: {self.base_mode_flag}")
        self.last_thumbstick_state = left_controller_state["thumbstick"]
        state["base_mode"] = self.base_mode_flag

        if left_controller_state["b_button"] == 1 and self.last_x_button_state == 0:
            print(colored("x button pressed", "blue"))
            state["x_button_pressed"] = True
        else:
            state["x_button_pressed"] = False
        self.last_x_button_state = left_controller_state["b_button"]

        if left_controller_state["a_button"] == 1 and self.last_y_button_state == 0:
            print(colored("y button pressed", "blue"))
            state["y_button_pressed"] = True
        else:
            state["y_button_pressed"] = False
        self.last_y_button_state = left_controller_state["a_button"]

        b_pressed = right_controller_state["b_button"]
        if not b_pressed and self.last_start_state:
            state["reset"] = self.started
            self.started = not self.started
        if state["reset"]:
            return None
        self.last_start_state = b_pressed

        state["started"] = self.started
        if self.arm_count == 1:
            state[f"{self.active_arm}_abs"] = self.pose2action_xyzw(state["rpose_abs"])
            state[f"{self.active_arm}_delta"] = self.pose2action(state["rpose_delta"])
            state[f"{self.active_arm}_gripper"] = state["rgrasp"]
        else:
            state["left_arm_abs"] = self.pose2action_xyzw(state["lpose_abs"])
            state["left_arm_delta"] = self.pose2action(state["lpose_delta"])
            state["right_arm_abs"] = self.pose2action_xyzw(state["rpose_abs"])
            state["right_arm_delta"] = self.pose2action(state["rpose_delta"])
            state["left_gripper"] = state["lgrasp"]
            state["right_gripper"] = state["rgrasp"]

        if state.get("x_button_pressed", False):
            if "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()

        if state.get("y_button_pressed", False):
            self.checkpoint_frame_origin_left_wrist_mat = self.last_left_wrist_mat.copy()
            self.checkpoint_frame_origin_right_wrist_mat = self.last_right_wrist_mat.copy()
            self.checkpoint_frame_first_left_wrist_mat = self.first_left_wrist_mat.copy()
            self.checkpoint_frame_first_right_wrist_mat = self.first_right_wrist_mat.copy()
            if "M" in self._additional_callbacks:
                self._additional_callbacks["M"]()

        return state

    def _display_controls(self):
        """
        Method to pretty print controls.
        """
        super()._display_controls()
        print("Use the right controller to control the robot.")
        self.print_command("Control", "Command")
        self.print_command("B button", "reset simulation")
        self.print_command("Trigger (hold)", "close gripper")
        self.print_command("X button (left controller)", "rollback to checkpoint (N key)")
        self.print_command("Y button (left controller)", "save checkpoint (M key)")
        self.print_command("Move controller position/rotation", "move arm to bring gripper to corresponding position/rotation")
        self.print_command("Control+C", "quit")
        self.print_command("move Thumbstick up/down", "move the base forward/backward")
        self.print_command("move Thumbstick left/right", "rotate the base left/right")
        self.print_command("move Thumbstick left/right with thumbstick pressed", "move the base left/right")
        self.print_command("s", "switch active arm (if multi-armed robot)")
        self.print_command("=", "switch active robot (if multi-robot environment)")
        print("")

    # NOTE: Interface to IsaacLab
    def reset(self):
        self._reset_internal_state()
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(4)  # (roll, pitch, yaw)
        self._close_gripper = False


class VRHand(VRDevice):
    tv_device_type = "hand"

    def _display_controls(self):
        """
        Method to pretty print controls.
        """
        super()._display_controls()
        print("Use the hand to control the robot.")
        self.print_command("Control", "Command")
        self.print_command("Control+C", "quit")
        self.print_command("b", "start the simulation")
        self.print_command("s", "switch active arm (if multi-armed robot)")
        self.print_command("=", "switch active robot (if multi-robot environment)")
        self.print_command("r", "reset simulation")
        print("")

    def get_controller_state(self):
        head_mat, origin_left_wrist_mat, origin_right_wrist_mat, abs_left_wrist_mat, abs_right_wrist_mat = super().get_controller_state()

        if not self.has_started:
            return head_mat, abs_left_wrist_mat, abs_right_wrist_mat, np.zeros((5, 3)), np.zeros((5, 3))

        # homogeneous
        left_fingers = np.concatenate([self.tv.left_landmarks.copy().T, np.ones((1, self.tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([self.tv.right_landmarks.copy().T, np.ones((1, self.tv.right_landmarks.shape[0]))])

        # change of basis
        left_fingers = consts.grd_yup2grd_zup @ left_fingers
        right_fingers = consts.grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(origin_left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(origin_right_wrist_mat) @ right_fingers
        rel_left_fingers = (consts.hand2inspire_l_finger.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (consts.hand2inspire_r_finger.T @ rel_right_fingers)[0:3, :].T
        vr_left = rel_left_fingers[consts.tip_indices]
        vr_right = rel_right_fingers[consts.tip_indices]
        left_fingers, right_fingers = vr_left, vr_right

        left_fingers = left_fingers if left_fingers is not None else np.zeros((5, 3))
        right_fingers = right_fingers if right_fingers is not None else np.zeros((5, 3))

        return head_mat, abs_left_wrist_mat, abs_right_wrist_mat, left_fingers, right_fingers

    def input2action(self):
        state = {}
        reset = state["reset"] = bool(self._reset_state)
        if reset:
            self._reset_state = False
            return state

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )
        self.last_drotation = np.array(self.raw_drotation)
        head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers = self.get_controller_state()
        state['dpos'] = dpos
        state['raw_drotation'] = raw_drotation

        state["head"] = head_mat
        state["lpose"] = rel_left_wrist_mat
        state["rpose"] = rel_right_wrist_mat
        state["lhand"] = rel_left_fingers
        state["rhand"] = rel_right_fingers
        state["base_mode"] = 1  # TODO
        state["started"] = self.started
        # robot = self.env.robots[self.active_robot]
        drotation = raw_drotation[[1, 0, 2]]
        dpos, drotation = self._postprocess_device_outputs(dpos, drotation)
        state["left_arm_abs"] = self.pose2action_xyzw(state["lpose"])
        state["left_arm_delta"] = self.pose2action(state["lpose"])
        state["left_finger_tips"] = rel_left_fingers
        state["right_arm_abs"] = self.pose2action_xyzw(state["rpose"])
        state["right_arm_delta"] = self.pose2action(state["rpose"])
        state["right_finger_tips"] = rel_right_fingers
        state["base_loco_mode"] = self.base_mode_flag
        # if robot.is_mobile:
        base_mode = bool(state["base_mode"])
        if base_mode is True:
            arm_norm_delta = np.zeros(6)
            base_ac = np.array([dpos[0], dpos[1], drotation[2]])
            torso_ac = np.array([dpos[2]])
        else:
            arm_norm_delta = np.concatenate([dpos, drotation])
            base_ac = np.zeros(3)
            torso_ac = np.zeros(1)

        state["base"] = base_ac
        state["torso"] = torso_ac
        state["base_mode"] = np.array([1 if base_mode is True else -1])
        return state

    def _postprocess_device_outputs(self, dpos, drotation):
        drotation = drotation * 1.5
        dpos = dpos * 75

        dpos = np.clip(dpos, -1, 1)
        drotation = np.clip(drotation, -1, 1)

        return dpos, drotation

    def reset(self):
        self._reset_internal_state()
        self._left_delta = np.zeros(6)
        self._right_delta = np.zeros(6)
