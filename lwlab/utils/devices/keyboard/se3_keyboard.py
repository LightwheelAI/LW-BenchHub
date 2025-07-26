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

"""Keyboard controller for SE(3) control."""

import torch
import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation
import carb
import omni

from isaaclab.devices.device_base import DeviceBase



class Se3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8, base_sensitivity: float = 0.1, base_yaw_sensitivity: float = 0.1):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.base_sensitivity = base_sensitivity
        self.base_yaw_sensitivity = base_yaw_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._delta_base = np.zeros(2)  # (x, y, z)
        self._delta_base_yaw = np.zeros(1)  # (roll, pitch, yaw)
        self._delta_base_pitch = np.zeros(4)  # ( pitch)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle gripper (open/close): B\n"
        msg += "\tMove arm along x-axis: W/S\n"
        msg += "\tMove arm along y-axis: A/D\n"
        msg += "\tMove arm along z-axis: Q/E\n"
        msg += "\tRotate arm along x-axis: Z/X\n"
        msg += "\tRotate arm along y-axis: T/G\n"
        msg += "\tRotate arm along z-axis: C/V\n"
        msg += "\tMove base along x-axis: I/K\n"
        msg += "\tMove base along y-axis: J/L\n"
        msg += "\tRotate base along z-axis: U/O\n"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._delta_base = np.zeros(2)  # (x, y)
        self._delta_base_yaw = np.zeros(1)  # ( yaw)
        self._delta_base_pitch = np.zeros(4)  # ( pitch)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, bool, np.ndarray]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        arm_control = np.concatenate([self._delta_pos, rot_vec]).astype("float32")
        base_control = np.concatenate([self._delta_base, self._delta_base_yaw, np.array([0])]).astype("float32")
        gripper = torch.zeros(1, 1)
        gripper[:] = -1.0 if self._close_gripper else 1.0
        arm_control = torch.tensor(arm_control).repeat(1, 1)
        base_control = torch.tensor(base_control).repeat(1, 1)
        return torch.concat([arm_control, gripper, base_control], dim=1)
        # return self._delta_base_pitch
    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset()
            if event.input.name == "B":
                self._close_gripper = not self._close_gripper
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot += self._INPUT_KEY_MAPPING[event.input.name]

            elif event.input.name in ["I", "K", "J", "L"]:
                self._delta_base += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["U", "O"]:
                self._delta_base_yaw += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4"]:
                idx = {"NUMPAD_1": 0, "NUMPAD_2": 1, "NUMPAD_3": 2, "NUMPAD_4": 3}
                self._delta_base_pitch[idx[event.input.name]] += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "T", "G", "C", "V"]:
                self._delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["I", "K", "J", "L"]:
                self._delta_base -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["U", "O"]:
                self._delta_base_yaw -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["NUMPAD_5", "NUMPAD_6", "NUMPAD_7", "NUMPAD_8"]:
                idx = {"NUMPAD_5": 0, "NUMPAD_6": 1, "NUMPAD_7": 2, "NUMPAD_8": 3}
                self._delta_base_pitch[idx[event.input.name]] += self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        base_yaw_sensitivity = 0.05
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "B": True,
            # x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # pitch (around y-axis)
            "T": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "G": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            # base
            "I": np.asarray([1.0, 0.0]) * self.base_sensitivity,
            "K": np.asarray([-1.0, 0.0]) * self.base_sensitivity,
            # base
            "J": np.asarray([0.0, 1.0]) * self.base_sensitivity,
            "L": np.asarray([0.0, -1.0]) * self.base_sensitivity,
            # yaw
            "U": np.asarray([1.0]) * self.base_yaw_sensitivity,
            "O": np.asarray([-1.0]) * self.base_yaw_sensitivity,

            # yaw
            "NUMPAD_1": np.asarray([1.0]) * base_yaw_sensitivity,
            "NUMPAD_5": np.asarray([-1.0]) * base_yaw_sensitivity,
            # yaw
            "NUMPAD_2": np.asarray([1.0]) * base_yaw_sensitivity,
            "NUMPAD_6": np.asarray([-1.0]) * base_yaw_sensitivity,
            # yaw
            "NUMPAD_3": np.asarray([1.0]) * base_yaw_sensitivity,
            "NUMPAD_7": np.asarray([-1.0]) * base_yaw_sensitivity,
            # yaw
            "NUMPAD_4": np.asarray([1.0]) * base_yaw_sensitivity,
            "NUMPAD_8": np.asarray([-1.0]) * base_yaw_sensitivity,

        }

KEYCONTROLLER_MAP = {
    'keyboard-pandaomron': Se3Keyboard,
}
