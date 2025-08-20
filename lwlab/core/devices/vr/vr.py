"""VR controller for SE(3) control."""
import threading
import time
import yaml
import numpy as np
from pynput.keyboard import Controller, Key, Listener
from collections.abc import Callable

import torch
# NOTE: from robosuit
from . import transform_utils as T
from . import consts
from isaaclab.devices.device_base import DeviceBase
import copy
import requests
from lwlab.utils.opentelevision import OpenTeleVision


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


# NOTE: from robosuit
class Device(DeviceBase):
    def __init__(self, env):
        """
        Args:
            env (RobotEnv): The environment which contains the robot(s) to control
                            using this device.
        """
        self.env = env

        # NOTE: isaaclab robot
        self.robot = env.scene.articulations['robot']
        self.actuators = self.robot.actuators

        # NOTE: dummy robot arm
        if hasattr(env.action_manager.cfg, 'arm_action'):
            self.arm_count = 1
            self.all_robot_arms = [['arm']]
        elif hasattr(env.action_manager.cfg, 'left_arm_action') and hasattr(env.action_manager.cfg, 'right_arm_action'):
            self.arm_count = 2
            self.all_robot_arms = [['left_arm', 'right_arm']]
        else:
            raise ValueError("No arm action found in the environment")

        self.active_robot = 0

        self.num_robots = 1  # TODO: 多机器人
        self._display_controls()

    def _reset_internal_state(self):
        """
        Resets internal state related to robot control
        """
        self.grasp_states = [[False] * self.arm_count for i in range(self.num_robots)]
        self.active_arm_indices = [0] * self.num_robots
        self.base_modes = [False] * self.num_robots
        if hasattr(self.env.cfg, 'reset_robot_cfg_state'):
            self.env.cfg.reset_robot_cfg_state()

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

    def advance(self):
        action = self.input2action()
        if action is None:
            return self.env.action_manager.action
        if action['reset']:
            return None
        if not action['started']:
            return False
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                action[key] = torch.tensor(value, device=self.env.device, dtype=torch.float32)
        return self.env.cfg.preprocess_device_action(action, self)


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
        robot_mode="arm",  # or "humanoid" # TODO
    ):
        super().__init__(env)
        # TODO: init OpenTeleVision
        self._additional_callbacks = {}
        self.robot_mode = robot_mode
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
        # launch a new listener thread to listen to SpaceMouse
        # self.thread = threading.Thread(target=self.run)
        # self.thread.daemon = True
        # self.thread.start()

        # also add a keyboard for aux controls
        self.listener = Listener(on_press=self.on_press, on_release=self.on_release)

        # start listening
        self.listener.start()

        self.last_thumbstick_state = 0  # 记录上一次thumbstick的状态
        self.base_mode_flag = 1       # 切换的模式变量

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

        config = self.env.cfg.offset_config
        self.vuer_head_mat = config.get("vuer_head_mat", self.vuer_head_mat)
        self.vuer_right_wrist_mat = config.get("vuer_right_wrist_mat", self.vuer_right_wrist_mat)
        self.vuer_left_wrist_mat = config.get("vuer_left_wrist_mat", self.vuer_left_wrist_mat)

        self.first_left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.first_right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.last_left_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_left_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)
        self.last_right_wrist_mat = consts.grd_yup2grd_zup @ self.vuer_right_wrist_mat.copy() @ fast_mat_inv(consts.grd_yup2grd_zup)

        self.rel_keep_mat_left = np.eye(4)
        self.rel_keep_mat_right = np.eye(4)
        self.has_started = False
        self.has_keep = False

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
    # 绕y轴旋转

    def get_y_rotation(self, offset, homogeneous=False):
        if not homogeneous:
            return np.array([[np.cos(offset), 0, np.sin(offset)], [0, 1, 0], [-np.sin(offset), 0, np.cos(offset)]])
        else:
            return np.array([[np.cos(offset), 0, np.sin(offset), 0], [0, 1, 0, 0], [-np.sin(offset), 0, np.cos(offset), 0], [0, 0, 0, 1]])

    # 绕z轴旋转
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
        self.last_thumbstick_state = 0  # 记录上一次thumbstick的状态
        self.base_mode_flag = 1       # 切换的模式变量

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
        ac[:3] = pose[:3, 3]  # 位置 xyz
        quat = T.mat2quat(pose[:3, :3])  # 旋转四元数
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

        if not self.has_started and self.started:
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


class VRController(VRDevice):
    tv_device_type = "controller"

    def get_controller_state(self):
        head_mat, origin_left_wrist_mat, origin_right_wrist_mat, abs_left_wrist_mat, abs_right_wrist_mat = super().get_controller_state()

        if self.has_started and not self.has_keep and (self.tv.right_controller_state["a_button"] or self.tv.left_controller_state["a_button"]):
            self.rel_keep_mat_left[:3, 3] = self.first_left_wrist_mat[:3, 3] - origin_left_wrist_mat[:3, 3]
            self.rel_keep_mat_left[:3, :3] = (self.first_left_wrist_mat @ fast_mat_inv(origin_left_wrist_mat))[:3, :3]
            self.rel_keep_mat_right[:3, 3] = self.first_right_wrist_mat[:3, 3] - origin_right_wrist_mat[:3, 3]
            self.rel_keep_mat_right[:3, :3] = (self.first_right_wrist_mat @ fast_mat_inv(origin_right_wrist_mat))[:3, :3]
        if self.has_started and self.has_keep and not (self.tv.right_controller_state["a_button"] or self.tv.left_controller_state["a_button"]):
            self.first_left_wrist_mat[:3, 3] = origin_left_wrist_mat[:3, 3] + self.rel_keep_mat_left[:3, 3]
            self.first_left_wrist_mat[:3, :3] = self.rel_keep_mat_left[:3, :3] @ origin_left_wrist_mat[:3, :3]
            self.first_right_wrist_mat[:3, 3] = origin_right_wrist_mat[:3, 3] + self.rel_keep_mat_right[:3, 3]
            self.first_right_wrist_mat[:3, :3] = self.rel_keep_mat_right[:3, :3] @ origin_right_wrist_mat[:3, :3]
        self.has_keep = self.tv.right_controller_state["a_button"] or self.tv.left_controller_state["a_button"]

        rel_left_wrist_mat = np.eye(4)
        rel_left_wrist_mat[:3, :3] = (origin_left_wrist_mat[:3, :3] @ np.linalg.inv(self.last_left_wrist_mat[:3, :3]))
        rel_left_wrist_mat[:3, 3] = (origin_left_wrist_mat[:3, 3] - self.last_left_wrist_mat[:3, 3])
        self.last_left_wrist_mat = origin_left_wrist_mat.copy()
        rel_right_wrist_mat = np.eye(4)
        rel_right_wrist_mat[:3, :3] = (origin_right_wrist_mat[:3, :3] @ np.linalg.inv(self.last_right_wrist_mat[:3, :3]))
        rel_right_wrist_mat[:3, 3] = (origin_right_wrist_mat[:3, 3] - self.last_right_wrist_mat[:3, 3])
        self.last_right_wrist_mat = origin_right_wrist_mat.copy()
        return head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, self.tv.left_controller_state, self.tv.right_controller_state

    def input2action(self):
        state = {}
        reset = state["reset"] = bool(self._reset_state)
        if reset:
            self._reset_state = False
            return state
        while True:
            head_mat, abs_left_wrist_mat, abs_right_wrist_mat, rel_left_wrist_mat, rel_right_wrist_mat, left_controller_state, right_controller_state = self.get_controller_state()
            if not (right_controller_state["a_button"] or left_controller_state["a_button"]):
                break
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

        if left_controller_state["thumbstick"] == 1 and self.last_thumbstick_state == 0:
            self.base_mode_flag = 1 - self.base_mode_flag
            print(f"base_mode_flag 切换为: {self.base_mode_flag}")
        self.last_thumbstick_state = left_controller_state["thumbstick"]
        state["base_mode"] = self.base_mode_flag

        b_pressed = right_controller_state["b_button"] or left_controller_state["b_button"]
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
            state[f"left_arm_abs"] = self.pose2action_xyzw(state["lpose_abs"])
            state[f"left_arm_delta"] = self.pose2action(state["lpose_delta"])
            state[f"right_arm_abs"] = self.pose2action_xyzw(state["rpose_abs"])
            state[f"right_arm_delta"] = self.pose2action(state["rpose_delta"])
            state[f"left_gripper"] = state["lgrasp"]
            state[f"right_gripper"] = state["rgrasp"]

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
        # return head_mat, abs_left_wrist_mat, abs_right_wrist_mat, vr_left, vr_right
        # TODO use hamer to get the pose of the hand
        # mano_left, mano_right, left_rotate, right_rotate = self.get_controller_state_hamer()
        mano_left, mano_right = None, None

        if mano_left is not None:
            mano_left, left_rotate = np.array(mano_left), np.array(left_rotate)
            mano_left = mano_left[consts.tip_indices_mano] - mano_left[0]
            mano_left = mano_left @ left_rotate @ consts.left_rotation_matrix
        if mano_right is not None:
            mano_right, right_rotate = np.array(mano_right), np.array(right_rotate)
            mano_right = mano_right[consts.tip_indices_mano] - mano_right[0]
            mano_right = mano_right @ right_rotate @ consts.right_rotation_matrix

        # TODO fuse vr hand and hamer hand
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
