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

import isaaclab.utils.math as math_utils
from isaaclab_arena.embodiments.g1.g1 import G1WBCJointEmbodiment, G1WBCPinkEmbodiment
from isaaclab_arena.utils.pose import Pose

import lw_benchhub.utils.math_utils.transform_utils.torch_impl as T
from lw_benchhub.core.robots.unitree.g1 import G1_GEARWBC_CFG
from lw_benchhub.core.robots.robot_arena_base import LwEmbodimentBase


class G1BaseEmbodiment(LwEmbodimentBase):

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.init_robot_base_height = 0.75
        self.init_waist_yaw = 0.0
        self.init_waist_pitch = 0.0
        self.init_waist_roll = 0.0
        # self.sim.dt = 1 / 200  # physics frequency: 100Hz
        # self.decimation = 4  # action frequency: 50Hz

    def set_default_offset_config(self):
        self.offset_config = {
            "left_offset": np.array([0.2, 0.16, 0.09523]),
            "right_offset": np.array([0.2, -0.16, 0.09523]),
            "left2arm_transform": np.array([[1.0, 0.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0, 0.0],
                                            [0.0, 0.0, 1.0, 0.0],
                                            [0.0, 0.0, 0.0, 1.0]]),
            "right2arm_transform": np.array([[1.0, 0.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0, 0.0],
                                             [0.0, 0.0, 1.0, 0.0],
                                             [0.0, 0.0, 0.0, 1.0]]),
            "vuer_head_mat": np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 1.1],
                                       [0, 0, 1, -0.0],
                                       [0, 0, 0, 1]]),
            "vuer_right_wrist_mat": np.array([[1, 0, 0, 0.25],  # -y
                                              [0, 1, 0, 0.7],  # z
                                              [0, 0, 1, -0.3],  # -x
                                              [0, 0, 0, 1]]),
            "vuer_left_wrist_mat": np.array([[1, 0, 0, -0.25],
                                             [0, 1, 0, 0.7],
                                             [0, 0, 1, -0.3],
                                             [0, 0, 0, 1]]),
            "left2finger_transform": np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
            "right2finger_transform": np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
            "robot_arm_length": 0.7
        }

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        """
        Base action [lin_x_local, lin_y_local, ang_z_local, base_height_cmd, torso_orientation_roll_cmd, torso_orientation_pitch_cmd, torso_orientation_yaw_cmd]
        range of -1, 1 each
        """
        base_action = torch.zeros(7,)
        # default as standing
        base_action[3] = self.init_robot_base_height
        base_action[4:] = torch.tensor([self.init_waist_roll, self.init_waist_pitch, self.init_waist_yaw], device=action['lbase'].device)
        # Left squeeze released: Moving-base related cmds
        # left joystick up/down to control pitch movement of the base
        base_action[5] = self.init_waist_pitch + 0.02 * torch.tensor([action['lbase'][0]], device=action['lbase'].device)
        base_action[5] = torch.clamp(base_action[5], min=-0.5, max=0.5)
        self.init_waist_pitch = base_action[5]

        if action['lsqueeze'] <= 0.5:
            if action['rsqueeze'] > 0.5:
                # right joystick up/down to control linear x, left/right to control turning yaw
                base_action[:3] = torch.tensor([action['rbase'][0], 0, action['rbase'][1]], device=action['rbase'].device)
            else:
                # right joystick up/down to control linear x, left/right to control linear y
                base_action[:3] = torch.tensor([action['rbase'][0], action['rbase'][1], 0,], device=action['rbase'].device)
            # left joystick left/right to control yaw, up/down to control pitch movement of the base
            # base_action[4:] = torch.tensor([0, action['lbase'][0], action['lbase'][1]], device=action['lbase'].device)
            # left joystick left/right to control yaw movement of the base
            base_action[6] = self.init_waist_yaw + 0.05 * torch.tensor([action['lbase'][1]], device=action['lbase'].device)
            base_action[6] = torch.clamp(base_action[6], min=-2, max=2)
            self.init_waist_yaw = base_action[6]
        # Left squeeze pressed
        else:
            # left joystick left/right to control roll, up/down to control pitch movement of the base
            # base_action[4:] = torch.tensor([-1 * action['lbase'][1], action['lbase'][0], 0], device=action['lbase'].device)
            # left joystick left/right to control roll movement of the base
            base_action[4] = self.init_waist_roll + 0.02 * torch.tensor([-action['lbase'][1]], device=action['lbase'].device)
            base_action[4] = torch.clamp(base_action[4], min=-0.5, max=0.5)
            self.init_waist_roll = base_action[4]

            # Left squeeze pressed and Right squeeze pressed: right joystick up/down to control base_height_cmd (relative to current height)
            if action['rsqueeze'] > 0.5:
                # joystaick returns -1 to 1, remap to a smaller range
                base_action[3] = self.init_robot_base_height + 0.01 * torch.tensor([action['rbase'][0]], device=action['rbase'].device)
                # Clip base_action[3] to be within 0.5 to 1.0
                base_action[3] = torch.clamp(base_action[3], min=0.2, max=0.75)
                self.init_robot_base_height = base_action[3]

        _cumulative_base, base_quat = math_utils.subtract_frame_transforms(device.robot.data.root_com_pos_w[0],
                                                                           device.robot.data.root_com_quat_w[0],
                                                                           device.robot.data.body_link_pos_w[0, 3],
                                                                           device.robot.data.body_link_quat_w[0, 3])
        # base_quat = T.quat_multiply(device.robot.data.body_link_quat_w[0,3][[1,2,3,0]],device.robot.data.root_com_quat_w[0][[1,2,3,0]], )
        base_yaw = T.quat2axisangle(base_quat[[1, 2, 3, 0]])[2]
        base_quat = T.axisangle2quat(torch.tensor([0, 0, base_yaw], device=device.env.device))
        base_movement = torch.tensor([_cumulative_base[0], _cumulative_base[1], 0], device=device.env.device)

        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        rot_mat_2d = torch.tensor([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ], device=device.env.device)

        # Decoupled WBC does not use x/y/angular in global frame, use local frame
        # robot_x = base_action[0]
        # robot_y = base_action[1]
        # local_xy = torch.tensor([robot_x, robot_y], device=device.env.device)
        # world_xy = torch.matmul(rot_mat_2d, local_xy)
        # base_action[0] = world_xy[0]
        # base_action[1] = world_xy[1]

        left_arm_action = None
        right_arm_action = None

        for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
            pose_quat = abs_arm[3:7]
            combined_quat = T.quat_multiply(base_quat, pose_quat)
            arm_action = abs_arm.clone()
            rot_mat = T.quat2mat(base_quat)
            gripper_movement = torch.matmul(rot_mat, arm_action[:3])
            pose_movement = base_movement + gripper_movement
            arm_action[:3] = pose_movement
            arm_action[3] = combined_quat[3]  # Update quaternion from xyzw to wxyz
            arm_action[4:7] = combined_quat[:3]
            if arm_idx == 0:
                left_arm_action = arm_action  # Robot frame
            else:
                right_arm_action = arm_action  # Robot frame
        # range [-1, 1]
        left_gripper = torch.tensor([action["left_gripper"]], device=action['rbase'].device)
        right_gripper = torch.tensor([action["right_gripper"]], device=action['rbase'].device)
        # convert to range [0, 1]
        left_gripper = -(left_gripper + 1) / 2
        right_gripper = -(right_gripper + 1) / 2
        return torch.concat([left_gripper, right_gripper, left_arm_action, right_arm_action, base_action]).unsqueeze(0)


class G1ArenaJointEmbodiment(G1WBCJointEmbodiment, G1BaseEmbodiment):
    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config.robot = G1_GEARWBC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config.left_arm_action = None
        self.action_config.right_arm_action = None


class G1ArenaPinkEmbodiment(G1WBCPinkEmbodiment, G1BaseEmbodiment):
    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config.robot = G1_GEARWBC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config.left_arm_action = None
        self.action_config.right_arm_action = None
