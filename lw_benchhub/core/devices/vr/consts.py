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

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

R = np.array([[0, 0, -1, 0],
              [0, 1, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 1]])

hand2inspire_l_arm = np.array([[0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1]]) @ R

hand2inspire_r_arm = np.array([[0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1]]) @ R

hand2inspire_l_finger = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])

hand2inspire_r_finger = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])

controller2gripper_l_arm = np.array([[1, 0, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 0, 1]])

controller2gripper_l_arm = np.array([                 # pitch up 45°
    [0.7071068, 0, -0.7071068, 0],
    [0, 1, 0, 0],
    [0.7071068, 0, 0.7071068, 0],
    [0, 0, 0, 1]])  @ controller2gripper_l_arm

controller2gripper_r_arm = np.array([                 # reset to gipper downward
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]])

controller2gripper_r_arm = np.array([                 # pitch up 10°
    [0.9848078, 0, -0.1736482, 0],
    [0, 1, 0, 0],
    [0.1736482, 0, 0.9848078, 0],
    [0, 0, 0, 1]])  @ controller2gripper_r_arm

left_rotation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # swap x and y axes
right_rotation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # swap x and y axes, invert z axis

tip_indices = [4, 9, 14, 19, 24]  # tip of thumb, index, middle, ring, pinky
tip_indices_mano = [4, 8, 12, 16, 20]  # tip of thumb, index, middle, ring, pinky
