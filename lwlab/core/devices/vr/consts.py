import numpy as np

grd_yup2grd_zup = np.array([[0, 0, -1, 0],
                            [-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])

# hand2inspire_l_arm = np.array([[1, 0, 0, 0],
#                                [0, 0, -1, 0],
#                                [0, 1, 0, 0],
#                                [0, 0, 0, 1]])

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


# hand2inspire_r_arm = np.array([[1, 0, 0, 0],
#                                [0, 0, 1, 0],
#                                [0, -1, 0, 0],
#                                [0, 0, 0, 1]])


# hand2inspire_r_arm = np.array([[1, 0, 0, 0],
#                                [0, 0, 1, 0],
#                                [0, -1, 0, 0],
#                                [0, 0, 0, 1]])

hand2inspire_l_finger = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])

hand2inspire_r_finger = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 0, 1]])


controller2gripper_l_arm = np.array([[1, 0, 0, 0],  # TODO
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


# controller2gripper_r_arm = np.array([[0.0000000,  1.0000000,  0.0000000, 0],
#                                [0.7071068,  0.0000000, -0.7071068, 0],
#                                [-0.7071068,  0.0000000, -0.7071068, 0],
#                                [0., 0., 0., 1.]])


controller2gripper_r_arm = np.array([                 # pitch up 10°
    [0.9848078, 0, -0.1736482, 0],
    [0, 1, 0, 0],
    [0.1736482, 0, 0.9848078, 0],
    [0, 0, 0, 1]])  @ controller2gripper_r_arm

left_rotation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])  # x和y轴互换
right_rotation_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])  # x和y轴互换，z轴取反

tip_indices = [4, 9, 14, 19, 24]  # tip of thumb, index, middle, ring, pinky
tip_indices_mano = [4, 8, 12, 16, 20]  # tip of thumb, index, middle, ring, pinky
