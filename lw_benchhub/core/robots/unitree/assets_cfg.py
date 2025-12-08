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

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lw_benchhub.data import LW_BENCHHUB_DATA_PATH

ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "g1_three_fingers.usd"
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            # linear_damping=0.0,
            # angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            sleep_threshold=0.005, stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.8, 0.8),
        joint_pos={
            # ".*_hip_pitch_joint": -0.20,
            # ".*_knee_joint": 0.42,
            # ".*_ankle_pitch_joint": -0.23,
            # ".*_elbow_joint": 0.87,
            # "left_shoulder_roll_joint": 0.16,
            # "left_shoulder_pitch_joint": 0.35,
            # "right_shoulder_roll_joint": -0.16,
            # "right_shoulder_pitch_joint": 0.35,
            ".*_wrist_yaw_joint": 0.0,
            ".*_wrist_pitch_joint": 0.0,
            ".*_wrist_roll_joint": 0.0,
            ".*_shoulder_pitch_joint": 0,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0,
            "left_elbow_joint": 0,
            # "left_1_joint": 1.0,
            # "right_1_joint": -1.0,
            # "left_2_joint": 0.52,
            # "right_2_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_.*"],
            effort_limit_sim=100000,
            velocity_limit_sim=1000,
            stiffness=1e6,
            damping=1e4,
        ),
        # "waist": ImplicitActuatorCfg(
        #     joint_names_expr=["waist_pitch_joint", "waist_yaw_joint","waist_roll_joint"],
        #     effort_limit_sim=50,
        #     velocity_limit_sim=100.0,
        #     stiffness=1e6,
        #     damping=1e4,
        # ),
        # "legs": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_hip_yaw_joint",
        #         ".*_hip_roll_joint",
        #         ".*_hip_pitch_joint",
        #         ".*_knee_joint",
        #     ],
        #     effort_limit_sim=300,
        #     velocity_limit_sim=100.0,
        #     stiffness={
        #         ".*_hip_yaw_joint": 150.0,
        #         ".*_hip_roll_joint": 150.0,
        #         ".*_hip_pitch_joint": 200.0,
        #         ".*_knee_joint": 200.0,
        #     },
        #     damping={
        #         ".*_hip_yaw_joint": 50,
        #         ".*_hip_roll_joint": 50,
        #         ".*_hip_pitch_joint": 50,
        #         ".*_knee_joint": 50,
        #     },
        #     armature={
        #         ".*_hip_.*": 0.01,
        #         ".*_knee_joint": 0.01,
        #     },
        # ),
        # "feet": ImplicitActuatorCfg(
        #     effort_limit_sim=20,
        #     joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        #     stiffness=20.0,
        #     damping=20,
        #     armature=0.01,
        # ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_shoulder_.*",
                "right_wrist_.*",
                "right_elbow_joint",
            ],
            effort_limit_sim=5,
            velocity_limit_sim=3.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "left_arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_shoulder_.*",
                "left_wrist_.*",
                "left_elbow_joint",
            ],
            effort_limit_sim=5,
            velocity_limit_sim=3.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*_0_joint", ".*_1_joint", ".*_2_joint"],
            effort_limit_sim=600,
            velocity_limit_sim=4,
            stiffness=4e3,
            damping=1e2,
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""

G1_HIGH_PD_CFG = G1_CFG.copy()
G1_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True

OFFSET_CONFIG_G1 = {
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
    # "left_offset": np.array([0,0,0]),
    # "right_offset": np.array([0,0,0]),
    # "left2arm_transform": np.array([[ 0.541,  0.001,  0.841,  0.111],
    #                                 [ 0.249,  0.955, -0.161,  0.25 ],
    #                                 [-0.804,  0.296,  0.516, -0.081],
    #                                 [ 0.000,  0.000,  0.000,  1.000]]),
    # "right2arm_transform": np.array([[ 5.407e-01, -7.000e-04,  8.412e-01,  1.110e-01],
    #                                 [-2.486e-01,  9.552e-01,  1.606e-01, -2.501e-01],
    #                                 [-8.036e-01, -2.960e-01,  5.163e-01, -8.100e-02],
    #                                 [ 0.000e+00,  0.000e+00,  0.000e+00,  1.000e+00]]),
    "vuer_head_mat": np.array([[1, 0, 0, 0],
                               [0, 1, 0, 1.1],
                               [0, 0, 1, -0.0],
                               [0, 0, 0, 1]]),
    "vuer_right_wrist_mat": np.array([[1, 0, 0, 0.25],  # -y
                                      [0, 1, 0, 0.7],   # z
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


# """Configuration for the Unitree G1-Loco robot"""
ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "g1_29dof_with_hand.usd"
G1_Loco_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=100.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            sleep_threshold=0.005, stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 1.0, 0.835),
        # rot = (0.707,0.0,0.0,0.707),
        joint_pos={
            ".*_hip_pitch_joint": -0.092,
            ".*_hip_roll_joint": 0.0354,
            ".*_hip_yaw_joint": 0.000,
            ".*_knee_joint": 0.311,
            ".*_ankle_pitch_joint": -0.238,
            ".*_ankle_roll_joint": 0.038,
            "left_shoulder_roll_joint": 0.3,
            "right_shoulder_roll_joint": -0.3,
            ".*_elbow_joint": 1,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_yaw_joint", "waist_roll_joint"],
            effort_limit_sim=50,
            velocity_limit_sim=100.0,
            stiffness=1e6,
            damping=1e4,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=3000,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_joint": 200.0,
                ".*_ankle_pitch_joint": 40,
                ".*_ankle_roll_joint": 40,
            },
            damping={
                ".*_hip_yaw_joint": 2,
                ".*_hip_roll_joint": 2,
                ".*_hip_pitch_joint": 2,
                ".*_knee_joint": 4,
                ".*_ankle_pitch_joint": 2,
                ".*_ankle_roll_joint": 2,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=2000,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=200.0,
            damping=20,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_wrist_yaw_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=20,
            velocity_limit_sim=10.0,
            stiffness=10000.0,
            damping=1000.0,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*_0_joint", ".*_1_joint", ".*_2_joint"],
            effort_limit_sim=1,
            velocity_limit_sim=5,
            stiffness=10000.0,
            damping=1000.0,
        ),
    },
)

"""Configuration for the Unitree G1 robot used in Gear's WBC policy training"""
G1_GEARWBC_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(LW_BENCHHUB_DATA_PATH / "assets" / "g1_29dof_with_hand_rev_1_0_wbc.usd"),  # "omniverse://isaac-dev.ov.nvidia.com/Isaac/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.8, -1.38, 0.78),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # target angles [rad]
            "left_hip_pitch_joint": -0.389,
            "right_hip_pitch_joint": -0.390,
            "waist_yaw_joint": -0.000,
            "left_hip_roll_joint": 0.022,
            "right_hip_roll_joint": -0.016,
            "waist_roll_joint": -0.000,
            "left_hip_yaw_joint": 0.025,
            "right_hip_yaw_joint": -0.025,
            "waist_pitch_joint": 0.049,
            "left_knee_joint": 0.653,
            "right_knee_joint": 0.657,
            "left_shoulder_pitch_joint": 0.027,
            "right_shoulder_pitch_joint": 0.027,
            "left_ankle_pitch_joint": -0.285,
            "right_ankle_pitch_joint": -0.289,
            "left_shoulder_roll_joint": 0.015,
            "right_shoulder_roll_joint": -0.015,
            "left_ankle_roll_joint": -0.007,
            "right_ankle_roll_joint": 0.000,
            "left_shoulder_yaw_joint": 0.035,
            "right_shoulder_yaw_joint": -0.036,
            "left_elbow_joint": 0.088,
            "right_elbow_joint": 0.087,
            "left_wrist_roll_joint": -0.011,
            "right_wrist_roll_joint": 0.011,
            "left_wrist_pitch_joint": 0.054,
            "right_wrist_pitch_joint": 0.053,
            "left_wrist_yaw_joint": -0.035,
            "right_wrist_yaw_joint": 0.035,
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0
        },
        joint_vel={".*": 0.0}
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 278.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 32.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 300.0,  # 100.0,
                ".*_hip_roll_joint": 300.0,  # 100.0,
                ".*_hip_pitch_joint": 300.0,  # 100.0
                ".*_knee_joint": 600.0,  # 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 4.0,  # 2.5,
                ".*_hip_roll_joint": 4.0,  # 2.5,
                ".*_hip_pitch_joint": 4.0,  # 2.5,
                ".*_knee_joint": 8.0,  # 5.0,
            },
            armature={
                ".*_hip_.*": 0.03,
                ".*_knee_joint": 0.03,
            },
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness={
                ".*_ankle_pitch_joint": 40.0,  # 20.0,
                ".*_ankle_roll_joint": 40.0,  # 20.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2,  # 0.2,
                ".*_ankle_roll_joint": 2,  # 0.1,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 50.0,
                ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_ankle_pitch_joint": 37.0,
                ".*_ankle_roll_joint": 37.0,
            },
            armature=0.03,
            friction=0.03
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=[
                "waist_.*_joint",
            ],
            effort_limit={
                "waist_yaw_joint": 88.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
            },
            velocity_limit={
                "waist_yaw_joint": 32.0,
                "waist_roll_joint": 37.0,
                "waist_pitch_joint": 37.0,
            },
            # changing from 400 to 200
            stiffness={
                "waist_yaw_joint": 500.0,  # 300.0,
                "waist_roll_joint": 500.0,  # 300.0,
                "waist_pitch_joint": 500.0,  # 300.0,
            },
            damping={
                "waist_yaw_joint": 5.0,
                "waist_roll_joint": 5.0,
                "waist_pitch_joint": 5.0,
            },
            armature=0.03,
            friction=0.03
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 200.0,
                ".*_shoulder_roll_joint": 200.0,
                ".*_shoulder_yaw_joint": 200.0,
                ".*_elbow_joint": 200.0,
                ".*_wrist_roll_joint": 200.0,
                ".*_wrist_pitch_joint": 20.0,
                ".*_wrist_yaw_joint": 20.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 200.0,  # 90.0,
                ".*_shoulder_roll_joint": 200.0,  # 60.0,
                ".*_shoulder_yaw_joint": 80.0,  # 20.0,
                ".*_elbow_joint": 80.0,  # 60.0,
                ".*_wrist_.*_joint": 40.0,  # 10.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 5.0,  # 2.0,
                ".*_shoulder_roll_joint": 5.0,  # 1.0,
                ".*_shoulder_yaw_joint": 2.0,  # 0.4,
                ".*_elbow_joint": 2.0,  # 1.0,
                ".*_wrist_.*_joint": 2.0,  # 0.2,
            },
            armature={
                ".*_shoulder_.*": 0.03,
                ".*_elbow_.*": 0.03,
                ".*_wrist_.*_joint": 0.03
            },
            friction=0.03
        ),
        # TODO: check with teleop
        "hands": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hand_.*",
            ],
            effort_limit=20.0,
            velocity_limit=2.0,
            stiffness=10.0,
            damping=1,
            armature=0.03,
            friction=0.03
        ),
    },
)
