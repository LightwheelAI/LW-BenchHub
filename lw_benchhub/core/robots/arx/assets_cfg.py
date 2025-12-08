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

"""Configuration for the X7 robot.

The following configurations are available:

* :obj:`X7_CFG`: X7 robot with basic configuration
* :obj:`X7_HIGH_PD_CFG`: X7 robot with stiffer PD control

Reference: X7 robot configuration
"""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lw_benchhub.data import LW_BENCHHUB_DATA_PATH
ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "x7s.usd"


##
# Configuration
##

X7_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        scale=(1.0, 1.0, 1.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "body_z_joint": 0.5,
            "body_y_joint": 0.0,
            "base.*": 0.0,
            "left_shoulder.*": 0.0,
            "right_shoulder.*": 0.0,
            "left_wrist.*": 0.0,
            "right_wrist.*": 0.0,
            "left_elbow.*": 0.0,
            "right_elbow.*": 0.0,
            "left_gripper.*": 0.044,
            "right_gripper.*": 0.0,
        },
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_.*"],
            effort_limit_sim=100000,
            velocity_limit_sim=1000,
            stiffness=1e6,
            damping=1e4,
        ),
        "body": ImplicitActuatorCfg(
            joint_names_expr=["body.*"],
            effort_limit=100000.0,
            velocity_limit=1000.0,
            stiffness=1e6,
            damping=1e4,
        ),

        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_shoulder.*", "left_shoulder.*"],
            effort_limit=87,
            velocity_limit=2.175,
            stiffness=400,
            damping=80,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["right_elbow.*", "left_elbow.*"],
            effort_limit=12,
            velocity_limit=2.61,
            stiffness=400,
            damping=80,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["right_wrist.*", "left_wrist.*"],
            effort_limit=12,
            velocity_limit=2.61,
            stiffness=400,
            damping=80,
        ),

        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["left_gripper.*", "right_gripper.*"],
            effort_limit=600,
            velocity_limit=0.03,
            stiffness=3000,
            damping=500,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of X7 robot."""


X7_HIGH_PD_CFG = X7_CFG.copy()


"""Configuration of X7 robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

OFFSET_CONFIG = {

    "left_offset": np.array([0.3725, 0.1508, 0.263]),
    "right_offset": np.array([0.3725, -0.1508, 0.263]),
    "left2arm_transform": np.array([[1, 0, 0, 0.],
                                    [0, 1, 0, 0.],
                                    [0, 0, 1, 0.],
                                    [0, 0, 0, 1]]),
    "right2arm_transform": np.array([[1, 0, 0, 0.],
                                     [0, 1, 0, -0.],
                                     [0, 0, 1, 0.],
                                     [0, 0, 0, 1]]),
    "left2finger_transform": np.eye(4),
    "right2finger_transform": np.eye(4),
    "robot_arm_length": 0.5,
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
}


VIS_HELPER_CFG = {
    "left_hand_line_z": {
        "relative_prim_path": f"left_hand_link",
        "translation": (0.18, 0, 0.025),
        "orientation": (1, 0, 0, 0),
        "color": (0.0, 0.7, 1.0),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.3,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.7, 1.0)
            ),
            physics_material=None,
            rigid_props=None,
        )
    },
    "left_hand_line_x": {
        "relative_prim_path": f"left_hand_link",
        "translation": (0.18, 0, 0.025),
        "orientation": (1, 0, 0, 0),
        "color": (1.0, 0.3, 0.3),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.3,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.3)
            ),
            physics_material=None,
            rigid_props=None,
        )
    },
    "left_hand_line_y": {
        "relative_prim_path": f"left_hand_link",
        "translation": (0.18, 0, 0.025),
        "orientation": (1, 0, 0, 0),
        "color": (1.0, 0.3, 0.3),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.3,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.3)
            ),
            physics_material=None,
            rigid_props=None,
        )
    },
    "right_hand_line_z": {
        "relative_prim_path": f"right_hand_link",
        "translation": (0.18, 0, 0.025),
        "orientation": (1, 0, 0, 0),
        "color": (0.0, 0.7, 1.0),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.3,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.7, 1.0)
            ),
            physics_material=None,
            rigid_props=None,
        )
    },
    "right_hand_line_x": {
        "relative_prim_path": f"right_hand_link",
        "translation": (0.18, 0, 0.025),
        "orientation": (1, 0, 0, 0),
        "color": (1.0, 0.3, 0.3),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.3,
            axis="X",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.3)
            ),
            physics_material=None,
            rigid_props=None,
        )
    },
    "right_hand_line_y": {
        "relative_prim_path": f"right_hand_link",
        "translation": (0.18, 0, 0.025),
        "orientation": (1, 0, 0, 0),
        "color": (1.0, 0.3, 0.3),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.3,
            axis="Y",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.3)
            ),
            physics_material=None,
            rigid_props=None,
        )
    }
}
