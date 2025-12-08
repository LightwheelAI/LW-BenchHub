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

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_OMRON_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_OMRON_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lw_benchhub.data import LW_BENCHHUB_DATA_PATH

ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "omron_franka.usd"

##
# Configuration
##

FRANKA_OMRON_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.00005,  # follow isaacsim 5.0.0 tutorial 7 setting
            stabilization_threshold=0.00001,
        ),
        scale=(1.0, 1.0, 1.0),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, -0.5, 0.05),
        rot=(0.707, 0, 0, 0.707),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,

            # base
            "mobilebase_forward": 0.0,
            "mobilebase_side": 0.0,
            "mobilebase_yaw": 0.0,
            "mobilebase_torso_height": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
        "mobile_base_movement": ImplicitActuatorCfg(
            joint_names_expr=["mobilebase_forward", "mobilebase_side", ],
            effort_limit_sim=100000,
            velocity_limit=0.1,
            stiffness=1e6,
            damping=1e1,
        ),
        "mobile_base_rotate": ImplicitActuatorCfg(
            joint_names_expr=["mobilebase_yaw"],
            effort_limit_sim=100000,
            velocity_limit=0.5,
            stiffness=1e6,
            damping=1e1,
        ),
        "mobile_base_torso": ImplicitActuatorCfg(
            joint_names_expr=["mobilebase_torso_height"],
            effort_limit_sim=100000000,
            velocity_limit=0.1,
            stiffness=1e6,
            damping=1e1,
        ),

    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_OMRON_HIGH_PD_CFG = FRANKA_OMRON_CFG.copy()
FRANKA_OMRON_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_OMRON_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_movement"].stiffness = 1e6
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_movement"].damping = 1e4
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_torso"].stiffness = 1e6
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_torso"].damping = 1e4
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_rotate"].stiffness = 1e6
FRANKA_OMRON_HIGH_PD_CFG.actuators["mobile_base_rotate"].damping = 1e4

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# Robot configuration dictionary
OFFSET_CONFIG = {
    "left_offset": np.array([0.3, 0.0, 1]),
    "right_offset": np.array([0.3, 0.0, 1]),
    "left2arm_transform": np.eye(4),
    "right2arm_transform": np.array([[1, 0, 0, 0],
                                     [0, -1, 0, 0],
                                     [0, 0, -1, 0],
                                     [0, 0, 0, 1]]),
    "left2finger_transform": np.eye(4),
    "right2finger_transform": np.eye(4),
    "robot_arm_length": 0.6,
}


ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "panda_2.usd"

##
# Configuration
##

DOUBLE_PANDA_CFG = ArticulationCfg(
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
            fix_root_link=True,
            sleep_threshold=0.00005,  # follow isaacsim 5.0.0 tutorial 7 setting
            stabilization_threshold=0.00001,
        ),
        scale=(1.0, 1.0, 1.0),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0.0, 0.0),
        # rot = (0.707, 0, 0, 0.707),
        joint_pos={
            "panda_[L,R]_joint1": 0.0,
            "panda_[L,R]_joint2": -0.569,
            "panda_[L,R]_joint3": 0.0,
            "panda_[L,R]_joint4": -2.810,
            "panda_[L,R]_joint5": 0.0,
            "panda_[L,R]_joint6": 3.037,
            "panda_[L,R]_joint7": 0.741,
            "panda_[L,R]_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_[L,R]_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_[L,R]_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_[L,R]_finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


DOUBLE_PANDA_HIGH_PD_CFG = DOUBLE_PANDA_CFG.copy()
DOUBLE_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
DOUBLE_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
DOUBLE_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
DOUBLE_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
DOUBLE_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# Robot configuration dictionary
DOUBLE_PANDA_OFFSET_CONFIG = {
    "left_offset": np.array([0.389, 0.2, 0.4658]),
    "right_offset": np.array([0.389, -0.2, 0.4658]),
    "left2arm_transform": np.array([[0.7071, 0, 0.7071, 0],
                                    [0, -1, 0, 0],
                                    [0.7071, 0, -0.7071, 0],
                                    [0, 0, 0, 1]]),
    "right2arm_transform": np.array([[0.7071, 0, 0.7071, 0],
                                     [0, -1, 0, 0],
                                     [0.7071, 0, -0.7071, 0],
                                     [0, 0, 0, 1]]),
    "left2finger_transform": np.eye(4),
    "right2finger_transform": np.eye(4),
    "robot_arm_length": 0.6,
}


VIS_HELPER_CFG = {
    "gripper_line": {
        "relative_prim_path": f"omron_v2/Franka/panda_hand",
        "translation": (-0.00111, 0.00182, 0.0998),
        "orientation": (0.01147, 0.50623, 0.8623, -0.00619),
        "color": (0.0, 0.7, 1.0),
        "spawn": sim_utils.CylinderCfg(
            radius=0.004,
            height=0.6,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.7, 1.0)
            ),
            physics_material=None,
            rigid_props=None,
        )
    }
}
