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

"""Configuration for the Piper  robots.

The following configurations are available:

* :obj:`FRANKA_OMRON_CFG`: Piper robot with Panda hand
* :obj:`FRANKA_OMRON_HIGH_PD_CFG`: Piper robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lw_benchhub.data import LW_BENCHHUB_DATA_PATH


##
# Configuration
##

ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "piper.usd"

##
# Configuration
##

PIPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        scale=(1.0, 1.0, 1.0),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "finger_joint_left": 0.035,
            "finger_joint_right": -0.035,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint.*"],
            effort_limit_sim=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


PIPER_HIGH_PD_CFG = PIPER_CFG.copy()
PIPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PIPER_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
PIPER_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
PIPER_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
PIPER_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# Robot configuration dictionary
PIPER_OFFSET_CONFIG = {
    "left_offset": np.array([0, 0.0, 0]),
    "right_offset": np.array([0, 0.0, 0]),
    # "left2arm_transform": np.eye(4),
    "left2arm_transform": np.array([[0.0859, -0.0000, 0.9963, 0.056],
                                    [-0.0000, 1.0000, 0.0000, -0.0000],
                                    [-0.9963, -0.0000, 0.0859, 0.213],
                                    [0.0000, 0.0000, 0.0000, 1.0000]]),

    "right2arm_transform": np.array([[0.0859, -0.0000, 0.9963, 0.056],
                                    [-0.0000, 1.0000, 0.0000, -0.0000],
                                    [-0.9963, -0.0000, 0.0859, 0.213],
                                    [0.0000, 0.0000, 0.0000, 1.0000]]),
    "left2finger_transform": np.eye(4),
    "right2finger_transform": np.eye(4),
    "robot_arm_length": 0.6,
}

########################################################
# Double Piper
########################################################


ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "double_piper.usd"

DOOUBLE_PIPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,

        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
            fix_root_link=True,

        ),
        scale=(1.0, 1.0, 1.0),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "joint1.*": 0.0,
            "joint2.*": 0.0,
            "joint3.*": 0.0,
            "joint4.*": 0.0,
            "joint5.*": 0.0,
            "joint6.*": 0.0,
            "finger_joint_left.*": 0.0,
            "finger_joint_right.*": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-4]_[lr]"],
            effort_limit_sim=87.0,
            velocity_limit=2.175,
            stiffness=10.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["joint[5-7]_[lr]"],
            effort_limit_sim=12.0,
            velocity_limit=2.61,
            stiffness=20.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint.*_l", "finger_joint.*_r"],
            effort_limit=600,
            velocity_limit=0.03,
            stiffness=4e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


DOOUBLE_PIPER_HIGH_PD_CFG = DOOUBLE_PIPER_CFG.copy()
DOOUBLE_PIPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
DOOUBLE_PIPER_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
DOOUBLE_PIPER_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
DOOUBLE_PIPER_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
DOOUBLE_PIPER_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0


"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""

# Robot configuration dictionary
DOOUBLE_PIPER_OFFSET_CONFIG = {
    "left_offset": np.array([0, 0, 0]),
    "right_offset": np.array([0, 0, 0]),
    "left2arm_transform": np.array([[0.0859, -0.0000, 0.9963, 0.056],
                                    [-0.0000, 1.0000, 0.0000, 0.0000],
                                    [-0.9963, -0.0000, 0.0859, 0.213],
                                    [0.0000, 0.0000, 0.0000, 1.0000]]),

    "right2arm_transform": np.array([[0.0859, -0.0000, 0.9963, 0.056],
                                    [-0.0000, 1.0000, 0.0000, -0.0000],
                                    [-0.9963, -0.0000, 0.0859, 0.213],
                                    [0.0000, 0.0000, 0.0000, 1.0000]]),
    "left2finger_transform": np.eye(4),
    "right2finger_transform": np.eye(4),
    "robot_arm_length": 0.6,
}

DOUBLE_PIPER_VIS_HELPER_CFG = {
    "left_hand_line_z": {
        "relative_prim_path": f"piper_L/hand_link_l",
        "translation": (0.0, 0, 0.13),
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
        "relative_prim_path": f"piper_L/hand_link_l",
        "translation": (0.0, 0.0, 0.13),
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
        "relative_prim_path": f"piper_L/hand_link_l",
        "translation": (0.0, 0, 0.13),
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
        "relative_prim_path": f"piper_R/hand_link_r",
        "translation": (0.0, 0, 0.13),
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
        "relative_prim_path": f"piper_R/hand_link_r",
        "translation": (0.0, 0, 0.13),
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
        "relative_prim_path": f"piper_R/hand_link_r",
        "translation": (0.0, 0, 0.13),
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
