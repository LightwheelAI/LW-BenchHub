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

"""Configuration for the SO101 Follower Robot.

The following configurations are available:

* :obj:`SO101_FOLLOWER_CFG`: SO101 Follower robot configuration

"""

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from lw_benchhub.data import LW_BENCHHUB_DATA_PATH


"""Configuration for the LERobot."""
SO101_FOLLOWER_ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "so101_follower.usd"

SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
            sleep_threshold=0.00005,  # follow isaacsim 5.0.0 tutorial 7 setting
            stabilization_threshold=0.00001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.20, 0.40, 0.71),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": np.pi / 2,
            "wrist_roll": np.pi / 2,
            "gripper": 0.87,
        }
    ),
    actuators={
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll"],
            effort_limit_sim=100,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=100.0,
            velocity_limit_sim=1.0,
            stiffness=1000,
            damping=100,
        ),
    },
    soft_joint_pos_limit_factor=1,
)

SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}

"""Configuration for the LERobot."""
SO100_FOLLOWER_ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "so100_follower.usd"

SO100_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO100_FOLLOWER_ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
            sleep_threshold=0.00005,  # follow isaacsim 5.0.0 tutorial 7 setting
            stabilization_threshold=0.00001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.20, 0.40, 0.71),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": np.pi / 2,
            "wrist_roll": np.pi / 2,
            "gripper": 0.0
        }
    ),
    actuators={
        "shoulder_pan": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan"],
            effort_limit_sim=100,
            # velocity_limit_sim=1.0,
            stiffness=1500,
            damping=100,
        ),
        "shoulder_lift": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_lift"],
            effort_limit_sim=100,
            # velocity_limit_sim=1.0,
            stiffness=1500,
            damping=100,
        ),
        "elbow_flex": ImplicitActuatorCfg(
            joint_names_expr=["elbow_flex"],
            effort_limit_sim=100,
            # velocity_limit_sim=1.0,
            stiffness=1500,
            damping=100,
        ),
        "wrist_flex": ImplicitActuatorCfg(
            joint_names_expr=["wrist_flex"],
            effort_limit_sim=100,
            # velocity_limit_sim=1.0,
            stiffness=1500,
            damping=100,
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["wrist_roll"],
            effort_limit_sim=100,
            # velocity_limit_sim=1.0,
            stiffness=1500,
            damping=100,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=100.0,
            # velocity_limit_sim=1.0,
            stiffness=1500,
            damping=100,
        ),
    },
    soft_joint_pos_limit_factor=1,
)

SO100_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-114.0, 114.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-90.0, 90.0),
    "wrist_flex": (-103.0, 103.0),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (-63, 63.0),
}

BI_SO101_FOLLOWER_ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "bi_so101_follower.usd"

BI_SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(BI_SO101_FOLLOWER_ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "left_shoulder_pan": 0.0,
            "left_shoulder_lift": 0.0,
            "left_elbow_flex": 0.0,
            "left_wrist_flex": 0.0,
            "left_wrist_roll": 0.0,
            "left_gripper": 0.0,
            "right_shoulder_pan": 0.0,
            "right_shoulder_lift": 0.0,
            "right_elbow_flex": 0.0,
            "right_wrist_flex": 0.0,
            "right_wrist_roll": 0.0,
            "right_gripper": 0.0
        }
    ),
    actuators={
        "sts3215": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=10,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)


"""Configuration for the LERobot."""
SO101_FOLLOWER_YELLOW_ASSET_PATH = LW_BENCHHUB_DATA_PATH / "assets" / "so101_follower_yellow.usd"

SO101_FOLLOWER_YELLOW_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_YELLOW_ASSET_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
            sleep_threshold=0.00005,  # follow isaacsim 5.0.0 tutorial 7 setting
            stabilization_threshold=0.00001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.20, 0.40, 0.71),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0
        }
    ),
    actuators={
        "sts3215": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=10,
            velocity_limit_sim=10.0,
            stiffness=17.8,
            damping=0.60,
        )
    },
    soft_joint_pos_limit_factor=1,
)
