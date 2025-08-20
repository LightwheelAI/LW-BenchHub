# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO101 Follower Robot.

The following configurations are available:

* :obj:`SO101_FOLLOWER_CFG`: SO101 Follower robot configuration

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from pathlib import Path
import numpy as np
from lwlab.data import LWLAB_DATA_PATH


"""Configuration for the LERobot."""
SO101_FOLLOWER_ASSET_PATH = LWLAB_DATA_PATH / "assets" / "so101_follower.usd"

SO101_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_FOLLOWER_ASSET_PATH),
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
    soft_joint_pos_limit_factor=1.0,
)

SO101_FOLLOWER_USD_JOINT_LIMLITS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10, 100.0),
}


BI_SO101_FOLLOWER_ASSET_PATH = LWLAB_DATA_PATH / "assets" / "bi_so101_follower.usd"

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
