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

from dataclasses import MISSING

from isaaclab.utils import configclass

import lw_benchhub.core.mdp as mdp
from lw_benchhub.core.mdp.actions.g1_action import G1ActionCfg


@configclass
class ActionsCfg:
    """Teleop Action specifications for the MDP."""

    base_action: mdp.RelativeJointPositionActionCfg = MISSING
    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING


@configclass
class PinkActionsCfg:
    """Teleop Action specifications for the MDP."""

    base_action: mdp.RelativeJointPositionActionCfg = MISSING
    arms_action: G1ActionCfg = MISSING
    left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING


@configclass
class LocoActionsCfg:
    """Loco Action specifications for the MDP."""
    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING
    base_action: mdp.RelativeJointPositionActionCfg = MISSING


@configclass
class RLActionsCfg:
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING


@configclass
class DecoupledWBCActionsCfg:
    """Actions specifications for the G1 robot, with decoupled WBC policy from Gear."""

    # Sequential action terms. Upper body actions are used for WBC policy (base_action).
    # The following ordering must be enforced for WBC policy to function correctly.
    left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING
