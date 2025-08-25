from isaaclab.utils import configclass
import lwlab.core.mdp as mdp
from dataclasses import MISSING

from isaac_arena.core.mdp.actions.decoupled_wbc_action import G1DecoupledWBCActionCfg

@configclass
class ActionsCfg:
    """Teleop Action specifications for the MDP."""
    base_action: mdp.RelativeJointPositionActionCfg = MISSING
    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
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
class DecoupledWBCActionsCfg:
    """Actions specifications for the G1 robot, with decoupled WBC policy from Gear."""

    # Sequential action terms. Upper body actions are used for WBC policy (base_action).
    # The following ordering must be enforced for WBC policy to function correctly.
    left_hand_action: mdp.ActionTermCfg = MISSING
    right_hand_action: mdp.ActionTermCfg = MISSING
    base_action: G1DecoupledWBCActionCfg = MISSING