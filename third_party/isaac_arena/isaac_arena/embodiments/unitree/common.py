from isaaclab.utils import configclass
import lwlab.core.mdp as mdp
from dataclasses import MISSING


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
