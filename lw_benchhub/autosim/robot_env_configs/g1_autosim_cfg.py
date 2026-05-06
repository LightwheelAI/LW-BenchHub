"""G1 autosim action / observation / event configs.

These configs are shared across G1 autosim pipelines and are injected into
the RoboCasa-based env (replacing the default robot action/obs/event terms).
"""

from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import lw_benchhub.core.mdp as lw_benchhub_mdp


# ---------------------------------------------------------------------------
# Actions  (names must match what G1ActionAdapter expects)
# ---------------------------------------------------------------------------

@configclass
class G1ActionsCfg:
    """Action terms for G1 autosim.

    Action vector layout:
        [0:4]   base_action   (loco cmd: vx, vy, vyaw, mode)
        [4:11]  right_arm     (absolute joint position)
        [11:18] left_arm      (absolute joint position)
        [18:32] gripper       (14 finger joints, absolute position)
    """

    base_action: lw_benchhub_mdp.LegPositionActionCfg = lw_benchhub_mdp.LegPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint",
            "right_hip_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint", "left_ankle_pitch_joint",
            "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
        ],
        body_name="base",
        scale=1.0,
        loco_config="g1_loco.yaml",
        squat_config="g1_squat.yaml",
    )

    right_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )

    left_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )

    gripper_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_hand_index_0_joint",  "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
            "right_hand_thumb_0_joint",  "right_hand_thumb_1_joint",  "right_hand_thumb_2_joint",
            "left_hand_index_0_joint",   "left_hand_index_1_joint",
            "left_hand_middle_0_joint",  "left_hand_middle_1_joint",
            "left_hand_thumb_0_joint",   "left_hand_thumb_1_joint",   "left_hand_thumb_2_joint",
        ],
        scale=1.0,
        use_default_offset=False,
    )


# ---------------------------------------------------------------------------
# Observations / Events (minimal — autosim reads scene data directly)
# ---------------------------------------------------------------------------

@configclass
class G1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions   = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1EventCfg:
    pass
