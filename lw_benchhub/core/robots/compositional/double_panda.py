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
from typing import Optional

import torch

from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_arena.utils.pose import Pose

import lw_benchhub.core.mdp as mdp
from lw_benchhub.core.robots.robot_arena_base import EmbodimentBaseObservationCfg, EmbodimentBasePolicyObservationCfg, LwEmbodimentBase

from .assets_cfg import DOUBLE_PANDA_CFG, DOUBLE_PANDA_HIGH_PD_CFG, DOUBLE_PANDA_OFFSET_CONFIG  # isort: skip

##
# Pre-defined configs
##
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    right_arm_action: mdp.DifferentialInverseKinematicsActionCfg = MISSING
    left_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    right_gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class DoublePandaActionsCfg:
    left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_L_joint.*"],
        body_name="panda_L_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0434),
                                                                         ),
    )
    right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_R_joint.*"],
        body_name="panda_R_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0434),
                                                                         ),
    )
    left_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_L_finger.*"],
        open_command_expr={"panda_L_finger.*": 0.04},
        close_command_expr={"panda_L_finger.*": 0.0},
    )
    right_gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_R_finger.*"],
        open_command_expr={"panda_R_finger.*": 0.04},
        close_command_expr={"panda_R_finger.*": 0.0},
    )


@configclass
class DoublePandaSceneCfg:
    robot: ArticulationCfg = DOUBLE_PANDA_HIGH_PD_CFG
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                  prim_path="{ENV_REGEX_NS}/Robot/panda_L/panda_L_hand",
                  name="ee_tcp_L",
                  offset=OffsetCfg(
                      pos=(0.0, 0.0, 0.1034),
                  ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_R/panda_R_hand",
                name="ee_tcp_R",
                offset=OffsetCfg(
                    pos=(0.0, 0.0, 0.1034),
                ),
            ),
        ]
    )
    base_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/panda_R/panda_R_link0",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor*"],
    )


@configclass
class DoublePandaObservationsCfg(EmbodimentBaseObservationCfg):
    """Observation specifications for the MDP."""


@configclass
class DoublePandaPolicyObservationsCfg(EmbodimentBasePolicyObservationCfg):
    """Observations for policy group with state values."""

    def __post_init__(self):
        self.concatenate_terms = False


class DoublePandaEnvCfg(LwEmbodimentBase):

    def __init__(self, enable_cameras: bool = False, initial_pose: Optional[Pose] = None):
        super().__init__(enable_cameras, initial_pose)
        self.name = "DoublePanda"
        self.action_config = DoublePandaActionsCfg()
        self.scene_config = DoublePandaSceneCfg()
        self.offset_config = DOUBLE_PANDA_OFFSET_CONFIG
        self.observation_config = DoublePandaObservationsCfg()
        self.policy_observation_config = DoublePandaPolicyObservationsCfg()

    def preprocess_device_action(self, action: dict[str, torch.Tensor], device) -> torch.Tensor:
        num_envs = device.env.num_envs
        left_arm_action = None
        right_arm_action = None
        if self.action_config.left_arm_action.controller.use_relative_mode:  # Relative mode
            left_arm_action = action["left_arm_delta"]
            right_arm_action = action["right_arm_delta"]
        else:  # Absolute mode

            for arm_idx, abs_arm in enumerate([action["left_arm_abs"], action["right_arm_abs"]]):
                arm_action = abs_arm.clone()
                arm_action[3:] = abs_arm[[6, 3, 4, 5]]
                if arm_idx == 0:
                    left_arm_action = arm_action  # robot frame
                else:
                    right_arm_action = arm_action  # robot frame
        left_gripper = torch.tensor([-1.0 if action["left_gripper"] > 0 else 1.0], device=action['rbase'].device)
        right_gripper = torch.tensor([-1.0 if action["right_gripper"] > 0 else 1.0], device=action['rbase'].device)
        return torch.concat([left_arm_action, right_arm_action,
                             left_gripper, right_gripper]).unsqueeze(0)


# @configclass
class DoublePandaRelEnvCfg(DoublePandaEnvCfg):
    # We switch here to a stiffer PD controller for IK tracking to be better.
    def __init__(self, enable_cameras: bool = False, initial_pose: Optional[Pose] = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config.robot = DOUBLE_PANDA_CFG
        self.offset_config = DOUBLE_PANDA_OFFSET_CONFIG
        self.name = "DoublePanda-Rel"


@configclass
class DoublePandaAbsActionsCfg(DoublePandaActionsCfg):
    # Set actions for the specific robot type (franka)
    left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_L_joint.*"],
        body_name="panda_L_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0434),
                                                                         ),
    )
    right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_R_joint.*"],
        body_name="panda_R_hand",
        controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1,
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0434),
                                                                         ),
    )


class DoublePandaAbsEnvCfg(DoublePandaEnvCfg):
    def __init__(self, enable_cameras: bool = False, initial_pose: Optional[Pose] = None):
        super().__init__(enable_cameras, initial_pose)
        self.scene_config.robot = DOUBLE_PANDA_CFG
        self.name = "DoublePanda-Abs"
