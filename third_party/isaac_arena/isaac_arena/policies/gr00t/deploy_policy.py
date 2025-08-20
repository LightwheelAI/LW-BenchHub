import numpy as np
import torch
import dill
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from lwlab.third_party.isaac_arena.isaac_arena.policies.gr00t.gr00t_n1_5_policy import Gr00tN15Policy
from lwlab.third_party.isaac_arena.isaac_arena.policies.gr00t.policy_cfg import GR00TN15Config
from lwlab.third_party.isaac_arena.isaac_arena.policies.gr00t.robot_joints import JointsAbsPosition
from lwlab.third_party.isaac_arena.isaac_arena.policies.gr00t.io_utils import load_robot_joints_config
from lwlab.third_party.isaac_arena.isaac_arena.policies.gr00t.joints_conversion import \
    (remap_sim_joints_to_policy_joints, remap_policy_joints_to_sim_joints)

# TODO(xinjie.yao): change to better design, a hack
g1_state_joints_config = load_robot_joints_config('g1/43dof_joint_space.yaml')
gr00t_joints_config = load_robot_joints_config('g1/gr00t_43dof_joint_space.yaml')
simulation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_joints = len(g1_state_joints_config)

# Encode observation for the model
def encode_obs(observation, language_instruction):
    # TODO(xinjie.yao): check if this is correct
    full_body_joints = observation["joint_action"]["vector"]
    rgb = observation["observation"]["head_camera"]["rgb"]

    robot_state_sim = JointsAbsPosition(
        full_body_joints, g1_state_joints_config, simulation_device
    )
    robot_state_policy = remap_sim_joints_to_policy_joints(robot_state_sim, gr00t_joints_config)
    # Pack inputs to dictionary and run the inference
    observations = {
        "annotation.human.action.task_description": [language_instruction],  # list of strings
        "video.ego_view": rgb.reshape(-1, 1, 256, 256, 3),  # numpy array of shape (N, 1, 256, 256, 3)
        "state.left_arm": robot_state_policy["left_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
        "state.right_arm": robot_state_policy["right_arm"].reshape(-1, 1, 7),  # numpy array of shape (N, 1, 7)
        "state.left_hand": robot_state_policy["left_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
        "state.right_hand": robot_state_policy["right_hand"].reshape(-1, 1, 6),  # numpy array of shape (N, 1, 6)
    }
    return observations

def decode_action(robot_action_policy):
    robot_action_sim = remap_policy_joints_to_sim_joints(
        robot_action_policy, gr00t_joints_config, g1_state_joints_config, simulation_device
    )
    full_body_target_joints_pos = robot_action_sim.get_joints_pos(simulation_device)

    base_height_command = robot_action_policy["action.base_height_command"]
    navigate_command = robot_action_policy["action.navigate_command"]
    return full_body_target_joints_pos, base_height_command, navigate_command

def get_policy_config(usr_args):
    return GR00TN15Config(
        model_path=usr_args["model_path"],
        task_name=usr_args["task_name"],
        action_horizon=usr_args["action_horizon"],
        embodiment_tag=usr_args["embodiment_tag"],
        denoising_steps=usr_args["denoising_steps"],
        num_feedback_actions=usr_args["num_feedback_actions"],
    )

def get_model(usr_args):
    policy_config = get_policy_config(usr_args)
    return Gr00tN15Policy(policy_config), policy_config

def eval(TASK_ENV, model, observation):
    # TODO(xinjie.yao): change to better design, a hack
    policy_config = model.args
    language_instruction = policy_config.language_instruction

    # TODO(xinjie.yao): change to better design to extract a language instruction from the task environment, a hack
    # if model.observation_window is None:
    #     instruction = TASK_ENV.get_instruction()
    #     model.set_language(instruction)

    observations = encode_obs(observation, language_instruction)
    # ======== Get Action ========
    robot_action_policy = model.get_action(observations)
    full_body_target_joints_pos, base_height_command, navigate_command = decode_action(robot_action_policy)
    # TODO(xinjie.yao): add assertion checks
    assert full_body_target_joints_pos.shape[0] == TASK_ENV.action_space.shape[0], f"Full body target joints pos shape: {full_body_target_joints_pos.shape}, num envs: {TASK_ENV.action_space.shape[0]}"
    assert full_body_target_joints_pos.shape[1] == num_joints, f"Full body target joints pos shape: {full_body_target_joints_pos.shape}, num joints: {num_joints}"

    # take only the first num_feedback_actions, the rest are ignored, preventing over memorization
    for i in range(policy_config.num_feedback_actions):
        # Note (xinjie.yao): check with the new action term definition
        # the action space is [num_joints, base_height_command, navigate_command]
        rollout_action = torch.zeros([1, TASK_ENV.action_space.shape[1]], device=simulation_device)

        rollout_action[:, :num_joints] = full_body_target_joints_pos[:, i].clone()
        # convert from np array to torch tensor
        rollout_action[:, -1] = torch.from_numpy(base_height_command[:, i])
        rollout_action[:, -2] = 1
        rollout_action[:, -5:-2] = torch.from_numpy(navigate_command[:, i])
        TASK_ENV.take_action(rollout_action)
    # ============================

def reset_model(model):
    # Note (xinjie.yao): single shot policy, no need to reset
    pass
