import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)


from third_party.openvla_oft.experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
)
from third_party.openvla_oft.experiments.robot.isaaclab.run_isaaclab_eval import (
    prepare_observation,
)
from openvla_oft_policy import OpenVLAOFTPolicy


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # lwlab environment-specific parameters
    #################################################################################################################
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    max_steps: int = 1500                            # Max number of steps per rollout
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def get_model(user_args):
    cfg = GenerateConfig(**user_args)
    return OpenVLAOFTPolicy(cfg), cfg


def eval(TASK_ENV, model, observation):
    # Initialize model and components

    observation, _ = prepare_observation(observation)

    actions = get_action(
        model.cfg,
        model.model,
        observation,
        model.task_description,
        processor=model.processor,
        action_head=model.action_head,
        proprio_projector=model.proprio_projector,
        noisy_action_projector=model.noisy_action_projector,
        use_film=model.cfg.use_film,
    )


def reset_model(model):
    pass
