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

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import random
from dataclasses import dataclass
from typing import Optional

import tyro
from tqdm import tqdm


@dataclass
class Args:
    action_dim: int = 6
    uid: str = None
    checkpoint: Optional[str] = None
    seed: int = 3
    torch_deterministic = True


def main(args_cli):
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    import gymnasium as gym
    import numpy as np
    import torch
    from lw_benchhub.sim2real.lerobot_follower.so100_follower import SO100Follower
    from lw_benchhub.sim2real.lerobot_follower.so101_follower import SO101Follower
    from policy.maniskill_ppo.agent import PPOArgs, PPO, observation

    def set_seed(seed, torch_deterministic):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = torch_deterministic

    args_cli.device = "cuda:0"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = 42
    set_seed(args_cli.seed, args_cli.torch_deterministic)

    if args_cli.uid == "so101":
        follower = SO101Follower(port="/dev/ttyACM0", calibration_file_name="so101_follower.json", camera_index=0, use_degrees=True)
        reset_qpos = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.87])
    elif args_cli.uid == "so100":
        follower = SO100Follower(port="/dev/ttyACM0", calibration_file_name="so100_follower.json", camera_index=0, use_degrees=True)
        reset_qpos = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])
    else:
        raise ValueError(f"Invalid robot: {args_cli.uid}")

    obs = {
        'joint_pos': torch.zeros((1, 6), device=args_cli.device, dtype=torch.float32),
        'target_qpos': torch.zeros((1, 6), device=args_cli.device, dtype=torch.float32),
        'delta_reset_qpos': torch.zeros((1, 5), device=args_cli.device, dtype=torch.float32),
        'image_global': torch.tensor(follower.get_sensor_images()['global_camera']['rgb'], device=args_cli.device),
    }
    sample_obs = observation(obs)
    agent = PPO(envs=None, sample_obs=sample_obs, args=args, device=args_cli.device, train=False)
    if args_cli.checkpoint:
        agent.load_model(args_cli.checkpoint)

    reset_qpos_tensor = torch.tensor(reset_qpos, device=args_cli.device, dtype=torch.float32).unsqueeze(0)

    while True:
        with torch.no_grad():
            follower.reset(reset_qpos)
            real_target_qpos = follower.qpos.to(args_cli.device, dtype=torch.float32)
            for _ in tqdm(range(100)):
                real_qpos = follower.qpos.to(args_cli.device, dtype=torch.float32)
                obs['joint_pos'] = real_qpos
                obs['target_qpos'] = real_target_qpos
                obs['delta_reset_qpos'] = obs['target_qpos'][:, :-1] - reset_qpos_tensor[:, :-1]
                obs['image_global'] = torch.tensor(follower.get_sensor_images()['global_camera']['rgb'], device=args_cli.device)
                obs_real = observation(obs)
                action = agent.agent.get_action(obs_real, deterministic=True)
                real_target_qpos = agent.clip_action(action) + real_qpos

                follower.set_target_qpos(real_target_qpos)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
