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

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from lw_benchhub.utils.lerobot_common import common


@dataclass
class PPOArgs:
    action_dim: int = 6
    exp_name: Optional[str] = "ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    """the name of this experiment"""
    seed: int = 3
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_kwargs: dict = field(default_factory=dict)
    """extra environment kwargs to pass to the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    num_steps: int = 16
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.9
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.2
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    save_model_freq: int = 100
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def observation(obs, device="cuda"):
    ret = dict()
    image_keys = [k for k in obs.keys() if "image" in k]
    if len(image_keys) > 0:
        rgb_images = []
        for key in image_keys:
            rgb_images.append(obs[key])
            del obs[key]
        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        ret["rgb"] = rgb_images
    # flatten the rest of the data which should just be state data
    state_obs = common.flatten_state_dict(
        obs, use_torch=True, device=device
    )
    ret["state"] = state_obs
    return ret


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_logger(args):
    writer = SummaryWriter(f"policy/maniskill_ppo/logs/runs/{args.exp_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)
    return logger


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        if "rgb" in sample_obs:
            in_channels = sample_obs["rgb"].shape[-1]
            image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

            # here we use a NatureCNN architecture to process images, but any architecture is permissble here
            cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=8,
                    stride=4,
                    padding=0,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
                ),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
                ),
                nn.ReLU(),
                nn.Flatten(),
            )

            # to easily figure out the dimensions after flattening, we pass a test tensor
            with torch.no_grad():
                n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
            extractors["rgb"] = nn.Sequential(cnn, fc)
            self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                             torch.uint8 if v.dtype == np.uint8 else
                             torch.int16 if v.dtype == np.int16 else
                             torch.int32 if v.dtype == np.int32 else
                             v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class Agent(nn.Module):
    def __init__(self, sample_obs, action_dim):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, obs, action=None):
        x = self.feature_net(obs)
        action_mean = self.actor_mean(x)
        # action_mean = torch.clamp(action_mean, -0.2, 0.2)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            # action = torch.clamp(action, -0.2, 0.2)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PPO():
    def __init__(self, envs, sample_obs, args, device="cuda", train=True):
        self.device = device
        self.action_dim = np.prod(envs.unwrapped.single_action_space.shape) if envs is not None else args.action_dim
        self.agent = Agent(sample_obs, action_dim=self.action_dim).to(device)
        if train:
            self.args = args
            self.num_envs = args.num_envs
            self.num_steps = args.num_steps
            self.batch_size = int(self.num_envs * self.num_steps)
            self.minibatch_size = int(self.batch_size // args.num_minibatches)
            self.num_iterations = args.total_timesteps // self.batch_size * self.num_steps
            self.envs = envs
            self.cumulative_times = defaultdict(float)
            self.global_step = 0
            self.obs_rms = torch.zeros(sample_obs["state"].shape[-1], device=device)
            self.obs_var = torch.ones(sample_obs["state"].shape[-1], device=device)
            self.obs_count = 0

            self.optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)
            state_dim = sample_obs["state"].shape[-1]
            self.logger = get_logger(args)
            obs_space = gym.spaces.Dict({'state': gym.spaces.Box(-np.inf, np.inf, (state_dim,), dtype=np.float32)})
            if "rgb" in sample_obs:
                rgb_shape = sample_obs["rgb"].shape[1:]
                obs_space['rgb'] = gym.spaces.Box(0, 255, rgb_shape, dtype=np.uint8)

            self.obs = DictArray((self.num_steps, self.num_envs), obs_space, device=device)
            self.actions = torch.zeros((self.num_steps, self.num_envs) + (self.action_dim,)).to(device)
            self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
            self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
            self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
            self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)
            self.final_values = torch.zeros((self.num_steps, self.num_envs), device=device)

    def load_model(self, pt_path):
        self.agent.load_state_dict(torch.load(pt_path))

    def save_model(self, iteration):
        model_path = f"policy/maniskill_ppo/logs/runs/{self.args.exp_name}/ckpt_{iteration}.pt"
        torch.save(self.agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        return model_path

    def clip_action(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        action[:, -1] = action[:, -1] * 0.2
        action[:, :-1] = action[:, :-1] * 0.05
        return action

    def collect_data(self, next_obs, next_done):
        self.final_values = torch.zeros((self.args.num_steps, self.args.num_envs), device=self.device)
        self.rollout_time = time.perf_counter()
        for step in range(0, self.num_steps):
            self.global_step += self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            self.actions[step] = action
            self.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = self.envs.step(action)
            next_obs = observation(next_obs['policy'])
            if (next_obs['state'] == torch.inf).sum() > 0:
                breakpoint()

            # if hasattr(self, 'obs_rms'):
            #     next_obs["state"] = (next_obs["state"] - self.obs_rms) / torch.sqrt(self.obs_var + 1e-8)
            # else:
            #     next_obs = next_obs

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            self.rewards[step] = reward.view(-1) * self.args.reward_scale
            self.logger.add_scalar("Reward/reaching_reward", infos['log']['Episode_Reward/reaching_reward'].cpu().numpy(), self.global_step)
            self.logger.add_scalar("Reward/grasp_reward", infos['log']['Episode_Reward/grasp_reward'].cpu().numpy(), self.global_step)
            self.logger.add_scalar("Reward/place_reward", infos['log']['Episode_Reward/place_reward'].cpu().numpy(), self.global_step)
            self.logger.add_scalar("Reward/touching_table", infos['log']['Episode_Reward/touching_table'].cpu().numpy(), self.global_step)
            if next_done.any():
                final_obs = observation(common.torch_clone_dict(infos['final_obs'])['policy'])
                done_mask = next_done.clone().bool()
                for k in final_obs:
                    final_obs[k] = final_obs[k][done_mask]
                with torch.no_grad():
                    self.final_values[step, torch.arange(self.args.num_envs, device=self.device)[done_mask]] = self.agent.get_value(final_obs).view(-1)
        self.rollout_time = time.perf_counter() - self.rollout_time
        self.cumulative_times["rollout_time"] += self.rollout_time
        return next_obs, next_done

    def anneal_lr(self, iteration):
        frac = 1.0 - (iteration - 1.0) / self.num_iterations
        lrnow = frac * self.args.learning_rate
        self.optimizer.param_groups[0]["lr"] = lrnow

    def get_value(self, next_obs, next_done):
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                real_next_values = next_not_done * nextvalues + self.final_values[t]  # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if self.args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == self.num_steps - 1:  # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0.  # the sum of the second term
                        value_term_sum = 0.  # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + self.args.gae_lambda * lam_coef_sum
                    reward_term_sum = self.args.gae_lambda * self.args.gamma * reward_term_sum + lam_coef_sum * self.rewards[t]
                    value_term_sum = self.args.gae_lambda * self.args.gamma * value_term_sum + self.args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - self.values[t]
                else:
                    delta = self.rewards[t] + self.args.gamma * real_next_values - self.values[t]
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * next_not_done * lastgaelam  # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + self.values
            return advantages, returns

    def update(self, next_obs, next_done):

        advantages, returns = self.get_value(next_obs, next_done)
        # flatten the batch
        b_obs = self.obs.reshape((-1,))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + (self.action_dim,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        self.agent.train()
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                    v_loss = None
                    break

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        self.cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        if v_loss is not None:
            self.logger.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.logger.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            self.logger.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            self.logger.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
            self.logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
            self.logger.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
            self.logger.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
            self.logger.add_scalar("losses/explained_variance", explained_var, self.global_step)
            self.logger.add_scalar("time/step", self.global_step, self.global_step)
            self.logger.add_scalar("time/update_time", update_time, self.global_step)
            self.logger.add_scalar("time/rollout_time", self.rollout_time, self.global_step)
            self.logger.add_scalar("time/rollout_fps", self.num_envs * self.num_steps / self.rollout_time, self.global_step)
            for k, v in self.cumulative_times.items():
                self.logger.add_scalar(f"time/total_{k}", v, self.global_step)
            self.logger.add_scalar("time/total_rollout+update_time", self.cumulative_times["rollout_time"] + self.cumulative_times["update_time"], self.global_step)
