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

import argparse
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

from lw_benchhub.utils.config_loader import config_loader
from lw_benchhub.utils.place_utils.env_utils import set_seed

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument("--task_config", type=str, default="default", help="task config")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
yaml_args = config_loader.load(args_cli.task_config)
args_cli.__dict__.update(yaml_args.__dict__)

app_launcher_args = vars(args_cli)
# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)

simulation_app = app_launcher.app

args_cli.device = f"cuda:0"


def tile_images(images, nrows=1):
    """
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched: 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image


def overlay_envs(real_imgs, sim_imgs):
    overlaid_imgs = []
    real_img = real_imgs / 255
    sim_img = sim_imgs / 255
    overlaid_imgs.append(0.5 * real_img + 0.5 * sim_img)

    return tile_images(overlaid_imgs)


def main(args):
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    import gymnasium as gym
    import torch
    import numpy as np
    from isaaclab.envs import ViewerCfg, ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg
    from lw_benchhub.sim2real.lerobot_follower.so100_follower import SO100Follower
    from lw_benchhub.sim2real.lerobot_follower.so101_follower import SO101Follower

    if "-" in args_cli.task:
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        task_name = args_cli.task

    else:  # robocasa
        from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode

        env_cfg = parse_env_cfg(
            scene_backend=args_cli.scene_backend,
            task_backend=args_cli.task_backend,
            task_name=args_cli.task,
            robot_name=args_cli.robot,
            scene_name=args_cli.layout,
            rl_name=args_cli.rl,
            robot_scale=args_cli.robot_scale,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            first_person_view=args_cli.first_person_view,
            enable_cameras=app_launcher._enable_cameras,
            execute_mode=ExecuteMode.TRAIN,
            usd_simplify=args_cli.usd_simplify,
            seed=args_cli.seed,
            sources=args_cli.sources,
            object_projects=args_cli.object_projects,
            headless_mode=args_cli.headless,
        )
        task_name = f"Robocasa-{args_cli.task}-{args_cli.robot}-v0"

        gym.register(
            id=task_name,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",  # lw_benchhub.enhance.envs:ManagerBasedRLDigitalTwinEnv
            kwargs={
            },
            disable_env_checker=True,
        )

    # modify configuration
    env_cfg.terminations.time_out = None
    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg)  # .unwrapped
    set_seed(env_cfg.seed, env.unwrapped, args.torch_deterministic)
    from policy.maniskill_ppo.agent import PPOArgs, PPO, observation

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    next_obs, _ = env.reset()
    next_obs = observation(next_obs['policy'])

    if args_cli.uid == "so101":
        follower = SO101Follower(port="/dev/ttyACM0", calibration_file_name="so101_follower.json", camera_index=0, use_degrees=True)
        reset_qpos = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.87])
    elif args_cli.uid == "so100":
        follower = SO100Follower(port="/dev/ttyACM0", calibration_file_name="so100_follower.json", camera_index=0, use_degrees=True)
        reset_qpos = np.array([0.0, 0.0, 0.0, np.pi / 2, np.pi / 2, 0.0])
    else:
        raise ValueError(f"Invalid robot: {args_cli.uid}")

    args.num_envs = args_cli.num_envs
    agent = PPO(env, next_obs, args, env_cfg.sim.device, train=False)
    if args_cli.checkpoint:
        agent.load_model(args_cli.checkpoint)

    if args_cli.debug:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)

        # Disable all default key bindings
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        fig.canvas.manager.key_press_handler_id = None

        real_imgs = follower.get_sensor_images()['global_camera']['rgb'].squeeze(0)
        sim_imgs = next_obs['policy']['image_global'].squeeze(0).cpu().numpy()
        # initialize the plot
        overlaid_imgs = overlay_envs(real_imgs, sim_imgs)
        im = ax.imshow(overlaid_imgs)
        im2 = ax2.imshow(sim_imgs)
        im3 = ax3.imshow(real_imgs)
        qposs = []
        real_target_qposs = []
        sim_qpos = []
    while True:
        with torch.no_grad():
            follower.reset(reset_qpos)
            real_pos = follower.qpos.to(args_cli.device, dtype=torch.float32)
            next_obs, _ = env.reset()
            for _ in tqdm(range(64)):
                next_obs['policy']['joint_pos'] = real_pos
                next_obs = observation(next_obs['policy'])
                real_cam_obs = follower.get_sensor_images()

                next_obs['rgb'] = torch.tensor(real_cam_obs['global_camera']['rgb'], device=args_cli.device)
                action = agent.agent.get_action(next_obs, deterministic=True)
                joint_pos = env.env.scene.articulations['robot'].data.joint_pos.clone()
                next_obs, reward, terminations, truncations, infos = env.step(action)
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)
                # real_target_qpos = env.env.scene.articulations['robot'].data.joint_pos.clone()
                real_target_qpos = joint_pos + agent.clip_action(action)

                follower.set_target_qpos(real_target_qpos)

                real_pos = follower.qpos.to(args_cli.device, dtype=torch.float32)
                env.env.scene.articulations['robot'].write_joint_position_to_sim(real_pos)

                if args_cli.debug:
                    qposs.append(follower.qpos)
                    real_target_qposs.append(real_target_qpos.cpu())
                    sim_qpos.append(env.env.scene.articulations['robot'].data.joint_pos.clone().cpu().numpy())
                    sim_imgs = next_obs['policy']['image_global'].squeeze(0).cpu().numpy()
                    real_imgs = real_cam_obs['global_camera']['rgb'].squeeze(0)
                    overlaid_imgs = overlay_envs(real_imgs, sim_imgs)
                    im.set_data(overlaid_imgs)
                    im2.set_data(sim_imgs)
                    im3.set_data(real_imgs)
                    # Redraw the plot
                    fig.canvas.draw()
                    fig.show()
                    fig.canvas.flush_events()

                if next_done.any():
                    if args_cli.debug:
                        sim_qpos = np.array(sim_qpos).reshape(-1, 6)
                        qposs = np.array(qposs).reshape(-1, 6)
                        real_target_qposs = np.array(real_target_qposs).reshape(-1, 6)
                        fig, axs = plt.subplots(6, 1, figsize=(8, 12))
                        motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                        for i in range(6):
                            l1, = axs[i].plot(np.arange(len(qposs)), qposs[:, i], c='r', label='real_qpos')
                            l2, = axs[i].plot(np.arange(len(real_target_qposs)), real_target_qposs[:, i], c='b', label='action')
                            l3, = axs[i].plot(np.arange(len(sim_qpos)), sim_qpos[:, i], c='g', label='sim_qpos')
                            axs[i].set_title(f"{i+1} ({motor_names[i]}) ")
                            axs[i].set_xlabel("step")
                            axs[i].set_ylabel("rad")
                            axs[i].legend(loc='best')
                        plt.tight_layout()
                        plt.show()
                        qposs = []
                        real_target_qposs = []
                        sim_qpos = []
                    break


from policy.maniskill_ppo.agent import PPOArgs


@dataclass
class Args:
    # env_id: str
    # """The environment id to train on"""
    # env_kwargs_json_path: Optional[str] = None
    """Path to a json file containing additional environment kwargs to use."""
    ppo: PPOArgs = field(default_factory=PPOArgs)


if __name__ == "__main__":
    from torch import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    args = Args()
    main(args.ppo)
