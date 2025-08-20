import numpy as np
from .dp_model import DP
import yaml
import torch


def encode_obs(observation):
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):
    ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = int(usr_args['actions_dim'] / usr_args['decimation'])  # 2 gripper

    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']

    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)


def eval(TASK_ENV, model, observation, usr_args, v):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)

    # ======== Get Action ========
    actions = model.get_action(obs)[:usr_args['decimation']]
    actions = torch.from_numpy(actions.flatten().reshape(1, -1)).float().to("cpu")
    obs, _, ter, _, _, = TASK_ENV.step(actions)

    observation = {}
    observation["observation"] = {
        "head_camera": {
            "rgb": obs['policy'][usr_args['head_camera']].cpu().numpy()[0]
        },
        "left_camera": {
            "rgb": obs['policy'][usr_args['left_camera']].cpu().numpy()[0]
        },
        "right_camera": {
            "rgb": obs['policy'][usr_args['right_camera']].cpu().numpy()[0]
        }
    }
    observation["joint_action"] = {
        "vector": obs['policy']['joint_pos'][0].cpu().numpy()
    }
    v.add_image(obs['policy'][usr_args['head_camera']].cpu().numpy()[0])
    obs = encode_obs(observation)
    model.update_obs(obs)
    return ter


def reset_model(model):
    model.reset_obs()
