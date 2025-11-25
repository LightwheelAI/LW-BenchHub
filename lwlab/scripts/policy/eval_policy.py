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

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
from pathlib import Path
import mediapy as media
import tqdm


# add argparse arguments
parser = argparse.ArgumentParser(description="Eval policy in Isaac Lab environments.")
parser.add_argument("--config", type=str, default="/home/zsy/workspace/lwlab-private/third_party/lwlab/policy/PI/deploy_policy_lerobot.yml")
parser.add_argument("--overrides", nargs=argparse.REMAINDER)

# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""
import contextlib
import importlib
import sys
import yaml
sys.path.append(f"./")
sys.path.append(f"../../policy")


def parse_args_and_config():

    with open(args_cli.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except Exception as e:
                print(f"parsing override {key}: {value}, error: {e}")
            override_dict[key] = value
        return override_dict

    if args_cli.overrides:
        overrides = parse_override_pairs(args_cli.overrides)
        config.update(overrides)

    return config


def main(usr_args):

    from lwlab.distributed.proxy import RemoteEnv
    env = RemoteEnv.make(address=('127.0.0.1', 50000), authkey=b'lightwheel')
    from lwlab.distributed.restful import DotDict
    if "env_cfg" in usr_args and usr_args["env_cfg"]:
        env_cfg = DotDict(usr_args["env_cfg"])
        defaults = {
            "scene_backend": "robocasa",
            "task_backend": "robocasa",
            "device": "cuda:0",
            "robot_scale": 1.0,
            "first_person_view": False,
            "disable_fabric": False,
            "num_envs": 1,
            "usd_simplify": False,
            "video": False,
            "for_rl": False,
            "variant": "Visual",
            "concatenate_terms": False,
            "distributed": False,
            "seed": 42,
            "sources": None,
            "object_projects": None,
            "execute_mode": "eval",
            "replay_cfgs": {"add_camera_to_observation": True},
        }
        for key, value in defaults.items():
            if key not in env_cfg:
                env_cfg[key] = value
    env.attach(env_cfg)
    env = env.unwrapped

    policy_name = usr_args["policy_name"]
    policy_module = importlib.import_module("policy")
    policy_class = getattr(policy_module, policy_name)
    policy = policy_class(usr_args)

    usr_args['actions_dim'] = env.action_space.shape[1]
    usr_args['decimation'] = env.cfg.decimation

    has_success = False

    test_num = 10
    suc_num = 0
    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for idx in tqdm.tqdm(range(test_num)):
            eval_video_path = Path(f"./eval_result/episode{idx}.mp4")
            eval_video_path.parent.mkdir(parents=True, exist_ok=True)
            with media.VideoWriter(path=eval_video_path, shape=(usr_args['height'], usr_args['width'] * len(usr_args['record_camera'])), fps=30) as v:
                obs, _ = env.reset()
                policy.reset_model()
                has_success = policy.eval(env, obs, usr_args, v)
                if has_success:
                    suc_num += 1
    print(f"Success rate: {suc_num / test_num}")
    env.close()
    env.detach()


if __name__ == "__main__":
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
