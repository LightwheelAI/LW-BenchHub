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
import contextlib
import importlib
import json
import sys
from pathlib import Path

import mediapy as media
import tqdm
import yaml

sys.path.append("./")
sys.path.append("../../policy")

# add argparse arguments
# Get project root directory (assuming script is in lw_benchhub/scripts/policy/)
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
default_config_path = project_root / "policy" / "PI" / "deploy_policy_lerobot.yml"

parser = argparse.ArgumentParser(description="Eval policy in Isaac Lab environments.")
parser.add_argument("--config", type=str, default=str(default_config_path))
parser.add_argument("--overrides", nargs=argparse.REMAINDER)

# parse the arguments
args_cli = parser.parse_args()


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
            except Exception:
                pass

            # use ':' to split config
            if ':' in key:
                keys = key.split(':')
                current_level = override_dict
                for k in keys[:-1]:
                    if k not in current_level:
                        current_level[k] = {}
                    current_level = current_level[k]
                current_level[keys[-1]] = value
            else:
                override_dict[key] = value

        return override_dict

    def deep_merge(original, update):
        for key, value in update.items():
            if (key in original and
                isinstance(original[key], dict) and
                    isinstance(value, dict)):
                deep_merge(original[key], value)
            else:
                original[key] = value
        return original

    if args_cli.overrides:
        overrides = parse_override_pairs(args_cli.overrides)
        config = deep_merge(config, overrides)

    return config


def main(usr_args):

    from lw_benchhub.distributed.proxy import RemoteEnv
    env = RemoteEnv.make(address=('127.0.0.1', 50000), authkey=b'lightwheel')
    from lw_benchhub.distributed.restful import DotDict
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

    policy_name = usr_args["policy_name"]
    policy_module = importlib.import_module("policy")
    policy_class = getattr(policy_module, policy_name)
    policy = policy_class(usr_args)

    usr_args['actions_dim'] = env.action_space.shape[1]
    usr_args['decimation'] = env.unwrapped.cfg.decimation

    has_success = False

    test_num = usr_args.get('test_num', 10)  # default 10
    suc_num = 0
    with (
        contextlib.suppress(KeyboardInterrupt),  # and torch.inference_mode(),
    ):
        for idx in tqdm.tqdm(range(test_num)):
            eval_video_path = Path(f"./eval_result/video/episode{idx}.mp4")
            eval_video_path.parent.mkdir(parents=True, exist_ok=True)
            with media.VideoWriter(path=eval_video_path, shape=(usr_args['height'], usr_args['width'] * len(usr_args['record_camera'])), fps=30) as v:
                obs, _ = env.reset()
                policy.reset_model()
                has_success = policy.eval(env, obs, usr_args, v)
                if has_success:
                    suc_num += 1
                print(f"Current test result: {has_success}. Success/total tested: {suc_num}/{idx+1}")
    print(f"Success rate: {suc_num / test_num}")

    results = {
        "test_count": test_num,
        "success_count": suc_num,
        "success_rate": suc_num / test_num
    }
    with open("./eval_result/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    env.close()
    # env.detach()
    env.close_connection()


if __name__ == "__main__":
    # example: "python lw_benchhub/scripts/policy/eval_policy.py --config policy/GR00T/deploy_policy_piper.yml \
    #           --overrides --env_cfg:task SizeSorting --env_cfg:layout robocasakitchen \
    #           --instruction  "Stack objects on counter from large to small" --test_num 10
    # run the main function
    usr_args = parse_args_and_config()
    main(usr_args)
