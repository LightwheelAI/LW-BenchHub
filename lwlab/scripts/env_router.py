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
import argparse
from functools import partial
import gymnasium as gym
from isaaclab.app import AppLauncher
from lwlab.distributed.env_router import EnvRouter

# add argparse arguments
parser = argparse.ArgumentParser(description="environments.")
parser.add_argument("--remote_protocol", type=str, default="ipc", help="Remote protocol, can be ipc or restful")
parser.add_argument("--ipc_host", type=str, default="127.0.0.1", help="IPC host")
parser.add_argument("--ipc_port", type=int, default=50000, help="IPC port")
parser.add_argument("--ipc_authkey", type=str, default="lightwheel", help="IPC authkey")
parser.add_argument("--restful_host", type=str, default="0.0.0.0", help="Restful host")
parser.add_argument("--restful_port", type=int, default=8000, help="Restful port")
parser.add_argument("--worker_count", type=int, default=1, help="Env Worker Count")
parser.add_argument("--router_device", type=str, default="cuda:0", help="Router Device")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# yaml_args = config_loader.load(args_cli.task_config)
# args_cli.__dict__.update(yaml_args.__dict__)


if args_cli.remote_protocol == "restful":
    from lwlab.distributed.restful import RestfulEnvWrapper
    RemoteEnvWrapper = partial(RestfulEnvWrapper, address=(args_cli.restful_host, args_cli.restful_port))
elif args_cli.remote_protocol == "ipc":   # ipc
    from lwlab.distributed.ipc import IpcDistributedEnvWrapper
    RemoteEnvWrapper = partial(IpcDistributedEnvWrapper, address=(args_cli.ipc_host, args_cli.ipc_port), authkey=args_cli.ipc_authkey.encode())


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    """Rest everything follows."""
    env = EnvRouter(
        worker_count=args_cli.worker_count,
        device=args_cli.router_device,
        app_launcher_args=vars(args_cli),
        address=(args_cli.ipc_host, args_cli.ipc_port + 1)
    )
    with RemoteEnvWrapper(env=env) as env_server:
        env_server.serve()


if __name__ == "__main__":
    main()
