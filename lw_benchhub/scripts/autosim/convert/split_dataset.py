"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="dataset split.")
parser.add_argument("--input_file", type=str, default="./datasets/autosim/test_dataset.hdf5", help="File path to load autosim final generated demos.")
parser.add_argument("--output_dir", type=str, default="./datasets/autosim/test_dataset_split", help="File path to save processed split dataset from autosim final generated demos.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
from tqdm import tqdm
import json

from isaaclab.utils.datasets import HDF5DatasetFileHandler


def main():
    # Load dataset
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The dataset file {args_cli.input_file} does not exist.")
    input_dataset_handler = HDF5DatasetFileHandler()
    input_dataset_handler.open(args_cli.input_file)

    env_args = json.loads(input_dataset_handler._hdf5_data_group.attrs["env_args"])

    os.makedirs(args_cli.output_dir, exist_ok=True)

    output_episode_index = 0
    episode_names = list(input_dataset_handler.get_episode_names())
    for episode_name in tqdm(episode_names):
        episode_data = input_dataset_handler.load_episode(episode_name, device=args_cli.device)
        if episode_data.success is not None and not episode_data.success:
            continue
        if "actions" not in episode_data.data:
            continue
        output_episode_dir = os.path.join(args_cli.output_dir, f"demo_{output_episode_index}")
        output_episode_index += 1
        os.makedirs(output_episode_dir, exist_ok=True)
        output_file = os.path.join(output_episode_dir, "dataset_success.hdf5")
        output_dataset_handler = HDF5DatasetFileHandler()
        output_dataset_handler.create(output_file)
        output_dataset_handler._hdf5_data_group.attrs['env_args'] = json.dumps(env_args)
        output_dataset_handler.write_episode(episode_data)

        output_dataset_handler.flush()
        output_dataset_handler.close()

    input_dataset_handler.close()


if __name__ == "__main__":
    # run the main function
    main()
