
#%% Isaac Sim - Start up.

print("Step 1 - Starting up Isaac Sim")

import argparse

import pinocchio

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
AppLauncher.add_app_launcher_args(parser)
app_launcher_args = parser.parse_args([])

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


#%% Lightwheel Labs - Load kitchen USDs and layout

print("Step 2 - LightWheel Labs - Load kitchen USDs and layout")

import gymnasium as gym

from isaaclab_tasks.utils import import_packages
from isaaclab.utils.configclass import configclass

from lwlab.utils.env import ExecuteMode
from lwlab.utils.env import load_robocasa_cfg_cls_from_registry
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn

# Load the environment configs into gym
import_packages(
    "lwlab.core",
    # The blacklist is used to prevent importing configs from sub-packages
    blacklist_pkgs=["utils", ".mdp", ".devices"]
)
import_packages("tasks")

# Get the Kitchen Config from the registry
layout = "robocasakitchen-4-2"
scene_type = layout.split("-", 1)[0]
layout_cfg = load_robocasa_cfg_cls_from_registry("scene", scene_type.capitalize(), "env_cfg_entry_point")

# Add all the missing attributes to the config
layout_cfg.scene_name = layout
layout_cfg.device = "cuda"
layout_cfg.num_envs = 1
layout_cfg.usd_simplify = False
layout_cfg.first_person_view = None
layout_cfg.max_scene_retry = 10
layout_cfg.max_object_placement_retry = 100

# Get a task and robot
task_name = "OpenDishwasher"
robot_name = "PandaOmron-Rel"
task_env_cfg = load_robocasa_cfg_cls_from_registry("task", task_name, "env_cfg_entry_point")
robot_env_cfg = load_robocasa_cfg_cls_from_registry("robot", robot_name, "env_cfg_entry_point")

@configclass
class RobocasaEnvCfg(robot_env_cfg, task_env_cfg, layout_cfg):
    pass

env_cfg = RobocasaEnvCfg(
    execute_mode=ExecuteMode.TELEOP,
    usd_path=layout,
    robot_scale=1.0,
)

from isaaclab.scene import InteractiveSceneCfg
# CAUTION: Hacks on hacks!
# Undo lwlabs monkey patch in: /workspaces/lwlab/lwlab/core/scenes/base.py
if '_usd_path' in InteractiveSceneCfg.__dataclass_fields__:
    del InteractiveSceneCfg.__dataclass_fields__['_usd_path']


#%% 

print("Step 3 - Isaac Arena - Compile the environment")

from isaaclab.scene import InteractiveSceneCfg

from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.assets.object import Object, ObjectType
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import Scene
from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.tasks.open_door_task import OpenDoorTask
from isaac_arena.assets.object_reference import OpenableObjectReference


# Wrap the background
# TODO(alexmillane, 2025.09.17): We should switch this for a background!
kitchen_background = Object(
    name="background",
    prim_path="{ENV_REGEX_NS}/background",
    usd_path=env_cfg.usd_path,
    object_type=ObjectType.BASE,
)

# Collect all the (distractor) objects
objects: list[Object] = []
for position_xyz, rotation_xyzw, obj in env_cfg.object_placements.values():
    rotation_wxyz = Tn.convert_quat(rotation_xyzw, to="wxyz")
    arena_object = Object(
        name=obj.task_name,
        prim_path="{ENV_REGEX_NS}/" + f"{obj.task_name}",
        usd_path=obj.obj_path,
        object_type=None,
        initial_pose=Pose(
            position_xyz=position_xyz,
            rotation_wxyz=rotation_wxyz,
        )
    )
    objects.append(arena_object)

# Create a reference to the task-relavent object (the dishwasher)
dishwasher = OpenableObjectReference(
    name="dishwasher",
    prim_path="{ENV_REGEX_NS}/background/dishwasher_left_group",
    parent_asset=kitchen_background,
    openable_joint_name = "door_joint",
    openable_open_threshold = 0.5
)

# Compose the scene
scene = Scene([kitchen_background, *objects, dishwasher])

# Create the task
task = OpenDoorTask(dishwasher)

# Get the robot
embodiment = FrankaEmbodiment(
    # In front of the dishwasher
    initial_pose=Pose(
        position_xyz=(1.25, -3.0, 0.0),
        rotation_wxyz=(0.0, 0.0, 0.0, 1.0),
    )
)

isaac_arena_environment = IsaacArenaEnvironment(
    name="toaster_kitchen",
    embodiment=embodiment,
    scene=scene,
    task=task,
)

args_parser = get_isaac_arena_cli_parser()
args_cli = args_parser.parse_args([])

builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env = builder.make_registered()
env.reset()

#%%

import torch
import tqdm

print("Step 4 - Isaac Arena - Run the environment")

# Open the dishwasher to show we can interact with the scene.
dishwasher.open(env, env_ids=None, percentage=0.45)

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

#%%

