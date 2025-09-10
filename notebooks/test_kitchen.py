
#%% Lightwheel Labs - Load kitchen USDs and layout

import argparse
from pathlib import Path


import torch
import tqdm
import pinocchio

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Arena CLI parser.")
AppLauncher.add_app_launcher_args(parser)
app_launcher_args = parser.parse_args([])

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab_tasks.utils import import_packages

from lwlab.utils.env import ExecuteMode # parse_env_cfg
from lwlab.utils.env import load_robocasa_cfg_cls_from_registry
from lwlab.core.scenes.loader import object_loader
from lwlab.core.models.fixtures.fixture import Fixture
from lwlab.core.models.fixtures.toaster import Toaster

# Load the environment configs into gym
import_packages(
    "lwlab.core",
    # The blacklist is used to prevent importing configs from sub-packages
    blacklist_pkgs=["utils", ".mdp", ".devices"]
)
import_packages("tasks")

# Get the Kitchen Config from the registry
layout = "robocasakitchen"
layout_cfg = load_robocasa_cfg_cls_from_registry("scene", layout.capitalize(), "env_cfg_entry_point")
print(layout_cfg)

# Add all the missing attributes to the config
layout_cfg.scene_name = layout
layout_cfg.execute_mode = ExecuteMode.TELEOP
layout_cfg.robot_scale = 1.0
layout_cfg.device = "cuda"
layout_cfg.num_envs = 1
layout_cfg.use_fabric = False
layout_cfg.fix_object_pose_cfg = None
layout_cfg.usd_simplify = False
layout_cfg.first_person_view = None

# Instantiate the config
kitchen_cfg = layout_cfg()

# Inspect the fixtures
for fixture_name, fixture in kitchen_cfg.fixtures.items():
    print(fixture_name)
    print(type(fixture))

# Get a toaster from the scene
for fixture in kitchen_cfg.fixtures.values():
    if isinstance(fixture, Toaster):
        print("Found a toaster")
        toaster = fixture

print("Toaster properties")
print(f"Toaster name: {toaster.name}")
print(f"Toaster folder: {toaster.folder}")
print(f"Toaster position: {toaster.pos}")
print(f"Toaster orientation: {toaster.quat}")


# Get the USD path on disk for the toaster
def get_usd_path(fixture: Fixture) -> str:
    xml_path = fixture.folder
    path_attr = "robocasa/models/assets/"
    xml_path = Path(xml_path[xml_path.rfind(path_attr) + len(path_attr):])
    usd_cache_path = object_loader.acquire_object(str(xml_path), "USD")
    return usd_cache_path

def get_prim_path(fixture: Fixture) -> str:
    return "{ENV_REGEX_NS}/Scene/" + f"{fixture.name}"


usd_cache_path = get_usd_path(toaster)
print(f"usd_cache_path: {usd_cache_path}")
print(f"prim_path: {get_prim_path(toaster)}")

#% Isaac Arena - Compile environment

from isaaclab.scene import InteractiveSceneCfg

from isaac_arena.assets.asset_registry import AssetRegistry
from isaac_arena.assets.object import Object, ObjectType
from isaac_arena.geometry.pose import Pose
from isaac_arena.scene.scene import Scene
from isaac_arena.embodiments.franka.franka import FrankaEmbodiment
from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
from isaac_arena.environments.compile_env import ArenaEnvBuilder
from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser
from isaac_arena.tasks.dummy_task import DummyTask

# CAUTION: Hacks on hacks!
# Undo lwlabs monkey patch in: /workspaces/lwlab/lwlab/core/scenes/base.py
if '_usd_path' in InteractiveSceneCfg.__dataclass_fields__:
    del InteractiveSceneCfg.__dataclass_fields__['_usd_path']

# Create a toaster in arena which matches the toaster in the kitchen
# TODO(alexmillane, 2025.09.09): Add a way to make a run-time Object.
# These define-time class variables are not ideal for wrapping external
# run-time objects.
# TODO(alexmillane, 2025.09.09): Add affordances!
# class ArenaToaster(Object):
#     """
#     Encapsulates the pick-up object config for a pick-and-place environment.
#     """

#     name = "toaster"
#     tags = ["object"]
#     usd_path = get_usd_path(toaster)
#     default_prim_path = "{ENV_REGEX_NS}/toaster"
#     object_type = ObjectType.ARTICULATION

#     def __init__(self, prim_path: str = default_prim_path, initial_pose: Pose | None = None):
#         super().__init__(prim_path=prim_path, initial_pose=initial_pose)

# Create a toaster in arena which matches the toaster in the kitchen
# TODO(alexmillane, 2025.09.09): Add a way to make a run-time Object.
# These define-time class variables are not ideal for wrapping external
# run-time objects.
# TODO(alexmillane, 2025.09.09): Move this to isaac_arena core.
# TODO(alexmillane, 2025.09.09): Add affordances!
class ObjectWrapper(Object):
    """ Wraps an object from an external source into isaac_arena"""

    def __init__(self,
                 name: str,
                 usd_path: str,
                 object_type: ObjectType = ObjectType.RIGID,
                 **kwargs):
        self.name = name
        self.usd_path = usd_path
        self.object_type = object_type
        super().__init__(**kwargs)


#%% Wrap fixtures to arena objects

def get_arena_prim_path(fixture: Fixture) -> str:
    return "{ENV_REGEX_NS}/" + f"{fixture.name}"

print("Converting fixtures to arena objects")
arena_objects = []
num_success = 0
MAX_OBJECTS = 10000
for fixture_name, fixture in tqdm.tqdm(kitchen_cfg.fixtures.items()):
    print(f"trying to convert {fixture_name} to an arena object")
    try:
        arena_object = ObjectWrapper(
            name=fixture.name,
            prim_path=get_arena_prim_path(fixture), #get_prim_path(fixture),
            usd_path=get_usd_path(fixture),
            # object_type=ObjectType.RIGID,
            object_type=ObjectType.BASE, # NEED AUTOMATIC DETECTION OF THE OBJECT TYPE.
        )
        arena_object.set_initial_pose(
            Pose(
                position_xyz=fixture.pos,
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0), # FOR SOME REASON THE KITCHEN QUAT IS NOT SET
            )
        )
        print(f"Pose for {fixture_name}: pos={fixture.pos}, quat={fixture.quat}")
    except Exception as e:
        print(f"failed to convert {fixture_name} to an arena object")
        print(e)
        continue
    print("Success!")
    num_success += 1
    arena_objects.append(arena_object)
    if num_success >= MAX_OBJECTS:
        break

print(f"Successfully converted {num_success} out of {len(kitchen_cfg.fixtures)} fixtures")

#%%



# TWO SPECIFIC ASSETS EXAMPLE.

# arena_asset_registry = AssetRegistry()
# arena_kitchen = arena_asset_registry.get_asset_by_name("kitchen")()

# arena_toaster = ObjectWrapper(
#     name="toaster",
#     prim_path="{ENV_REGEX_NS}/toaster", #get_prim_path(toaster), # FIX THE LWLAB PRIM PATH.
#     usd_path=get_usd_path(toaster),
#     object_type=ObjectType.ARTICULATION,
# )

# # try to wrap another fixture!
# paper_towel = kitchen_cfg.fixtures["paper_towel_main_group"]
# arena_paper_towel = ObjectWrapper(
#     name="paper_towel",
#     prim_path="{ENV_REGEX_NS}/paper_towel", #get_prim_path(paper_towel),
#     usd_path=get_usd_path(paper_towel),
#     object_type=ObjectType.RIGID,
# )

# # Position the toaster
# toaster_position = (0.34705, 0.0, 0.13254)
# toaster_rotation = (1, 0, 0, 0)
# arena_toaster.set_initial_pose(Pose(toaster_position, rotation_wxyz=toaster_rotation))

# # Position the paper towel
# toaster_position = (0.34705 + 0.3, 0.0, 0.13254)
# toaster_rotation = (1, 0, 0, 0)
# arena_paper_towel.set_initial_pose(Pose(toaster_position, rotation_wxyz=toaster_rotation))

# scene = Scene([arena_kitchen, arena_toaster, arena_paper_towel])


# ALL ASSETS EXAMPLE.

# scene = Scene([arena_kitchen, *arena_objects])
scene = Scene(arena_objects)

isaac_arena_environment = IsaacArenaEnvironment(
    name="toaster_kitchen",
    embodiment=FrankaEmbodiment(),
    scene=scene,
    task=DummyTask(),
)

args_parser = get_isaac_arena_cli_parser()
args_cli = args_parser.parse_args([])

builder = ArenaEnvBuilder(isaac_arena_environment, args_cli)
env = builder.make_registered()
env.reset()

#%%

# Run some zero actions.
env.reset()
NUM_STEPS = 1
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

#%%
