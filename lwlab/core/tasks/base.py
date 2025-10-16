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

from dataclasses import MISSING
from typing import Any, Dict, List
from copy import deepcopy
import torch

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaac_arena.tasks.task_base import TaskBase

from lwlab.core.context import get_context
import lwlab.core.mdp as mdp
from lwlab.utils.env import ExecuteMode
from lwlab.core.cfg import LwBaseCfg
from lwlab.utils.place_utils.contact_queue import ContactQueue
from lwlab.core.checks.checker_factory import get_checkers_from_cfg, form_checker_result
from lwlab.utils.place_utils.usd_object import USDObject
import lwlab.utils.place_utils.env_utils as EnvUtils
import numpy as np
from isaac_arena.utils.configclass import make_configclass
from isaac_arena.assets.object_library import LibraryObject
from isaac_arena.assets.object_base import ObjectType
import lwlab.utils.object_utils as OU
from lwlab.utils.fixture_utils import fixture_is_type
from lwlab.core.models.fixtures.fixture import FixtureType
import lwlab.utils.math_utils.transform_utils.numpy_impl as Tn
import lwlab.utils.math_utils.transform_utils.torch_impl as Tt
from lwlab.utils.log_utils import copy_dict_for_json
from lightwheel_sdk.loader import ENDPOINT
from lwlab.core.models.objects.LwLabObject import LwLabObject
from isaaclab.sensors import ContactSensorCfg
from lwlab.core.models.fixtures import Fixture, FixtureType, fixture_is_type
import lwlab.utils.fixture_utils as FixtureUtils
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
from isaac_arena.assets.object_reference import ObjectReference
from isaac_arena.assets.asset import Asset
from isaac_arena.utils.pose import Pose
from isaac_arena.environments.isaac_arena_manager_based_env import IsaacArenaManagerBasedRLEnvCfg


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        ee_pose = ObsTerm(func=mdp.ee_pose)

        # cabinet_joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel,
        #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
        # )
        # cabinet_joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
        # )
        # rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # cabinet_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
    #         "static_friction_range": (1.0, 1.25),
    #         "dynamic_friction_range": (1.25, 1.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    # reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    init_scene: EventTerm = MISSING


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    # approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # # 2. Grasp the handle
    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )

    # 3. Open the drawer
    # open_drawer_bonus = RewTerm(
    #     func=mdp.open_drawer_bonus,
    #     weight=7.5,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
    # )
    # multi_stage_open_drawer = RewTerm(
    #     func=mdp.multi_stage_open_drawer,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["corpus_to_drawer_0_0"])},
    # )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)
    success: DoneTerm = MISSING


class LwLabTaskBase(TaskBase):
    task_name: str
    task_type: str = "teleop"
    resample_objects_placement_on_reset: bool = True
    resample_robot_placement_on_reset: bool = True
    EMPTY_EXCLUDE_LAYOUTS: list = []
    OVEN_EXCLUDED_LAYOUTS: list = [1, 3, 5, 6, 8, 10, 11, 13, 14, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 36, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    DOUBLE_CAB_EXCLUDED_LAYOUTS: list = [32, 41, 59]
    DINING_COUNTER_EXCLUDED_LAYOUTS: list = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]
    ISLAND_EXCLUDED_LAYOUTS: list = [1, 3, 5, 6, 8, 9, 10, 13, 18, 19, 22, 27, 30, 36, 40, 43, 46, 47, 49, 52, 53, 60]
    STOOL_EXCLUDED_LAYOUT: list = [1, 3, 5, 6, 18, 20, 36, 39, 40, 43, 47, 50, 52]
    SHELVES_INCLUDED_LAYOUT: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    DOUBLE_CAB_EXCLUDED_LAYOUTS: list = [32, 41, 59]

    def __init__(self):
        self.context = get_context()
        self.usd_simplify = self.context.usd_simplify
        self.exclude_layouts = self.EMPTY_EXCLUDE_LAYOUTS
        self.cache_usd_version = self.context.ep_meta.get("cache_usd_version", {})
        self.objects_version = self.cache_usd_version.get("objects_version", None)
        self.sources = self.context.sources
        self.object_projects = self.context.object_projects
        self.seed = self.context.seed
        self.rng = np.random.default_rng(self.seed)
        if self.context.resample_objects_placement_on_reset is not None:
            self.resample_objects_placement_on_reset = self.context.resample_objects_placement_on_reset
        if self.context.resample_robot_placement_on_reset is not None:
            self.resample_robot_placement_on_reset = self.context.resample_robot_placement_on_reset
        self.init_robot_base_ref = None
        self.enable_fixtures = []
        self.movable_fixtures = []
        self.events_cfg = EventCfg()
        self.termination_cfg = TerminationsCfg(
            success=DoneTerm(func=self.check_success_caller)
        )
        self.assets = {}
        self.contact_sensors = {}
        self.init_checkers_cfg()
        self.checkers = get_checkers_from_cfg(self.checkers_cfg)
        self.checker_results = form_checker_result(self.checkers_cfg)
        self.contact_queues = [ContactQueue() for _ in range(self.context.num_envs)]

        # Initialize retry counts
        self.scene_retry_count = 0
        self.object_retry_count = 0

        self._success_cache: torch.Tensor = torch.tensor([0], device=self.context.device, dtype=torch.int32).repeat(self.context.num_envs)
        self._success_flag: torch.Tensor = torch.tensor([False], device=self.context.device, dtype=torch.bool).repeat(self.context.num_envs)

    def modify_env_cfg(self, env_cfg: IsaacArenaManagerBasedRLEnvCfg) -> IsaacArenaManagerBasedRLEnvCfg:
        if self.context.execute_mode == ExecuteMode.TELEOP:
            self._success_count = int(1 / env_cfg.sim.dt / 2)
        else:
            self._success_count = 1
        return env_cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def _check_success(self, env):
        return torch.tensor([False], device=env.device).repeat(env.num_envs)

    def check_success_caller(self, env):
        arena_env = env.cfg.isaac_arena_env
        arena_env.orchestrator.update_state(env)

        for checker in arena_env.task.checkers:
            arena_env.task.checker_results[checker.type] = checker.check(env)

        # at the begining of the episode, dont check success for stabilization
        success_check_result = self._check_success()

        assert isinstance(success_check_result, torch.Tensor), f"_check_success must be a torch.Tensor, but got {type(success_check_result)}"
        assert len(success_check_result.shape) == 1 and success_check_result.shape[0] == env.num_envs, f"_check_success must be a torch.Tensor of shape ({env.num_envs},), but got {success_check_result.shape}"
        success_check_result &= (env.episode_length_buf >= self.scene.start_success_check_count)

        # success delay
        self._success_flag &= (self._success_cache < self._success_count)
        self._success_cache *= (self._success_cache < self._success_count)
        self._success_flag |= success_check_result
        self._success_cache += self._success_flag.int()
        return self._success_cache >= self._success_count

    def init_checkers_cfg(self):
        # checkers
        if self.context.execute_mode in (ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE):
            print("INFO: Running in Replay Mode. Using replay-specific checker config.")
            self.checkers_cfg = {
                "motion": {
                    "warning_on_screen": False
                },
                "kitchen_clipping": {
                    "warning_on_screen": False
                },
                "velocity_jump": {
                    "warning_on_screen": False
                }
            }
        else:
            print("INFO: Running in Teleop Mode. Using default checker config.")
            self.checkers_cfg = {
                "motion": {
                    "warning_on_screen": False
                },
                "kitchen_clipping": {
                    "warning_on_screen": False
                },
                "velocity_jump": {
                    "warning_on_screen": False
                },
                "start_object_move": {
                    "warning_on_screen": False
                }
            }

    def get_checker_results(self):
        """
        Get all checker data for JSON export. This function integrates various types of checker results
        and can be extended to include additional results in the future.

        Returns:
            dict: Complete checker data combining all available checker results
        """
        checker_datas = {}

        for checker in self.checkers:
            checker_datas[checker.type] = checker.get_metrics(self.checker_results[checker.type])

        return checker_datas

    def get_metrics(self):
        return None

    def get_warning_text(self):
        warning_text = ""
        for checker in self.checkers:
            if self.checker_results[checker.type].get("warning_text"):
                warning_text += self.checker_results[checker.type].get("warning_text")
                warning_text += "\n"
        return warning_text

    def _get_obj_cfgs(self):
        """
        Returns a list of object configurations to use in the environment.
        The object configurations are usually environment-specific and should
        be implemented in the subclass.

        Returns:
            list: list of object configurations
        """

        return []

    def _create_objects(self):
        """
        Creates and places objects in the kitchen environment.
        Helper function called by _create_objects()
        """
        # add objects
        self.objects: Dict[str, USDObject] = {}
        if "object_cfgs" in self.context.ep_meta:
            self.object_cfgs: List[Dict[str, Any]] = self.context.ep_meta["object_cfgs"]
            for obj_num, cfg in enumerate(self.object_cfgs):
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                if self.objects_version is not None:
                    for obj_version in self.objects_version:
                        if cfg["name"] in obj_version:
                            object_version = obj_version[cfg["name"]]
                            break
                        # TODO(geng.wang): temp code for task_1w_dev data, will be deleted in the future
                        else:
                            if "mjcf_path" in cfg.get("info", {}).keys():
                                xml_path = cfg.get("info", {})["mjcf_path"]
                                name = xml_path.split("/")[-2]
                                if name in obj_version:
                                    object_version = obj_version[name]
                                    break
                        ##
                model, info = EnvUtils.create_obj(self, cfg, version=object_version)
                cfg["info"] = info
                self.objects[model.task_name] = model
        else:
            self.object_cfgs = self._get_obj_cfgs()
            self.object_cfgs = self.apply_object_init_offset(self.object_cfgs)
            all_obj_cfgs = []
            for obj_num, cfg in enumerate(self.object_cfgs):
                cfg["type"] = "object"
                if "name" not in cfg:
                    cfg["name"] = "obj_{}".format(obj_num + 1)
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
                self.objects[model.task_name] = model
                # self.model.merge_objects([model])
                try_to_place_in = cfg["placement"].get("try_to_place_in", None)
                object_ref = cfg["placement"].get("object", None)

                # place object in a container and add container as an object to the scene
                if try_to_place_in and (
                    "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    container_cfg = {
                        "name": cfg["name"] + "_container",
                        "obj_groups": cfg["placement"].get("try_to_place_in"),
                        "placement": deepcopy(cfg["placement"]),
                        "type": "object",
                    }

                    init_robot_here = cfg.get("init_robot_here", False)
                    if init_robot_here is True:
                        cfg["init_robot_here"] = False
                        container_cfg["init_robot_here"] = True

                    try_to_place_in_kwargs = cfg["placement"].get(
                        "try_to_place_in_kwargs", None
                    )
                    if try_to_place_in_kwargs is not None:
                        for k, v in try_to_place_in_kwargs.items():
                            container_cfg[k] = v

                    container_kwargs = cfg["placement"].get("container_kwargs", None)
                    if container_kwargs is not None:
                        for k, v in container_kwargs.items():
                            container_cfg[k] = v

                    # add in the new object to the model
                    all_obj_cfgs.append(container_cfg)
                    model, info = EnvUtils.create_obj(self, container_cfg)
                    container_cfg["info"] = info
                    self.objects[model.task_name] = model

                    # modify object config to lie inside of container
                    reset_regions = model.get_reset_regions()
                    if "int" in reset_regions:
                        int_region = reset_regions["int"]
                    else:
                        int_region = reset_regions[model.bounded_region_name]
                    cfg["placement"] = dict(
                        size=(int_region["size"][0] / 4, int_region["size"][1] / 4),
                        pos=int_region["offset"],
                        ensure_object_boundary_in_range=False,
                        sample_args=dict(
                            reference=container_cfg["name"],
                            ref_fixture=cfg["placement"]["fixture"],
                        ),
                    )
                elif (
                    object_ref
                    and "in_container" in cfg["info"]["groups_containing_sampled_obj"]
                ):
                    parent = object_ref
                    if parent in self.objects:
                        container_name = parent
                        container_obj = self.objects[parent]
                        container_size = container_obj.size
                        smaller_dim = min(container_size[0], container_size[1])
                        sampling_size = (smaller_dim * 0.5, smaller_dim * 0.5)
                        cfg["placement"] = {
                            "size": sampling_size,
                            "ensure_object_boundary_in_range": False,
                            "sample_args": {"reference": container_name},
                        }

                # append the config for this object
                all_obj_cfgs.append(cfg)

            self.object_cfgs = all_obj_cfgs

        # update cache_usd_version
        objects_version = []
        for cfg in self.object_cfgs:
            objects_version.append({cfg["name"]: cfg["info"]["obj_version"]})
        self.objects_version = objects_version

        for cfg in self.object_cfgs:
            self.add_asset(
                LwLabObject(
                    name=cfg["info"]["task_name"],
                    tags=["object"],
                    usd_path=cfg["info"]["obj_path"],
                    prim_path=f"{{ENV_REGEX_NS}}/{self.scene_type}/{cfg['info']['task_name']}",
                    object_type=ObjectType.RIGID,
                )
            )
            self.add_contact_sensor_cfg(
                name=f"{cfg['info']['task_name']}_contact",
                cfg=ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{self.scene_type}/{cfg['info']['task_name']}/{cfg['info']['name']}",
                    update_period=0.0,
                    history_length=6,
                    debug_vis=False,
                    force_threshold=0.0,
                    filter_prim_paths_expr=[],
                )
            )

    def get_obj_lang(self, obj_name="obj", get_preposition=False):
        """
        gets a formatted language string for the object (replaces underscores with spaces)
        """
        return OU.get_obj_lang(self, obj_name=obj_name, get_preposition=get_preposition)

    def _update_fxtr_obj_placement(self, object_placements):
        updated_obj_names = []
        for obj_name, obj_placement in object_placements.items():
            updated_placement = list(obj_placement)
            obj_cfg = [cfg for cfg in self.object_cfgs if cfg["name"] == obj_name][0]
            ref_fixture = None
            if "fixture" in obj_cfg["placement"]:
                ref_fixture = obj_cfg["placement"]["fixture"]
            if isinstance(ref_fixture, str):
                ref_fixture = self.get_fixture(ref_fixture)
            # TODO: add other sliding fxtr types
            if fixture_is_type(ref_fixture, FixtureType.DRAWER):
                ref_rot_mat = Tn.euler2mat(np.array([0, 0, ref_fixture.rot]))
                updated_placement[0] = np.array(updated_placement[0]) + ref_fixture._regions["int"]["per_env_offset"] @ ref_rot_mat.T
                updated_obj_names.append(obj_name)
            else:
                updated_placement[0] = np.array(updated_placement[0])[None, :].repeat(self.context.num_envs, axis=0)
            object_placements[obj_name] = updated_placement
        return object_placements, updated_obj_names

    def add_asset(self, asset: Asset):
        assert asset.name is not None, "Asset with the same name already exists"
        self.assets[asset.name] = asset

    def add_contact_sensor_cfg(self, name: str, cfg: ContactSensorCfg):
        assert name is not None, "Contact sensor with the same name already exists"
        self.contact_sensors[name] = cfg

    def get_scene_cfg(self):
        fields = []
        for asset in self.assets.values():
            for asset_cfg_name, asset_cfg in asset.get_cfgs().items():
                fields.append((asset_cfg_name, type(asset_cfg), asset_cfg))
        for sensor_name, sensor_cfg in self.contact_sensors.items():
            fields.append((sensor_name, type(sensor_cfg), sensor_cfg))
        NewConfigClass = make_configclass("TaskCfg", fields)
        new_config_class = NewConfigClass()
        return new_config_class

    def apply_object_init_offset(self, cfgs):
        if hasattr(self, "object_init_offset"):
            object_init_offset = getattr(self, "object_init_offset", [0.0, 0.0])
            for cfg in cfgs:
                if "placement" in cfg and "pos" in cfg["placement"]:
                    pos = cfg["placement"]["pos"]
                    cfg["placement"]["pos"] = ((object_init_offset[0] + pos[0]) if (isinstance(pos[0], float) or isinstance(pos[0], int)) else pos[0],
                                               (object_init_offset[1] + pos[1]) if (isinstance(pos[1], float) or isinstance(pos[1], int)) else pos[1])
        return cfgs

    def _load_placement(self):
        objects_placement = {}
        import h5py
        ep_names = self.context.replay_cfgs["ep_names"]
        if len(ep_names) > 1:
            ep_name = ep_names[0]
        else:
            ep_name = ep_names[-1]
        with h5py.File(self.context.replay_cfgs["hdf5_path"], 'r') as f:
            rigid_objects_path = f"data/{ep_name}/initial_state/rigid_object"

            # Check if rigid_objects_path exists in the file
            if rigid_objects_path not in f:
                return objects_placement

            rigid_objects_group = f[rigid_objects_path]

            for obj_name in rigid_objects_group.keys():
                if obj_name not in self.objects.keys():
                    continue
                pose_path = f"{rigid_objects_path}/{obj_name}"
                obj_group = f[pose_path]
                objects_placement[obj_name] = (
                    tuple(obj_group["root_pose"][0][0:3].tolist()), np.array([obj_group["root_pose"][0][4], obj_group["root_pose"][0][5], obj_group["root_pose"][0][6], obj_group["root_pose"][0][3]], dtype=np.float32), self.objects[obj_name]
                )
        return objects_placement

    def _setup_kitchen_references(self):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self.context.ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs = {
            k: self.get_fixture(v) for (k, v) in serialized_refs.items()
        }

    def get_fixture(self, id, ref=None, size=(0.2, 0.2), full_name_check=False, fix_id=None, full_depth_region=False) -> Fixture | None:
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

            full_depth_region (bool): if True, will only sample island counter regions that can be accessed

        Returns:
            Fixture: fixture object
        """
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in self.fixtures.keys():
            return self.fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in self.fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                if full_name_check:
                    matches = [name for name in self.fixtures.keys() if name == id]
                else:
                    matches = [name for name in self.fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if FixtureUtils.is_fxtr_valid(self, self.fixtures[name], size)
                ]
            if (
                len(matches) > 1
                and any("island" in name for name in matches)
                and full_depth_region
            ):
                island_matches = [name for name in matches if "island" in name]
                if len(island_matches) >= 3:
                    depths = [self.fixtures[name].size[1] for name in island_matches]
                    sorted_indices = sorted(range(len(depths)), key=lambda i: depths[i])
                    min_depth = depths[sorted_indices[0]]
                    next_min_depth = (
                        depths[sorted_indices[1]] if len(depths) > 1 else min_depth
                    )
                    if min_depth < 0.8 * next_min_depth:
                        keep = [
                            i
                            for i in range(len(island_matches))
                            if i != sorted_indices[0]
                        ]
                        filtered_islands = [island_matches[i] for i in keep]
                        matches = [
                            name for name in matches if name not in island_matches
                        ] + filtered_islands

            if len(matches) == 0:
                return None
            # sample random key
            if fix_id is not None:
                key = matches[fix_id]
            else:
                key = self.rng.choice(matches)
            return self.fixtures[key]
        else:
            ref_fixture = self.get_fixture(ref)

            # NOTE: I dont konw why error here?
            # assert isinstance(id, FixtureType)
            cand_fixtures: List[Fixture] = []
            for fxtr in self.fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = FixtureUtils.is_fxtr_valid(self, fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            if len(cand_fixtures) == 0:
                raise ValueError(f"No fixture found for {id} with size {size}")

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def register_fixture_ref(self, ref_name: str, fn_kwargs: dict):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference

            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        if ref_name not in self.fixture_refs:
            self.fixture_refs[ref_name] = self.get_fixture(**fn_kwargs)
        return self.fixture_refs[ref_name]

    def _init_ref_fixtures(self):
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                self.add_asset(
                    ObjectReference(
                        name=fixtr.name,
                        object_type=ObjectType.ARTICULATION,
                        prim_path=f"{{ENV_REGEX_NS}}/{self.scene_type}/{fixtr.name}",
                        parent_asset=self.scene_assets[self.scene_type]
                    )
                )
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                fixtr.setup_cfg(self)

    def _apply_object_placements(self, object_placements):
        for obj_pos, obj_quat, obj in object_placements.values():
            if obj.task_name in self.assets:
                obj_quat_wxyz = Tn.convert_quat(obj_quat, to="wxyz")
                obj_pos = Pose(position_xyz=obj_pos, rotation_wxyz=obj_quat_wxyz)
                self.assets[obj.task_name].set_initial_pose(obj_pos)

    def setup_env_config(self, orchestrator):
        self.scene_assets = orchestrator.scene.assets
        self.scene_type = orchestrator.scene.scene_type
        self.fixtures = orchestrator.scene.fixtures
        self._setup_kitchen_references()
        self._init_ref_fixtures()
        self._get_obj_cfgs()
        self._create_objects()

        object_placements = EnvUtils.sample_object_placements(orchestrator)
        self._apply_object_placements(object_placements)

    def get_ep_meta(self):
        ep_meta = {}
        ep_meta["object_cfgs"] = [copy_dict_for_json(cfg) for cfg in self.object_cfgs]
        # serialize np arrays to lists
        for cfg in ep_meta["object_cfgs"]:
            if cfg.get("reset_region", None) is not None:
                if isinstance(cfg["reset_region"], dict):
                    cfg["reset_region"] = [cfg["reset_region"]]
                for region in cfg["reset_region"]:
                    for (k, v) in region.items():
                        if isinstance(v, np.ndarray):
                            region[k] = list(v)

        ep_meta["lang"] = ""
        ep_meta["usd_simplify"] = self.context.usd_simplify
        ep_meta["objects_version"] = self.objects_version
        ep_meta["source"] = self.context.sources
        ep_meta["object_projects"] = self.context.object_projects
        ep_meta["seed"] = self.context.seed
        ep_meta["LW_API_ENDPOINT"] = ENDPOINT
        return ep_meta

    def get_mimic_env_cfg(self):
        return None

    def get_prompt(self):
        return self.get_ep_meta()["lang"]


class BaseTaskEnvCfg(LwBaseCfg):
    execute_mode: ExecuteMode = MISSING  # DONE
    observations: ObservationsCfg = ObservationsCfg()  # TODO
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()  # TODO
    terminations: TerminationsCfg = TerminationsCfg()  # DONE
    events: EventCfg = EventCfg()  # DONE
    task_name: str = MISSING
    reset_objects_enabled: bool = True
    reset_robot_enabled: bool = True
    task_type: str = "teleop"
    fix_object_pose_cfg: dict = None

    def set_reward_gripper_joint_names(self, joint_names):
        pass

    def set_reward_arm_joint_names(self, joint_names):
        pass

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["task_name"] = self.task_name

        return ep_meta

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # general settings
        self.episode_length_s = 8.0
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (3.0, 1.0, 0.3)
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
        # simulation settings
        self.sim.dt = 1 / 100  # physics frequency: 100Hz
        self.sim.render_interval = 4  # render frequency: 25Hz
        self.decimation = 2  # action frequency: 50Hz
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.contact_queues = [ContactQueue() for _ in range(self.num_envs)]

        # render camera settings
        # TODO: xiaowei.song
        if hasattr(self, "enable_cameras") and self.enable_cameras:
            for name, camera_infos in self.observation_cameras.items():
                if self.task_type in camera_infos["tags"]:
                    if self.task_type == "teleop" and self.execute_mode is not ExecuteMode.TELEOP:
                        setattr(self.observations.policy, name,
                                ObsTerm(
                                    func=mdp.image,
                                    params={
                                        "sensor_cfg": SceneEntityCfg(name),
                                        "data_type": "rgb",
                                        "normalize": False,
                                    }
                                )
                                )
