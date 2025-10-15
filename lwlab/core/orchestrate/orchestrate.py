from typing import Any, Dict, Optional
from isaac_arena.orchestrator.orchestrator_base import OrchestratorBase
from isaac_arena.embodiments.embodiment_base import EmbodimentBase
from isaac_arena.scene.scene import Scene
from isaac_arena.tasks.task_base import TaskBase
from isaac_arena.utils.pose import Pose
from lwlab.core.models.fixtures.fixture import Fixture as IsaacFixture
import torch
from lwlab.utils.place_utils import env_utils as EnvUtils


class PlacementStrategy:

    def compute_robot_pose(
        self,
        fixtures: Dict[str, Any],
        robot_cfg: Any
    ) -> Optional[Pose]:
        """
        Compute the pose of the robot in the scene.
        """
        robot_pose = EnvUtils.compute_robot_base_placement_pose(self.scene, fixtures, robot_cfg)
        return robot_pose

    def compute_object_poses(
        self,
        objects: Dict[str, Any],
        fixtures: Dict[str, Any],
        robot_pose: Optional[Pose]
    ) -> Dict[str, Pose]:
        """
        Compute the poses of the objects in the scene.
        """
        object_placements = EnvUtils.sample_object_placements(self.scene, need_retry=False)
        return object_placements


class LwLabBaseOrchestrator(OrchestratorBase):

    def __init__(self, placement_strategy: Optional[PlacementStrategy] = None):
        # self.fixture_controllers = dict()
        self.placement_strategy = placement_strategy
        self.scene = None
        self.embodiment = None
        self.task = None

    # def add_fixture_controller(self, fixture_controller):
    #     self.fixture_controllers[fixture_controller.name] = fixture_controller

    # def update_fixture_controllers(self, fixtures: Dict[str, Any]):
    #     for fixture_name, fixture_cfg in fixtures.items():
    #         if fixture_name in self.fixture_controllers:
    #             self.fixture_controllers[fixture_name].update(fixture_cfg)

    def orchestrate(self, embodiment: EmbodimentBase, scene: Scene, task: TaskBase) -> None:

        context = scene.context

        # Second stage: update scene, embodiment, task
        self.scene = scene
        self.embodiment = embodiment
        self.task = task
        embodiment.setup_env_config(self)
        scene.setup_env_config(self)
        task.setup_env_config(self)

        # set up kitchen references
        self.fixture_refs = self.task.fixture_refs

        if context.get("usd_simplify", False):
            # TODO: usd_simplify

            # TODO: modify background
            pass

        # init ref fixtures
        self._init_ref_fixtures()

        # add ref fixtureassets to arena
        self._add_ref_fixtures_to_arena()

        # place robot and objects
        self.place_robot_and_objects()

        # setup scene done terms
        self.setup_scene_done_terms()

        # setup scene event terms
        self.setup_scene_event_terms()

        # combine ep_meta
        self.combine_ep_meta()

    def _init_ref_fixtures(self):
        for fixtr in self.fixture_refs.values():
            if isinstance(fixtr, IsaacFixture):
                fixtr.setup_cfg(self)

    def _add_ref_fixtures_to_arena(self):
        # TODO: add ref fixtures to arena
        pass

    def _reset_internal(self, env_ids, env):
        """
        Reset the event.
        """
        self.scene._setup_scene(env_ids)
        self.scene.reset_root_state(env=env, env_ids=env_ids)

    def init_scene(self, env):
        for fixture_controller in self.fixture_refs.values():
            if isinstance(fixture_controller, IsaacFixture):
                fixture_controller.setup_env(env)

    def update_state(self, env):
        for fixture_controller in self.fixture_refs.values():
            if isinstance(fixture_controller, IsaacFixture):
                fixture_controller.update_state(env)

    def check_success_caller(self, env):
        self.update_state(env)

        for checker in self.scene.checkers:
            self.scene.checkers_results[checker.type] = checker.check(env)

        def _check_success(env):
            return torch.tensor([False], device=env.device).repeat(env.num_envs)

        # at the begining of the episode, dont check success for stabilization
        success_check_result = self.task._check_success(env) if self.task.hasattr('_check_success') else _check_success(env)

        assert isinstance(success_check_result, torch.Tensor), f"_check_success must be a torch.Tensor, but got {type(success_check_result)}"
        assert len(success_check_result.shape) == 1 and success_check_result.shape[0] == env.num_envs, f"_check_success must be a torch.Tensor of shape ({env.num_envs},), but got {success_check_result.shape}"
        success_check_result &= (env.episode_length_buf >= self.scene.start_success_check_count)

        # success delay
        self.scene.success_flag &= (self.scene.success_cache < self.scene.success_count)
        self.scene.success_cache *= (self.scene.success_cache < self.scene.success_count)
        self.scene.success_flag |= success_check_result
        self.scene.success_cache += self.scene.success_flag.int()
        return self.scene.success_cache >= self.scene.success_count

    def setup_scene_done_terms(self):
        """
        Update the state.
        """
        termination_cfg = self.scene.get_termination_cfg()
        from isaaclab.managers import TerminationTermCfg as DoneTerm
        termination_cfg.success = DoneTerm(func=self.check_success_caller)

    def setup_scene_event_terms(self):
        """
        setup the init_scene event.
        """
        events_cfg = self.scene.get_events_cfg()
        from isaaclab.managers import EventTermCfg as EventTerm
        events_cfg.init_scene = EventTerm(func=self.init_scene, mode="startup")

    def combine_ep_meta(self):
        """
        Combine the ep_meta of scene, embodiment, task.
        """
        ep_meta = self.scene.get_ep_meta()
        ep_meta.update(self.embodiment.get_ep_meta())
        ep_meta.update(self.task.get_ep_meta())
        return ep_meta

    def place_robot_and_objects(self):
        """
        Place the robot and objects in the scene.
        """
        if not self.placement_strategy:
            return

        scene_cfg = self.scene.get_scene_cfg()
        embodiment_cfg = self.embodiment.get_scene_cfg()
        task_cfg = self.task.get_scene_cfg()
        fixtures = self._extract_fixtures(scene_cfg)
        objects = self._extract_objects(task_cfg)

        robot_pose = self.placement_strategy.compute_robot_pose(fixtures, embodiment_cfg)
        object_poses = self.placement_strategy.compute_object_poses(objects, fixtures, robot_pose)

        self._apply_robot_pose(embodiment_cfg, robot_pose)
        self._apply_object_poses(task_cfg, object_poses)

    def _extract_fixtures(self, scene_cfg) -> Dict[str, Any]:
        fixtures = scene_cfg.get_fixtures()
        return fixtures

    def _extract_objects(self, task_cfg) -> Dict[str, Any]:
        objects = task_cfg.get_objects()
        return objects

    def _apply_robot_pose(self, embodiment_cfg, robot_pose: Optional[Pose]):
        if robot_pose and embodiment_cfg and hasattr(embodiment_cfg, 'robot'):
            robot_cfg = embodiment_cfg.robot
            if hasattr(robot_cfg, 'init_state'):
                robot_cfg.init_state.pos = robot_pose.position_xyz
                robot_cfg.init_state.rot = robot_pose.rotation_wxyz

    def _apply_object_poses(self, task_cfg, object_poses: Dict[str, Pose]):
        for obj_name, obj_pose in object_poses.items():
            if hasattr(task_cfg, obj_name):
                obj_cfg = getattr(task_cfg, obj_name)
                if hasattr(obj_cfg, 'init_state'):
                    obj_cfg.init_state.pos = obj_pose.position_xyz
                    obj_cfg.init_state.rot = obj_pose.rotation_wxyz

    def _extract_cfg(self, cfg, key):
        if hasattr(cfg, key):
            return getattr(cfg, key)
        return None

    def _apply_cfg(self, cfg, key, value):
        if hasattr(cfg, key):
            setattr(cfg, key, value)
