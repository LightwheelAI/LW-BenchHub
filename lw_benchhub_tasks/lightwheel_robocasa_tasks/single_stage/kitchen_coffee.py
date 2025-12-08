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

import numpy as np
import torch

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lw_benchhub.core.mdp as mdp
import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


@configclass
class G1VisualObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_target_pos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})

        first_person_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("first_person_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        left_hand_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_hand_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        right_hand_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_hand_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class PnPCoffee(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.COFFEE_MACHINE, FixtureType.COUNTER]
    """
    Class encapsulating the atomic pick and place coffee tasks.

    Args:
        behavior (str): "counter_to_machine" or "machine_to_counter". Used to define the desired
            pick and place behavior for the task.
    """
    behavior = "machine_to_counter"
    # observations: G1VisualObservationsCfg = G1VisualObservationsCfg()

    def _setup_scene(self, env, env_ids=None):
        return super()._setup_scene(env, env_ids)
        self.coffee_machine.reset_state()

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the coffee tasks. (Coffee machine and counter)
        """
        super()._setup_kitchen_references(scene)
        self.coffee_machine = self.register_fixture_ref("coffee_machine", dict(id=FixtureType.COFFEE_MACHINE))
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.coffee_machine)
        self.init_robot_base_ref = self.coffee_machine

    def get_ep_meta(self):
        """
        Get the episode metadata for the coffee tasks.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        if self.behavior == "counter_to_machine":
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the counter and place it under the coffee machine dispenser."
        elif self.behavior == "machine_to_counter":
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from under the coffee machine dispenser and place it on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the coffee tasks. This includes the object placement configurations.
        Place the mug on the counter or under the coffee machine dispenser based on the behavior.

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        if self.behavior == "counter_to_machine":
            cfgs.append(
                dict(
                    name="obj",
                    obj_groups="mug",
                    placement=dict(
                        fixture=self.counter,
                        sample_region_kwargs=dict(
                            ref=self.coffee_machine,
                        ),
                        size=(0.30, 0.40),
                        pos=("ref", -1.0),
                        rotation=[np.pi / 4, np.pi / 2],
                    ),
                )
            )
        elif self.behavior == "machine_to_counter":
            cfgs.append(
                dict(
                    name="obj",
                    obj_groups="mug",
                    placement=dict(
                        fixture=self.coffee_machine,
                        ensure_object_boundary_in_range=False,
                        margin=0.0,
                        ensure_valid_placement=False,
                        rotation=(np.pi / 8, np.pi / 4),
                    ),
                )
            )
        else:
            raise NotImplementedError

        return cfgs

    def _check_success(self, env):
        """
        Check if the coffee task is successful.
        This includes checking if the gripper is far from the object and the object is in correctly placed
        on the desired fixture (counter or coffee machine).
        """
        gripper_obj_far = OU.gripper_obj_far(env=env)
        obj_z_up = OU.is_obj_z_up(env, "obj")

        if self.behavior == "counter_to_machine":
            contact_check = self.coffee_machine.check_receptacle_placement_for_pouring(env, "obj").detach().clone().to(torch.bool)
        elif self.behavior == "machine_to_counter":
            contact_check = OU.check_obj_fixture_contact(env, "obj", self.counter).to(torch.bool)

        return contact_check & gripper_obj_far & obj_z_up


class CoffeeSetupMug(PnPCoffee):
    """
    Class encapsulating the coffee setup task. Pick the mug from the counter and place it under the coffee machine dispenser.
    """

    task_name: str = "CoffeeSetupMug"
    behavior = "counter_to_machine"

    def _check_success(self, env):
        result = super()._check_success(env)
        obj_height = env.scene.rigid_objects["obj"].data.body_com_pos_w[:, 0, 2]
        default_obj_height = env.scene.rigid_objects["obj"].data.default_root_state[:, 2]
        higher_than_default = obj_height > default_obj_height
        return result & higher_than_default


class CoffeeServeMug(PnPCoffee):
    """
    Class encapsulating the coffee serve task. Pick the mug from under the coffee machine dispenser and place it on the counter.
    """

    task_name: str = "CoffeeServeMug"
    behavior = "machine_to_counter"


class CoffeeSetupMugMimic(CoffeeSetupMug, MimicEnvCfg):
    task_name: str = "CoffeeSetupMug-Mimic"

    @configclass
    class SubtaskCfg(ObsGroup):
        """
        Subtask configuration for the coffee setup task.
        """
        grasp_cup = ObsTerm(func=OU.grasp_obj)
        put_to_coffee_machine = ObsTerm(func=OU.put_obj_to_coffee_machine)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "stack_bowls_in_sink"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        self.subtask_terms.put_to_coffee_machine.params = {'judge_obj_in_coffee_machine': self.coffee_machine.check_receptacle_placement_for_pouring}

        self.__prepare_subtask_configs()

    def __prepare_subtask_configs(self):
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref='obj',
                subtask_term_signal="grasp_cup",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp the mug",
                next_subtask_description="Put to coffee machine",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref='coffee_machine',
                subtask_term_signal="put_to_coffee_machine",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Put the mug to coffee machine",
                next_subtask_description="Leave coffee machine",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = subtask_configs


class StartCoffeeMachine(LwTaskBase):
    layout_registry_names: list[int] = [FixtureType.COFFEE_MACHINE, FixtureType.COUNTER]
    """
    Class encapsulating the coffee press button task. Press the button on the coffee machine to serve coffee.
    """

    task_name: str = "StartCoffeeMachine"

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.coffee_machine.reset_state()

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the coffee press button task. (Coffee machine and counter the coffee machine is on)
        """
        super()._setup_kitchen_references(scene)
        self.coffee_machine = self.register_fixture_ref("coffee_machine", dict(id=FixtureType.COFFEE_MACHINE))
        self.counter = self.get_fixture(FixtureType.COUNTER, ref=self.coffee_machine)
        self.init_robot_base_ref = self.coffee_machine

    def get_ep_meta(self):
        """
        Get the episode metadata for the coffee press button task.
        This includes the language description of the task.

        Returns:
            dict: Episode metadata.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Press the button on the coffee machine to serve coffee."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the coffee press button task. This includes the object placement configurations.
        Places the mug under the coffee machine dispenser.

        Returns:
            list: List of object configurations.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups="mug",
                placement=dict(
                    fixture=self.coffee_machine,
                    ensure_object_boundary_in_range=False,
                    margin=0.0,
                    ensure_valid_placement=False,
                    rotation=(np.pi / 8, np.pi / 4),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the coffee press button task is successful.
        This includes checking if the gripper is far from the object and the coffee machine is turned on/button has been pressed.
        """
        gripper_button_far = self.coffee_machine.gripper_button_far(env=env)
        turned_on = self.coffee_machine.get_state()["turned_on"]
        return turned_on & gripper_button_far
