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

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class StackBowlsInSink(LwTaskBase):
    """
    Stack Bowls: composite task for Washing Dishes activity.

    Simulates the task of stacking bowls in the sink.

    Steps:
        Stack the bowls in the sink.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.SINK]
    task_name: str = "StackBowlsInSink"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Stack the bowls in the sink."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.sink.set_handle_state(mode="off", env=env)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="receptacle1",
                obj_groups="bowl",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="nn",
                    ),
                    size=(0.40, 0.65),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="receptacle2",
                obj_groups="bowl",
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="nn",
                    ),
                    size=(0.40, 0.65),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        receptacle1_in_sink = OU.obj_inside_of(env, "receptacle1", self.sink)
        receptacle2_in_sink = OU.obj_inside_of(env, "receptacle2", self.sink)

        receptacle2_in_receptacle1 = OU.check_obj_in_receptacle(
            env, "receptacle2", "receptacle1"
        )
        receptacle1_in_receptacle2 = OU.check_obj_in_receptacle(
            env, "receptacle1", "receptacle2"
        )

        gripper_receptacle1_far = OU.gripper_obj_far(env, obj_name="receptacle1")
        gripper_receptacle2_far = OU.gripper_obj_far(env, obj_name="receptacle2")
        # print(f"receptacle1_in_sink: {receptacle1_in_sink}, receptacle2_in_sink: {receptacle2_in_sink}, receptacle2_in_receptacle1: {receptacle2_in_receptacle1}, receptacle1_in_receptacle2: {receptacle1_in_receptacle2}, gripper_receptacle1_far: {gripper_receptacle1_far}, gripper_receptacle2_far: {gripper_receptacle2_far}")
        return receptacle1_in_sink & receptacle2_in_sink & (receptacle2_in_receptacle1 | receptacle1_in_receptacle2) & gripper_receptacle1_far & gripper_receptacle2_far


class StackBowlsInSinkMimic(StackBowlsInSink, MimicEnvCfg):
    task_name: str = "StackBowlsInSink-Mimic"

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "stack_bowls_in_sink"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        self.__prepare_subtask_configs()

    def __prepare_subtask_configs(self):
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref=None,
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="move_2",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="random",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                description="Move to receptacle 2",
                next_subtask_description="Grasp receptacle 2",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref='receptacle2',
                subtask_term_signal="grasp_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp receptacle 2",
                next_subtask_description="Lift and put down receptacle 2 in sink",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="lift_put_down_2",
                subtask_term_offset_range=(10, 20),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Lift and put down receptacle 2 in sink",
                next_subtask_description="Move to receptacle 1",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal="move_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Move to receptacle 1",
                next_subtask_description="Grasp receptacle 1",
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                object_ref='receptacle1',
                subtask_term_signal="grasp_1",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp receptacle 1",
                next_subtask_description="Lift and put down receptacle 1 in sink",
            )
        )
        # last subtask
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
        self.subtask_configs["right"] = subtask_configs
