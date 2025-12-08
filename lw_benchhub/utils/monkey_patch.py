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

from copy import deepcopy
import time

import torch


def patch_reset():
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
    from isaacsim.core.simulation_manager import SimulationManager

    def reset(
        self: ManagerBasedRLEnv, seed: int | None = None, env_ids=None, options=None
    ):
        """Resets the specified environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset the specified environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()
        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()
        if hasattr(self.cfg.isaaclab_arena_env.task, "foreground_semantic_id_mapping"):
            self.cfg.isaaclab_arena_env.task.foreground_semantic_id_mapping
            # self.cfg.setup_camera_and_foreground(self.scene)
            self.cfg.isaaclab_arena_env.task.record_semantic_id_mapping(self.scene)
        # return observations
        return self.obs_buf, self.extras
    ManagerBasedRLEnv.reset = reset

# monkey patch the configclass to allow validate dict with key is not a string


# def patch_configclass():
#     from isaaclab.utils.configclass import configclass
#     import sys
#     configclass_module = sys.modules["isaaclab.utils.configclass"]

#     orig_validate = configclass_module._validate

#     def _validate_with_dict_key_not_string(obj, prefix=""):
#         if isinstance(obj, dict):
#             if any(not isinstance(key, str) for key in obj.keys()):
#                 obj = {str(key): value for key, value in obj.items()}
#         return orig_validate(obj, prefix=prefix)

#     configclass_module._validate = _validate_with_dict_key_not_string


def patch_configclass():
    from isaaclab.utils.configclass import configclass
    import sys

    configclass_module = sys.modules["isaaclab.utils.configclass"]
    orig_validate = configclass_module._validate

    def _validate_with_dict_key_not_string(obj, prefix=""):
        if not hasattr(_validate_with_dict_key_not_string, '_visited'):
            _validate_with_dict_key_not_string._visited = set()
            is_top_level = True
        else:
            is_top_level = False

        try:
            obj_id = id(obj)
            if obj_id in _validate_with_dict_key_not_string._visited:
                return []
            if isinstance(obj, (dict, list, tuple)) or hasattr(obj, "__dict__"):
                _validate_with_dict_key_not_string._visited.add(obj_id)

            return orig_validate(obj, prefix=prefix)
        finally:
            if is_top_level:
                delattr(_validate_with_dict_key_not_string, '_visited')

    configclass_module._validate = _validate_with_dict_key_not_string


# monkey patch the recorder manager to have ep_meta stored in the hdf5 file
def patch_recorder_manager_ep_meta():
    from .robocasa_utils import convert_fixture_to_name
    from isaaclab.managers.recorder_manager import RecorderManager

    def get_ep_meta(mgr: RecorderManager):
        ep_meta = mgr._env.cfg.isaaclab_arena_env.orchestrator.get_ep_meta()
        for obj in ep_meta["object_cfgs"]:
            obj["placement"] = convert_fixture_to_name(obj["placement"])
        return ep_meta

    orig_export_episodes = RecorderManager.export_episodes

    def export_episodes(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = list(range(self._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        if len(self.active_terms) and any(
            (env_id in self._episodes and not self._episodes[env_id].is_empty())
            for env_id in env_ids
        ):
            ep_meta = get_ep_meta(self)
            if self._dataset_file_handler is not None:
                self._dataset_file_handler.add_env_args(ep_meta)
            if self._failed_episode_dataset_file_handler is not None:
                self._failed_episode_dataset_file_handler.add_env_args(ep_meta)

            for env_id in env_ids:
                if env_id in self._episodes and not self._episodes[env_id].is_empty():
                    self._episodes[env_id].pre_export()
        orig_export_episodes(self, env_ids)

    RecorderManager.export_episodes = export_episodes


# patch the recorder manager to have joint targets stored in the hdf5 file
def patch_recorder_manager_joint_targets():
    from isaaclab.managers.recorder_manager import RecorderTerm, RecorderManager

    def record_term_record_pre_physics_step(self: RecorderTerm):
        return None, None
    RecorderTerm.record_pre_physics_step = record_term_record_pre_physics_step

    def recorder_manager_record_pre_physics_step(self: RecorderManager):
        if len(self.active_terms) == 0:
            return

        for term in self._terms.values():
            key, value = term.record_pre_physics_step()
            self.add_to_episodes(key, value)

    RecorderManager.record_pre_physics_step = recorder_manager_record_pre_physics_step

    # TODO: add RecorderManager.record_pre_physics_step to env step

    from isaaclab.utils.datasets.episode_data import EpisodeData

    EpisodeData._next_joint_target_index = 0

    def get_joint_target(episode_data: EpisodeData, joint_target_index) -> dict | torch.Tensor | None:
        """Get the joint target of the specified index from the dataset."""
        if "joint_targets" not in episode_data._data:
            return None

        joint_targets = episode_data._data["joint_targets"]

        def get_joint_target_helper(joint_targets, joint_target_index) -> dict | torch.Tensor | None:
            if isinstance(joint_targets, dict):
                output_joint_targets = dict()
                for key, value in joint_targets.items():
                    output_joint_targets[key] = get_joint_target_helper(value, joint_target_index)
                    if output_joint_targets[key] is None:
                        return None
            elif isinstance(joint_targets, torch.Tensor):
                if joint_target_index >= len(joint_targets):
                    return None
                output_joint_targets = joint_targets[joint_target_index]
            else:
                raise ValueError(f"Invalid joint target type: {type(joint_targets)}")
            return output_joint_targets

        output_joint_targets = get_joint_target_helper(joint_targets, joint_target_index)
        return output_joint_targets

    def get_next_joint_target(self) -> dict | torch.Tensor | None:
        """Get the next joint target from the dataset."""
        joint_target = get_joint_target(self, self._next_joint_target_index)
        if joint_target is not None:
            self._next_joint_target_index += 1
        return joint_target

    EpisodeData.get_next_joint_target = get_next_joint_target

    def get_state(self, state_index) -> dict | None:
        """Get the state of the specified index from the dataset."""
        if "states" not in self._data:
            return None

        states = self._data["states"]

        def get_state_helper(states, state_index) -> dict | torch.Tensor | None:
            if isinstance(states, dict):
                output_state = dict()
                for key, value in states.items():
                    output_state[key] = get_state_helper(value, state_index)
                    if output_state[key] is None:
                        return None
            elif isinstance(states, torch.Tensor):
                if state_index >= len(states):
                    return None
                output_state = states[state_index, None]  # fix here
            elif isinstance(states, list):
                if state_index >= len(states):
                    return None
                output_state = [states[state_index]]
            else:
                raise ValueError(f"Invalid state type: {type(states)}")
            return output_state

        output_state = get_state_helper(states, state_index)
        return output_state

    EpisodeData.get_state = get_state

    def add(self, key: str, value: torch.Tensor | dict):
        """Add a key-value pair to the dataset.

        The key can be nested by using the "/" character.
        For example: "obs/joint_pos".

        Args:
            key: The key name.
            value: The corresponding value of tensor type or of dict type.
        """
        # check datatype
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add(f"{key}/{sub_key}", sub_value)
            return

        sub_keys = key.split("/")
        current_dataset_pointer = self._data
        for sub_key_index in range(len(sub_keys)):
            if sub_key_index == len(sub_keys) - 1:
                # Add value to the final dict layer
                if sub_keys[sub_key_index] not in current_dataset_pointer:
                    current_dataset_pointer[sub_keys[sub_key_index]] = [value.clone()]
                else:
                    current_dataset_pointer[sub_keys[sub_key_index]].append(value.clone())
                break
            # key index
            if sub_keys[sub_key_index] not in current_dataset_pointer:
                current_dataset_pointer[sub_keys[sub_key_index]] = dict()
            current_dataset_pointer = current_dataset_pointer[sub_keys[sub_key_index]]
    EpisodeData.add = add

    def pre_export(self):
        def pre_export_helper(data):
            for key, value in data.items():
                if isinstance(value, list):
                    data[key] = torch.stack(value)
                elif isinstance(value, dict):
                    pre_export_helper(value)
        start_time = time.time()
        pre_export_helper(self._data)
        end_time = time.time()
        print(f"pre_export time: {end_time - start_time:.2f}s")
    EpisodeData.pre_export = pre_export


def patch_step():
    from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

    def step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_pre_physics_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_time_outs = self.termination_manager.time_outs
        self.extras["is_success"] = self.termination_manager._terminated_buf
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            obs_final_buf = self.observation_manager.compute(update_history=False)
            self.extras['final_obs'] = obs_final_buf

            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_buf, self.reset_time_outs, self.extras

    ManagerBasedRLEnv.step = step

    def reset_to_check_state(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids,
        seed: int | None = None,
        is_relative: bool = False,
    ):
        """Resets specified environments to provided states.

        This function resets the environments to the provided states. The state is a dictionary
        containing the state of the scene entities. Please refer to :meth:`InteractiveScene.get_state`
        for the format.

        The function is different from the :meth:`reset` function as it resets the environments to specific states,
        instead of using the randomization events for resetting the environments.

        Args:
            state: The state to reset the specified environments to. Please refer to
                :meth:`InteractiveScene.get_state` for the format.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # reset all envs in the scene if env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # trigger recorder terms for pre-reset calls
        # self.recorder_manager.record_pre_reset(env_ids)

        # set the seed
        if seed is not None:
            self.seed(seed)

        self._reset_idx(env_ids)

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)

        # update articulation kinematics
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # trigger recorder terms for post-reset calls
        # self.recorder_manager.record_post_reset(env_ids)

        # compute observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations
        return self.obs_buf, self.extras

    ManagerBasedRLEnv.reset_to_check_state = reset_to_check_state


YAML_CACHE = {}


def patch_yaml_load():
    import yaml

    orig_safe_load = yaml.safe_load

    def cached_safe_load(stream):
        if isinstance(stream, (str, bytes)):
            cache_key = stream
        else:
            cache_key = stream.name
        # print(f"yaml load {cache_key}")
        if cache_key not in YAML_CACHE:
            # print(f"cache miss {cache_key}")
            YAML_CACHE[cache_key] = orig_safe_load(stream)
        return deepcopy(YAML_CACHE[cache_key])

    yaml.safe_load = cached_safe_load


def patch_reward_manager():
    from isaaclab.managers.reward_manager import RewardManager

    def compute(self, dt: float) -> torch.Tensor:
        """Computes the reward signal as a weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Args:
            dt: The time-step interval of the environment.

        Returns:
            The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value * dt

            # Update current reward for this step.
            self._step_reward[:, term_idx] = value
        # Normalization
        self._reward_buf /= 3
        return self._reward_buf

    RewardManager.compute = compute


def patch_create_teleop_device():
    import isaaclab.devices.teleop_device_factory as teleop_device_factory
    from isaaclab.devices import DeviceBase, DeviceCfg
    from collections.abc import Callable
    from isaaclab.devices.teleop_device_factory import DEVICE_MAP, RETARGETER_MAP
    import inspect
    import omni

    def create_teleop_device(env, device_name: str, devices_cfg: dict[str, DeviceCfg], callbacks: dict[str, Callable] | None = None) -> DeviceBase:
        if device_name not in devices_cfg:
            raise ValueError(f"Device '{device_name}' not found in teleop device configurations")

        device_cfg = devices_cfg[device_name]
        callbacks = callbacks or {}

        # Check if device config type is supported
        cfg_type = type(device_cfg)
        if cfg_type not in DEVICE_MAP:
            raise ValueError(f"Unsupported device configuration type: {cfg_type.__name__}")

        # Get the constructor for this config type
        constructor = DEVICE_MAP[cfg_type]

        # Try to create retargeters if they are configured
        retargeters = []
        if hasattr(device_cfg, "retargeters") and device_cfg.retargeters is not None:
            try:
                # Create retargeters based on configuration
                for retargeter_cfg in device_cfg.retargeters:
                    cfg_type = type(retargeter_cfg)
                    if cfg_type in RETARGETER_MAP:
                        retargeters.append(RETARGETER_MAP[cfg_type](retargeter_cfg))
                    else:
                        raise ValueError(f"Unknown retargeter configuration type: {cfg_type.__name__}")

            except NameError as e:
                raise ValueError(f"Failed to create retargeters: {e}")

        # Check if the constructor accepts retargeters parameter
        constructor_params = inspect.signature(constructor).parameters
        params = {}
        if "retargeters" in constructor_params and retargeters:
            params["retargeters"] = retargeters
        if "env" in constructor_params:
            params["env"] = env

        device = constructor(cfg=device_cfg, **params)

        # Register callbacks
        for key, callback in callbacks.items():
            device.add_callback(key, callback)

        omni.log.info(f"Created teleoperation device: {device_name}")
        return device

    teleop_device_factory.create_teleop_device = create_teleop_device


patch_reset()
patch_configclass()
patch_recorder_manager_ep_meta()
patch_recorder_manager_joint_targets()
patch_step()
patch_yaml_load()
patch_reward_manager()
patch_create_teleop_device()
