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


import logging
import logging.handlers
import os
import queue
import subprocess
import time  # Ensure time module is imported since it's used in main
import traceback
from datetime import datetime  # Import datetime module
from pathlib import Path

import torch

LOG_ROOT_DIR = "lw_benchhub_logs"
CURRENT_DATE_DIR = None
_log_listener = None
_log_queue = None
_backend_handlers = {}
_is_initialized = False

# Log mode configuration
LOG_MODE_OVERWRITE = "overwrite"  # Default: overwrite mode
LOG_MODE_TIMESTAMP = "timestamp"  # Time-isolated mode
_current_log_mode = LOG_MODE_OVERWRITE

# Default logger configurations
DEFAULT_LOG_CONFIG = {
    'log': {'level': logging.DEBUG},
    'vr': {'level': logging.DEBUG},
    'error': {'level': logging.DEBUG}
}


class _LogDispatcher:
    """
    An internal class for QueueListener to receive log records and dispatch them to the correct Handler.
    """

    def __init__(self, handler_map):
        self.handler_map = handler_map  # Stores mapping from Logger name to Handler

    def handle(self, record):
        """
        Receives log records from the queue and passes them to the correct backend Handler.
        """
        # Try to find the Handler by Logger name
        # If not found, try to find Handler for parent Logger
        current_name = record.name
        handler_found = False
        while current_name:
            if current_name in self.handler_map:
                handler = self.handler_map[current_name]
                if record.levelno >= handler.level:  # Ensure record level meets Handler's minimum level
                    handler.handle(record)  # Pass record to dedicated Handler
                handler_found = True
                break
            # Search for parent Logger name
            if '.' in current_name:
                current_name = current_name.rsplit('.', 1)[0]
            else:
                current_name = ''  # Reached root Logger

        # If root Logger has a Handler (name is ''), ensure it also handles
        if not handler_found and '' in self.handler_map:
            root_handler = self.handler_map['']
            if record.levelno >= root_handler.level:
                root_handler.handle(record)


def set_log_mode(mode):
    """
    Set the logging mode.

    Args:
        mode (str): Logging mode. Can be:
            - LOG_MODE_OVERWRITE: Overwrite mode (default) - each run creates fresh logs
            - LOG_MODE_TIMESTAMP: Timestamp mode - creates time-isolated folders
    """
    global _current_log_mode

    if mode not in [LOG_MODE_OVERWRITE, LOG_MODE_TIMESTAMP]:
        raise ValueError(f"Invalid log mode: {mode}. Must be '{LOG_MODE_OVERWRITE}' or '{LOG_MODE_TIMESTAMP}'")

    _current_log_mode = mode
    print(f"Log mode set to: {mode}")


def get_log_mode():
    """
    Get the current logging mode.

    Returns:
        str: Current logging mode
    """
    return _current_log_mode


def configure_log_mode_from_args(args=None, log_mode_arg='log_mode'):
    """
    Configure log mode from command line arguments or configuration.

    Args:
        args: Argument parser object or dictionary containing log mode configuration
        log_mode_arg (str): Name of the argument/parameter for log mode

    Returns:
        str: The configured log mode
    """

    if args is None:
        return _current_log_mode

    # Handle different types of args
    if hasattr(args, log_mode_arg):
        # args is an argparse.Namespace object
        mode = getattr(args, log_mode_arg)
    elif isinstance(args, dict) and log_mode_arg in args:
        # args is a dictionary
        mode = args[log_mode_arg]
    else:
        # No log mode specified, keep current mode
        return _current_log_mode

    # Validate and set mode
    if mode is not None:
        if mode not in [LOG_MODE_OVERWRITE, LOG_MODE_TIMESTAMP]:
            print(f"Warning: Invalid log mode '{mode}'. Using default '{LOG_MODE_OVERWRITE}'")
            mode = LOG_MODE_OVERWRITE

        set_log_mode(mode)
        print(f"Log mode configured from arguments: {mode}")

    return _current_log_mode


def add_log_mode_argument(parser, arg_name='--log-mode', help_text=None):
    """
    Add log mode argument to an argument parser.

    Args:
        parser: ArgumentParser object
        arg_name (str): Name of the argument (default: '--log-mode')
        help_text (str): Help text for the argument

    Returns:
        ArgumentParser: The parser with log mode argument added
    """
    if help_text is None:
        help_text = f"Log mode: '{LOG_MODE_OVERWRITE}' (overwrite logs each run) or '{LOG_MODE_TIMESTAMP}' (time-isolated folders)"

    parser.add_argument(
        arg_name,
        choices=[LOG_MODE_OVERWRITE, LOG_MODE_TIMESTAMP],
        default=LOG_MODE_OVERWRITE,
        help=help_text
    )

    return parser


def _ensure_initialized():
    """Ensure the logging system is initialized."""
    global _is_initialized  # noqa: F824
    if not _is_initialized:
        setup_async_module_logging()


def _create_logger_config(logger_name, level=None, custom_config=None):
    """Create configuration for a specific logger."""
    global CURRENT_DATE_DIR

    # Determine log directory based on mode
    if _current_log_mode == LOG_MODE_OVERWRITE:
        # Overwrite mode: use fixed directory, overwrite existing logs
        log_dir = LOG_ROOT_DIR
    else:  # LOG_MODE_TIMESTAMP
        # Timestamp mode: create time-isolated directories
        if CURRENT_DATE_DIR is None:
            CURRENT_DATE_DIR = os.path.join(LOG_ROOT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_dir = CURRENT_DATE_DIR

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Use custom config if provided, otherwise use default
    if custom_config and 'level' in custom_config:
        log_level = custom_config['level']
    elif level is not None:
        log_level = level
    else:
        log_level = DEFAULT_LOG_CONFIG.get(logger_name, {}).get('level', logging.INFO)

    # Generate filename
    if logger_name == '':
        filename = "root_general.log"
    else:
        filename = f"{logger_name}.log"

    return {
        'file': os.path.join(log_dir, filename),
        'level': log_level
    }


def setup_async_module_logging(custom_config=None, args=None, log_mode_arg='log_mode'):
    """
    Setup the async module logging system.

    Args:
        custom_config (dict, optional): Custom logger configurations.
            Format: {'logger_name': {'level': logging.LEVEL}}
        args: Argument parser object or dictionary containing log mode configuration
        log_mode_arg (str): Name of the argument/parameter for log mode
    """
    global _log_listener, _log_queue, _backend_handlers, _is_initialized, CURRENT_DATE_DIR

    if _is_initialized:
        print("Logging system already initialized.")
        return _log_listener

    # Configure log mode from arguments if provided
    if args is not None:
        configure_log_mode_from_args(args, log_mode_arg)

    # Initialize log directory based on mode
    if _current_log_mode == LOG_MODE_OVERWRITE:
        # Overwrite mode: use fixed directory
        log_dir = LOG_ROOT_DIR
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log mode: OVERWRITE - Logs will be saved to: {log_dir}/")
    else:  # LOG_MODE_TIMESTAMP
        # Timestamp mode: create time-isolated directory
        if CURRENT_DATE_DIR is None:
            CURRENT_DATE_DIR = os.path.join(LOG_ROOT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_dir = CURRENT_DATE_DIR
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log mode: TIMESTAMP - Logs will be saved to: {log_dir}/")

    # --- A. Create central queue ---
    _log_queue = queue.Queue(-1)

    # --- B. Define backend file Handlers and map to Logger names ---
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Merge default and custom configs
    merged_config = DEFAULT_LOG_CONFIG.copy()
    if custom_config:
        merged_config.update(custom_config)

    # Create handlers for all configured loggers
    _backend_handlers = {}
    for logger_name, config in merged_config.items():
        logger_config = _create_logger_config(logger_name, custom_config=config)

        # Choose file mode based on log mode
        if _current_log_mode == LOG_MODE_OVERWRITE:
            file_mode = 'w'  # Overwrite mode: start fresh each time
        else:  # LOG_MODE_TIMESTAMP
            file_mode = 'a'  # Timestamp mode: append to existing files

        file_handler = logging.FileHandler(logger_config['file'], mode=file_mode, encoding='utf-8')
        file_handler.setLevel(logger_config['level'])
        file_handler.setFormatter(formatter)
        _backend_handlers[logger_name] = file_handler

    # Console Handler (not part of async queue, directly added to Logger)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # --- C. Create and start QueueListener ---
    dispatcher = _LogDispatcher(_backend_handlers)
    _log_listener = logging.handlers.QueueListener(_log_queue, dispatcher)
    _log_listener.start()

    # --- D. Configure frontend Loggers ---
    queue_handler_for_all_loggers = logging.handlers.QueueHandler(_log_queue)
    queue_handler_for_all_loggers.setLevel(logging.DEBUG)

    # Configure root Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(queue_handler_for_all_loggers)

    # Configure named Loggers
    for logger_name, config in merged_config.items():
        if logger_name == '':  # Root Logger already handled
            continue
        named_logger = logging.getLogger(logger_name)
        named_logger.setLevel(config.get('level', logging.INFO))
        named_logger.propagate = False
        named_logger.addHandler(queue_handler_for_all_loggers)

    _is_initialized = True
    print(f"Async module logs will be saved to: {CURRENT_DATE_DIR}/")
    return _log_listener


def get_logger(name=None, level=None):
    """
    Get a logger for the specified module.

    Args:
        name (str, optional): Logger name. If None, returns root logger.
        level (int, optional): Logging level. If None, uses default level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    _ensure_initialized()

    if name is None:
        return logging.getLogger()

    # Check if logger already exists and has handlers
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, it means it wasn't configured yet
    # Add it to the configuration and create handler
    if not logger.handlers:
        global _backend_handlers, _log_queue  # noqa: F824

        # Create configuration for this logger
        logger_config = _create_logger_config(name, level)

        # Create file handler
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Choose file mode based on log mode
        if _current_log_mode == LOG_MODE_OVERWRITE:
            file_mode = 'w'  # Overwrite mode: start fresh each time
        else:  # LOG_MODE_TIMESTAMP
            file_mode = 'a'  # Timestamp mode: append to existing files

        file_handler = logging.FileHandler(logger_config['file'], mode=file_mode, encoding='utf-8')
        file_handler.setLevel(logger_config['level'])
        file_handler.setFormatter(formatter)

        # Add to backend handlers
        _backend_handlers[name] = file_handler

        # Configure logger
        logger.setLevel(logger_config['level'])
        logger.propagate = False

        # Add queue handler
        queue_handler = logging.handlers.QueueHandler(_log_queue)
        queue_handler.setLevel(logging.DEBUG)
        logger.addHandler(queue_handler)

    return logger


def add_logger(name, level=logging.INFO):
    """
    Add a new logger configuration.

    Args:
        name (str): Logger name
        level (int): Logging level
    """
    global DEFAULT_LOG_CONFIG  # noqa: F824
    DEFAULT_LOG_CONFIG[name] = {'level': level}


def stop_logging():
    """Stop the logging system and cleanup."""
    global _log_listener, _is_initialized  # noqa: F824
    if _log_listener and _is_initialized:
        _log_listener.stop()
        _is_initialized = False
        print("Logging system stopped.")


def reset_logging_system():
    """Reset the logging system to allow mode changes."""
    global _log_listener, _log_queue, _backend_handlers, _is_initialized, CURRENT_DATE_DIR

    if _is_initialized:
        stop_logging()

    # Reset all global variables
    _log_listener = None
    _log_queue = None
    _backend_handlers = {}
    _is_initialized = False
    CURRENT_DATE_DIR = None

    print("Logging system reset. You can now change log mode and reinitialize.")


def get_default_logger():
    """Get logger for robot control module."""
    return get_logger('log')


def get_general_logger():
    """Get logger for general application logging."""
    return get_logger('general')


def get_vr_logger():
    """Get logger for VR module."""
    return get_logger('vr')


def get_error_logger():
    """Get logger for error module."""
    return get_logger('error')


def setup_async_module_logging_fixed():
    """Legacy function for backward compatibility."""
    return setup_async_module_logging()

# (Original log_scene_rigid_objects function remains unchanged)


def log_scene_rigid_objects(env):
    """
    Retrieves USD path information for all rigid objects in the environment and logs it using async logging.
    """
    # 1. Get all keys from env.scene.rigid_objects
    if not hasattr(env, 'scene') or not hasattr(env.scene, 'rigid_objects') or not isinstance(env.scene.rigid_objects, dict):
        print("Warning: env.scene.rigid_objects is unavailable or not dictionary type. Cannot log rigid object information.")
        return "default_scene_log"

    rigid_objects_dict = env.scene.rigid_objects
    key_list = list(rigid_objects_dict.keys())
    print(f"Detected {len(key_list)} rigid objects.")

    # 2. Record USD path information for each object
    objects_info = []
    for key in key_list:
        obj = rigid_objects_dict[key]
        usd_path = "N/A"
        try:
            if hasattr(obj, 'cfg') and hasattr(obj.cfg, 'spawn') and hasattr(obj.cfg.spawn, 'usd_path'):
                usd_path = obj.cfg.spawn.usd_path
            elif hasattr(obj, 'usd_path'):
                usd_path = obj.usd_path
        except AttributeError:
            usd_path = "Attribute Missing"
        except Exception as e:
            usd_path = f"Error: {e}"

        objects_info.append({
            "name": key,
            "usd_path": str(usd_path)
        })

    # 3. Use async logging to record scene objects
    try:
        scene_logger = get_logger('scene_objects')
        for obj_info in objects_info:
            scene_logger.info(f"Scene Object: {obj_info['name']} -> {obj_info['usd_path']}")
        print(f"Scene objects information logged asynchronously")
    except Exception as e:
        print(f"Error in async scene objects logging: {e}")


# handle_exception_and_log function definition
def handle_exception_and_log():
    """
    Handles exceptions caught in the program and logs them using async logging.
    Uses traceback to get exception information automatically.
    """
    print("An error occurred. Logging error information asynchronously.")

    # Use async logging to record exception
    try:
        error_logger = get_logger('error')
        error_logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error information logged asynchronously")
    except Exception as log_e:
        print(f"Error in async error logging: {log_e}")


def get_code_version():
    """
    Returns a dict of repo (relative dir) -> sha for all .git repos under the cwd.
    Uses git CLI for robustness; falls back to parsing HEAD/ref as last resort.
    """
    code_versions = {}
    root_path = Path.cwd().resolve()
    for git_dir in root_path.rglob('.git'):
        repo_path = git_dir.parent
        sha = None
        # Try 'git rev-parse HEAD' within this repo
        try:
            result = subprocess.run(
                ['git', '--git-dir', str(git_dir), '--work-tree', str(repo_path), 'rev-parse', 'HEAD'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', timeout=3
            )
            sha = result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            pass

        # fallback to parsing HEAD file
        if not sha:
            try:
                git_head_path = git_dir / 'HEAD'
                if git_head_path.exists():
                    with open(git_head_path, 'r') as f:
                        ref = f.readline().strip()
                    if ref.startswith('ref:'):
                        ref_rel = ref.split('ref:')[1].strip()
                        ref_path = git_dir / ref_rel
                        if ref_path.exists():
                            with open(ref_path, 'r') as rf:
                                sha = rf.readline().strip()
                    else:
                        # Detached HEAD, direct SHA
                        sha = ref
            except Exception:
                pass

        try:
            repo_rel_path = str(repo_path.relative_to(root_path))
            if repo_rel_path == ".":
                repo_rel_path = repo_path.name
        except Exception:
            repo_rel_path = str(repo_path)

        if sha:
            code_versions[repo_rel_path] = sha
        else:
            code_versions[repo_rel_path] = 'UNKNOWN'
    return code_versions


def get_camera_metadata(obs_cameras):
    metadata = {}
    for cam_name in obs_cameras.keys():
        metadata[cam_name] = {
            "camera_cfg": obs_cameras[cam_name]["camera_cfg"].to_dict(),
            "using_tags": obs_cameras[cam_name]["tags"],
        }
    return metadata


def get_action_space_def(env):
    from lw_benchhub.utils.usd_utils import OpenUsd as usd
    import lw_benchhub.core.mdp as mdp
    action_def = []
    active_term_names = env.action_manager.active_terms
    robot_prim = usd.get_prim_by_name(name="robot", prim=env.scene.stage.GetPseudoRoot())[0]
    root_name = usd.find_articulation_root(robot_prim).GetName()
    for term_name in active_term_names:
        term_info = {}
        term = env.action_manager.get_term(term_name)
        term_info["action_name"] = term_name
        term_info["action_dim"] = term.action_dim
        term_info["applied_joint_names"] = term._joint_names
        if term._joint_ids == slice(None):
            term_info["applied_joint_ids"] = list(range(len(term._joint_names)))
        else:
            term_info["applied_joint_ids"] = term._joint_ids
        term_info["action_type"] = term.__class__.__name__
        term_info["root_name"] = root_name
        term_info["clip"] = term.cfg.clip
        if type(term.cfg) is mdp.DifferentialInverseKinematicsActionCfg:
            term_info["eef_link_name"] = term.cfg.body_name
            term_info["eef_offset"] = term.cfg.body_offset.to_dict()
            term_info["command_type"] = term.cfg.controller.command_type
            term_info["action_content"] = [f"{term_info['eef_link_name']}_{i}" for i in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']]
            term_info["ik_method"] = term.cfg.controller.ik_method
            term_info["ik_params"] = term.cfg.controller.ik_params
            term_info["is_relative"] = term.cfg.controller.use_relative_mode
            if isinstance(term._scale, torch.Tensor):
                term_info["scale"] = term._scale.cpu().tolist()
            else:
                term_info["scale"] = term._scale
        elif type(term.cfg) is mdp.BinaryJointPositionActionCfg:
            term_info["open_command"] = term._open_command.tolist()
            term_info["close_command"] = term._close_command.tolist()
        elif type(term.cfg) is mdp.RelativeJointPositionActionCfg:
            term_info["scale"] = term.cfg.scale
            term_info["offset"] = term.cfg.offset
            term_info["use_zero_offset"] = term.cfg.use_zero_offset
        elif type(term.cfg) is mdp.JointPositionActionCfg:
            term_info["scale"] = term.cfg.scale
            term_info["offset"] = term.cfg.offset
            term_info["use_default_offset"] = term.cfg.use_default_offset
        elif type(term.cfg) is mdp.G1ActionCfg:
            term_info["preserve_order"] = term.cfg.preserve_order
            term_info["applied_joint_names"] = term.cfg.joint_names
            left_eef_link_name = "left_wrist_yaw_link"
            right_eef_link_name = "right_wrist_yaw_link"
            term_info["action_content"] = [f"{left_eef_link_name}_{i}" for i in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']] + \
                                          [f"{right_eef_link_name}_{i}" for i in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']]
        elif type(term.cfg) is mdp.DexRetargetingActionCfg:
            finger_names = term.finger_names
            term_info["action_content"] = [f"{finger_name}_{i}" for finger_name in finger_names for i in ['x', 'y', 'z']]
        elif type(term.cfg) is mdp.JointPositionMapActionCfg:
            term_info["action_content"] = ["open_degree"]
        elif type(term.cfg) is mdp.G1DecoupledWBCActionCfg:
            left_eef_link_name = "left_wrist_yaw_link"
            right_eef_link_name = "right_wrist_yaw_link"
            term_info["action_content"] = ["left_hand_state", "right_hand_state"] + \
                                          [f"{left_eef_link_name}_{i}" for i in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']] + \
                                          [f"{right_eef_link_name}_{i}" for i in ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']] + \
                                          ["navigate_vel_x", "navigate_vel_y", "navigate_vel_z"] + \
                                          ["base_height_cmd"] + \
                                          ["torso_ori_roll", "torso_ori_pitch", "torso_ori_yaw"]
        elif type(term.cfg) is mdp.LegPositionActionCfg:
            term_info["action_content"] = {
                "squt_mode": ["height_cmd", "unused", "unused", "mode_switch"],
                "loco_mode": ["fb_vel", "lr_vel", "turning_vel", "mode_switch"],
            }

        action_def.append(term_info)

    return action_def


def copy_dict_for_json(orig_dict):
    from lw_benchhub.core.models.fixtures.fixture import Fixture as IsaacFixture
    new_dict = {}
    for (k, v) in orig_dict.items():
        if isinstance(v, dict):
            new_dict[k] = copy_dict_for_json(v)
        elif isinstance(v, IsaacFixture):
            new_dict[k] = v.name
        else:
            new_dict[k] = v
    return new_dict
