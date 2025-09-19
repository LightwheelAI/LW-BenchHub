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

import gymnasium as gym
import importlib
import inspect
import os
import enum
import yaml

from isaaclab.utils.configclass import configclass
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg

from lwlab.utils.usd_utils import OpenUsd as usd_utils
from lwlab.utils.log_utils import get_default_logger
import lwlab.utils.math_utils.transform_utils.numpy_impl as T


class ExecuteMode(enum.Enum):
    """The mode to execute the task."""

    TRAIN = 0
    EVAL = 1
    TELEOP = 2
    REPLAY_JOINT_TARGETS = 3
    REPLAY_ACTION = 4
    REPLAY_STATE = 5


import numpy as np
import torch
import math
EPS = np.finfo(float).eps * 4.0


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )


def load_robocasa_cfg_cls_from_registry(cfg_type: str, cfg_name: str, entry_point_key: str) -> dict | object:
    """Load default configuration given its entry point from the gym registry.

    This function loads the configuration object from the gym registry for the given task name.
    It supports both YAML and Python configuration files.

    It expects the configuration to be registered in the gym registry as:

    .. code-block:: python

        gym.register(
            id="My-Awesome-Task-v0",
            ...
            kwargs={"env_entry_point_cfg": "path.to.config:ConfigClass"},
        )

    The parsed configuration object for above example can be obtained as:

    .. code-block:: python

        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        cfg = load_cfg_from_registry("My-Awesome-Task-v0", "env_entry_point_cfg")

    Args:
        cfg_name: The name of the environment.
        entry_point_key: The entry point key to resolve the configuration file.

    Returns:
        The parsed configuration object. If the entry point is a YAML file, it is parsed into a dictionary.
        If the entry point is a Python class, it is instantiated and returned.

    Raises:
        ValueError: If the entry point key is not available in the gym registry for the task.
    """
    # obtain the configuration entry point
    if not cfg_name:
        return None
    assert cfg_type in ["scene", "task", "robot", "rl"]
    cfg_name = f"Robocasa-{cfg_type.capitalize()}-{cfg_name}"
    cfg_entry_point = gym.spec(cfg_name).kwargs.get(entry_point_key)
    # check if entry point exists
    if cfg_entry_point is None:
        raise ValueError(
            f"Could not find configuration for the environment: '{cfg_name}'."
            f" Please check that the gym registry has the entry point: '{entry_point_key}'."
        )
    # parse the default config file
    if isinstance(cfg_entry_point, str) and cfg_entry_point.endswith(".yaml"):
        if os.path.exists(cfg_entry_point):
            # absolute path for the config file
            config_file = cfg_entry_point
        else:
            # resolve path to the module location
            mod_name, file_name = cfg_entry_point.split(":")
            mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
            # obtain the configuration file path
            config_file = os.path.join(mod_path, file_name)
        # load the configuration
        get_default_logger().info(f"[INFO]: Parsing configuration from: {config_file}")
        with open(config_file, encoding="utf-8") as f:
            cfg = yaml.full_load(f)
    else:
        if callable(cfg_entry_point):
            # resolve path to the module location
            mod_path = inspect.getfile(cfg_entry_point)
            # load the configuration
            cfg_cls = cfg_entry_point()
        elif isinstance(cfg_entry_point, str):
            # resolve path to the module location
            mod_name, attr_name = cfg_entry_point.split(":")
            mod = importlib.import_module(mod_name)
            cfg_cls = getattr(mod, attr_name)

        else:
            cfg_cls = cfg_entry_point
        # load the configuration
        print(f"[INFO]: Parsing configuration from: {cfg_entry_point}")

        if callable(cfg_cls) and "rsl_rl" in str(cfg_cls):
            cfg = cfg_cls()
        else:
            cfg = cfg_cls
    return cfg


def parse_env_cfg(
    scene_name: str,
    robot_name: str,
    task_name: str,
    robot_scale: float,
    execute_mode: ExecuteMode,
    for_rl: bool = False,
    rl_variant: str = None,
    device: str = "cuda:0", num_envs: int | None = None, use_fabric: bool | None = None,
    replay_cfgs: dict | None = None,
    first_person_view: bool = False,
    enable_cameras: bool = False,
    usd_simplify: bool = False,
    object_init_offset: list[float] = [0.0, 0.0],
    max_scene_retry: int = 5,
    max_object_placement_retry: int = 3,
    seed: int | None = None,
    **kwargs,
) -> ManagerBasedRLEnvCfg | DirectRLEnvCfg:
    """Parse configuration for an environment and override based on inputs.

    Args:
        task_name: The name of the environment.
        device: The device to run the simulation on. Defaults to "cuda:0".
        num_envs: Number of environments to create. Defaults to None, in which case it is left unchanged.
        use_fabric: Whether to enable/disable fabric interface. If false, all read/write operations go through USD.
            This slows down the simulation but allows seeing the changes in the USD through the USD stage.
            Defaults to None, in which case it is left unchanged.

    Returns:
        The parsed configuration object.

    Raises:
        RuntimeError: If the configuration for the task is not a class. We assume users always use a class for the
            environment configuration.
    """
    # import_all_inits(os.path.join(ISAAC_ROBOCASA_ROOT, './tasks/_APIs'))
    from isaaclab_tasks.utils import import_packages
    # Import all configs in this package
    import_packages(
        "lwlab.core",
        # The blacklist is used to prevent importing configs from sub-packages
        blacklist_pkgs=["utils", ".mdp", ".devices"]
    )
    import_packages("tasks")
    if for_rl:
        import_packages("lwlab_rl")
    if scene_name.endswith(".usd"):
        scene_type = "USD"
    else:
        scene_type, *_ = scene_name.split("-", 1)
    scene_env_cfg = load_robocasa_cfg_cls_from_registry("scene", scene_type.capitalize(), "env_cfg_entry_point")

    if not for_rl:
        if scene_type == "USD":
            task_env_cfg = load_robocasa_cfg_cls_from_registry("task", "Usd", "env_cfg_entry_point")
        else:
            task_env_cfg = load_robocasa_cfg_cls_from_registry("task", task_name, "env_cfg_entry_point")
        robot_env_cfg = load_robocasa_cfg_cls_from_registry("robot", robot_name, "env_cfg_entry_point")

        @configclass
        class RobocasaEnvCfg(robot_env_cfg, task_env_cfg, scene_env_cfg):
            pass
    else:
        if rl_variant:
            task_name = f"{task_name}-{rl_variant}"
        rl_robot_task_cfg = load_robocasa_cfg_cls_from_registry("rl", f"{robot_name}-{task_name}", "env_cfg_entry_point")

        @configclass
        class RobocasaEnvCfg(rl_robot_task_cfg, scene_env_cfg):
            pass

    # set num_envs in task_env_cfg
    if num_envs is not None:
        RobocasaEnvCfg.num_envs = num_envs
    RobocasaEnvCfg.device = device

    if replay_cfgs is not None:
        RobocasaEnvCfg.replay_cfgs = replay_cfgs
    RobocasaEnvCfg.first_person_view = first_person_view
    RobocasaEnvCfg.usd_simplify = usd_simplify
    RobocasaEnvCfg.enable_cameras = enable_cameras
    RobocasaEnvCfg.object_init_offset = object_init_offset

    RobocasaEnvCfg.max_scene_retry = max_scene_retry
    RobocasaEnvCfg.max_object_placement_retry = max_object_placement_retry

    if scene_type == "USD":
        cfg = RobocasaEnvCfg(
            execute_mode=execute_mode,
            usd_path=scene_name,
            robot_scale=robot_scale,
            seed=seed,
        )
    else:
        cfg = RobocasaEnvCfg(
            execute_mode=execute_mode,
            scene_name=scene_name,
            robot_scale=robot_scale,
            seed=seed,
        )

    # check that it is not a dict
    # we assume users always use a class for the configuration
    if isinstance(cfg, dict):
        raise RuntimeError(f"Configuration for the task: '{task_name}' is not a class. Please provide a class.")

    if execute_mode == ExecuteMode.REPLAY_JOINT_TARGETS:
        cfg.set_replay_joint_targets_action()
    # simulation device
    cfg.sim.device = device
    # disable fabric to read/write through USD
    if use_fabric is not None:
        cfg.sim.use_fabric = use_fabric
    # number of environments
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    if replay_cfgs and "ep_meta" in replay_cfgs and "sim_args" in replay_cfgs["ep_meta"]:
        cfg.sim.dt = replay_cfgs["ep_meta"]["sim_args"]["dt"]
        cfg.sim.decimation = replay_cfgs["ep_meta"]["sim_args"]["decimation"]
        # cfg.sim.render_interval = replay_cfgs["ep_meta"]["sim_args"]["render_interval"]
        # cfg.scene.num_envs = replay_cfgs["ep_meta"]["sim_args"]["num_envs"]
    return cfg


def set_camera_follow_pose(env, offset, lookat):
    if env.cfg.robot_base_link is not None:
        robot_base_link_idx = env.scene.articulations["robot"].data.body_names.index(env.cfg.robot_base_link)
        robot_mat = quat2mat(env.scene.articulations["robot"].data.body_com_quat_w[..., robot_base_link_idx, :][0].cpu().numpy()[[1, 2, 3, 0]])
        robot_pos = env.scene.articulations["robot"].data.body_com_pos_w[..., robot_base_link_idx, :][0].cpu().numpy()
        robot_pos += (robot_mat @ np.array(offset).reshape(3, 1))[:, 0]
        robot_lookat = robot_pos + (robot_mat @ np.array(lookat).reshape(3, 1))[:, 0]
    else:
        robot_mat = quat2mat(env.scene.articulations["robot"].data.body_com_quat_w[..., 0, :][0].cpu().numpy()[[1, 2, 3, 0]])
        robot_pos = env.scene.articulations["robot"].data.body_com_pos_w[..., 0, :][0].cpu().numpy()
        robot_pos += (robot_mat @ np.array(offset).reshape(3, 1))[:, 0]
        robot_lookat = robot_pos + (robot_mat @ np.array(lookat).reshape(3, 1))[:, 0]

    env.sim.set_camera_view(robot_pos, robot_lookat)


def set_robot_to_position(env, global_pos, global_ori=None, keep_z=True, env_ids=None):
    """
    Set robot root pose directly using position (x, y, z) and quaternion (w, x, y, z) in world coordinates.

    Args:
        env: The environment object.
        global_pos: Sequence of 3 floats [x, y, z] in world frame.
        global_ori: Sequence of 3 floats [roll, pitch, yaw] in world frame, if not provided, use the anchor orientation.
        keep_z: Whether to keep the z-coordinate of the robot's current position.
        env_ids: Tensor of environment indices, or None for all.
    """
    if env_ids is None:
        # default to all environments
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.int64)
    if keep_z:
        robot_z = env.scene.articulations["robot"].data.root_pos_w[0, 2]
    else:
        robot_z = global_pos[2]
    robot_pos = torch.tensor([[global_pos[0], global_pos[1], robot_z]], dtype=torch.float32, device=env.device) + env.scene.env_origins[env_ids]
    robot_ori = global_ori if global_ori is not None else env.cfg.init_robot_base_ori_anchor
    robot_quat = T.convert_quat(T.mat2quat(T.euler2mat(robot_ori)), "wxyz")
    robot_quat = torch.tensor(robot_quat, dtype=torch.float32, device=env.device).unsqueeze(0).repeat(env_ids.shape[0], 1)
    robot_pose = torch.concat([robot_pos, robot_quat], dim=-1)
    env.scene.articulations["robot"].write_root_pose_to_sim(robot_pose, env_ids=env_ids)
    env.sim.forward()


def setup_task_description_ui(env_cfg, env):
    """
    Set up UI for displaying task description in the overlay window.

    Args:
        desc (str): Description of the task
        env: Environment object

    Returns:
        tuple: (overlay_window, warning_label) for updating motion warnings
    """
    desc = None
    base_desc = None

    if hasattr(env_cfg, 'task_name') and hasattr(env_cfg, 'layout_id') and hasattr(env_cfg, 'style_id') and hasattr(env_cfg, 'get_ep_meta'):
        base_desc = "Task name: {}\nLayout id: {}\nStyle id: {}\nDesc: {}".format(env_cfg.task_name, env_cfg.layout_id, env_cfg.style_id, env_cfg.get_ep_meta()["lang"])
    elif hasattr(env_cfg, 'task_name') and hasattr(env_cfg, 'usd_path') and hasattr(env_cfg, 'get_ep_meta'):
        base_desc = "Task name: {}\nUSD path: {}\nDesc: {}".format(env_cfg.task_name, env_cfg.usd_path, env_cfg.get_ep_meta()["lang"])

    if base_desc is not None:
        desc = base_desc + "\n Checkpoints: not saved"

    if desc is None:
        return None, None

    import omni.ui as ui

    # Setup overlay window
    main_viewport = env.sim._viewport_window
    main_viewport.dock_tab_bar_visible = False
    env.sim.render()

    overlay_window = ui.Window(
        main_viewport.name,
        width=0,
        height=0,
        flags=ui.WINDOW_FLAGS_NO_TITLE_BAR |
        ui.WINDOW_FLAGS_NO_SCROLLBAR |
        ui.WINDOW_FLAGS_NO_RESIZE
    )
    env.sim.render()

    with overlay_window.frame:
        with ui.ZStack():
            ui.Spacer()
            with ui.VStack(style={"margin": 15}):
                # Task description (green text)
                task_desc_label = ui.Label(
                    desc,
                    alignment=ui.Alignment.LEFT_TOP,
                    style={
                        "color": 0xFF00FF00,    # Green
                        "font_size": 28,
                        "background_color": 0x00000080,  # Semi-transparent black
                        "padding": 6,
                        "border_radius": 4,
                    }
                )

                # Motion warnings (red text) - initially empty, positioned closer to desc
                warning_label = ui.Label(
                    "",
                    alignment=ui.Alignment.LEFT_TOP,
                    style={
                        "color": 0xFF0000FF,    # Red (corrected format)
                        "font_size": 24,
                        "background_color": 0x00000080,  # Semi-transparent black
                        "padding": 6,
                        "border_radius": 4
                    }
                )

    env.sim.render()

    # Expose the label/base text for minimal external updates without changing return signature
    try:
        setattr(env, "_task_desc_label", task_desc_label)
        setattr(env, "_warning_label", warning_label)
        setattr(env, "_base_desc", base_desc if base_desc is not None else desc)
    except Exception:
        pass

    return overlay_window


def update_motion_warnings(env, warning_text):
    """
    Update the motion warning label with new warning text.

    Args:
        warning_label: The UI label to update
        warning_text: The warning text to display
    """
    if env._warning_label and warning_text:
        env._warning_label.text = warning_text
    elif env._warning_label:
        env._warning_label.text = ""


def dock_window(space, name, location, ratio):
    """
    Dock a window in the specified space with the given name, location, and size ratio.

    Args:
        space: The workspace to dock the window in.
        name: The name of the window to dock.
        location: The docking position.
        ratio: Size ratio for the docked window.
    """
    import omni.ui as ui
    window = ui.Workspace.get_window(name)
    if window and space:
        window.dock_in(space, location, ratio=ratio)


def create_and_dock_viewport(env, parent_window_name, position, ratio, camera_path):
    """
    Create and configure a viewport window.

    Args:
        env: Environment object
        parent_window: Parent window to dock this viewport to
        position: Docking position
        ratio: Size ratio for the docked window
        camera_path: Prim path to the camera to set as active

    Returns:
        The created viewport window
    """
    from omni.kit.viewport.utility import create_viewport_window
    import omni.ui as ui
    viewport = create_viewport_window()
    env.sim.render()

    parent_window = ui.Workspace.get_window(parent_window_name)
    dock_window(parent_window, viewport.name, position, ratio)
    env.sim.render()

    viewport.viewport_api.set_active_camera(camera_path)
    viewport.viewport_api.set_texture_resolution((426, 240))

    env.sim.render()

    return viewport


# TODO to optimize this function
#  1. enable setup cameras with config
#  2. try to optimize camera config to increase performance
def setup_cameras(env):
    """
    Set up mulitiple viewports for the teleoperation.

    Args:
        env: Environment object
    Returns:
        viewports: Dictionary of created viewports with their names as keys
    """
    from pxr import UsdGeom
    import omni.ui as ui
    viewports = {}
    camera_prims = []
    for prim in env.sim.stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_prims.append(prim)
    left_hand_camera, right_hand_camera, left_shoulder_camera, right_shoulder_camera, eye_in_hand_camera = None, None, None, None, None
    for camera_prim in camera_prims:
        name = camera_prim.GetName().lower()
        if camera_prim.GetName().lower() == "left_hand_camera":
            left_hand_camera = camera_prim
        elif camera_prim.GetName().lower() == "right_hand_camera":
            right_hand_camera = camera_prim
        elif camera_prim.GetName().lower() == "left_shoulder_camera":
            left_shoulder_camera = camera_prim
        elif camera_prim.GetName().lower() == "right_shoulder_camera":
            right_shoulder_camera = camera_prim
        elif camera_prim.GetName().lower() == "eye_in_hand_camera":
            eye_in_hand_camera = camera_prim
    if eye_in_hand_camera is not None:
        viewport_eye_in_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.BOTTOM,
            0.25,
            eye_in_hand_camera.GetPath()
        )
        viewports["eye_in_hand"] = viewport_eye_in_hand
    if left_hand_camera is not None:
        viewport_left_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.LEFT,
            0.25,
            left_hand_camera.GetPath()
        )
        viewports["left_hand"] = viewport_left_hand
    if right_hand_camera is not None:
        viewport_right_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.RIGHT,
            0.25,
            right_hand_camera.GetPath()
        )
        viewports["right_hand"] = viewport_right_hand
    # if left_shoulder_camera is not None:
        # viewport_left_shoulder = create_and_dock_viewport(
        #     env,
        #     viewport_left_hand.name,
        #     ui.DockPosition.BOTTOM,
        #     0.5,
        #     left_shoulder_camera.GetPath()
        # )
        # viewports["left_shoulder"] = viewport_left_shoulder
    # if right_shoulder_camera is not None:
        # viewport_right_shoulder = create_and_dock_viewport(
        #     env,
        #     viewport_right_hand.name,
        #     ui.DockPosition.BOTTOM,
        #     0.5,
        #     right_shoulder_camera.GetPath())
        # viewports["right_shoulder"] = viewport_right_shoulder
    return viewports


def spawn_cylinder_with_xform(
    parent_prim_path,
    xform_name,
    cylinder_name,
    cfg,
    env,
):
    from pxr import UsdGeom, Sdf, Gf, UsdShade
    stage = env.sim.stage

    xform_path = f"{parent_prim_path}/{xform_name}"
    xform_prim = stage.GetPrimAtPath(xform_path)
    if xform_prim and xform_prim.IsValid():
        return xform_prim

    xform = UsdGeom.Xform.Define(stage, Sdf.Path(xform_path))

    xform.AddTranslateOp().Set(Gf.Vec3f(*cfg["translation"]))
    xform.AddOrientOp().Set(Gf.Quatf(*cfg["orientation"]))

    cyl_path = f"{xform_path}/{cylinder_name}"
    cyl = UsdGeom.Cylinder.Define(stage, Sdf.Path(cyl_path))
    cyl.CreateRadiusAttr(cfg["spawn"].radius)
    cyl.CreateHeightAttr(cfg["spawn"].height)
    cyl.CreateAxisAttr(cfg["spawn"].axis)

    material_path = f"{xform_path}/{cylinder_name}_Material"
    material = UsdShade.Material.Define(stage, Sdf.Path(material_path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{material_path}/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*cfg["color"]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    surface_output = material.CreateSurfaceOutput()
    shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    surface_output.ConnectToSource(shader_output)

    UsdShade.MaterialBindingAPI(cyl).Bind(material)

    return xform


def spawn_robot_vis_helper_general(env):
    # check if the robot_vis_helper_cfg is available
    if not hasattr(env.cfg, "robot_vis_helper_cfg"):
        return

    vis_helper_prims = []

    for prim in env.sim.stage.Traverse():
        if prim.GetName().lower() == "robot":
            robot_prim_path = prim.GetPath()
    for key, cfg in env.cfg.robot_vis_helper_cfg.items():
        prim_path = robot_prim_path.AppendPath(cfg["relative_prim_path"])
        cylinder_prim = spawn_cylinder_with_xform(
            parent_prim_path=prim_path,
            xform_name=key,
            cylinder_name="mesh",
            cfg=cfg,
            env=env,
        )
        vis_helper_prims.append(cylinder_prim)
    return vis_helper_prims


def spawn_robot_vis_helper(env):
    # Have problems with Isaaclab/IsaacSim 4.5, works fine with Isaaclab/IsaacSim 5.0
    # check if the robot_vis_helper_cfg is available
    if not hasattr(env.cfg, "robot_vis_helper_cfg"):
        return
    import isaaclab.sim as sim_utils

    robot_prim = None
    vis_helper_prims = []

    for prim in env.sim.stage.Traverse():
        if prim.GetName().lower() == "robot":
            robot_prim = prim
            robot_prim_path = prim.GetPath()
    for key, cfg in env.cfg.robot_vis_helper_cfg.items():
        prim_path = robot_prim_path.AppendPath(cfg["relative_prim_path"])
        prim = sim_utils.spawn_cylinder(prim_path, cfg['spawn'], translation=cfg['translation'], orientation=cfg['orientation'])
        vis_helper_prims.append(prim)
    return vis_helper_prims


def destroy_robot_vis_helper(prim_list, env):
    if not prim_list:
        return
    for prim in prim_list:
        if prim.GetPrim().IsValid():
            env.sim.stage.RemovePrim(prim.GetPath())


def generate_random_robot_pos(anchor_pos, anchor_ori, pos_dev_x, pos_dev_y):
    local_deviation = np.random.uniform(
        low=(-pos_dev_x, -pos_dev_y),
        high=(pos_dev_x, 0.0),
    )
    local_deviation = np.concatenate((local_deviation, [0.0]))
    global_deviation = np.matmul(
        T.euler2mat(anchor_ori + [0, 0, np.pi / 2]), -local_deviation
    )
    return anchor_pos + global_deviation


def get_safe_robot_anchor(env, unsafe_anchor_pos, unsafe_anchor_ori):
    """
    Takes the default "unsafe" anchor from robocasa and corrects it
    by moving it backwards based on the robot's arm reach to ensure safety.

    Args:
        env: The environment object.
        unsafe_anchor_pos (np.array): The original anchor position from robocasa.
        unsafe_anchor_ori (np.array): The original anchor orientation from robocasa.

    Returns:
        tuple(np.array, np.array): The new, safer anchor position and orientation.
    """
    # Calculate the required retreat distance based on arm reach
    try:
        left_offset = env.cfg.offset_config["left_offset"]
        left2arm_transform = env.cfg.offset_config["left2arm_transform"]
        right_offset = env.cfg.offset_config["right_offset"]
        right2arm_transform = env.cfg.offset_config["right2arm_transform"]

        left_ee_pos = left2arm_transform[0:3, 3] + left_offset
        right_ee_pos = right2arm_transform[0:3, 3] + right_offset

        robot_forward_reach = max(left_ee_pos[1], right_ee_pos[1])

        retreat_distance = robot_forward_reach + 0.05  # Add a small 5cm extra margin

    except (AttributeError, KeyError):
        # Fallback if config is not available
        retreat_distance = 0

    local_retreat_vector = np.array([0, -retreat_distance, 0])

    # Rotate this local vector into the global frame using the robot's orientation
    global_retreat_vector = np.matmul(
        T.euler2mat(unsafe_anchor_ori), local_retreat_vector
    )
    safe_anchor_pos = unsafe_anchor_pos + global_retreat_vector

    return safe_anchor_pos, unsafe_anchor_ori


def check_overlap(bbox1, bbox2):
    min1 = bbox1.GetMin()
    max1 = bbox1.GetMax()
    min2 = bbox2.GetMin()
    max2 = bbox2.GetMax()

    # check overlap along x axis
    overlaps_x = (min1[0] < max2[0]) and (max1[0] > min2[0])
    # check overlap along y axis
    overlaps_y = (min1[1] < max2[1]) and (max1[1] > min2[1])
    # check overlap along z axis
    overlaps_z = (min1[2] < max2[2]) and (max1[2] > min2[2])

    # return True if all three axes overlap
    return overlaps_x and overlaps_y and overlaps_z


def calculate_robot_bbox(env, robot_pos, arm_margin=0.08, floor_margin=0.1, env_ids=None):
    """
    Calculate the bounding box of the robot in the environment.

    Args:
        env: The environment object.
        robot_pos: The robot position.
        arm_margin: The margin for the robot's arm.
        floor_margin: The margin for the floor.
        env_ids: The environment IDs to calculate the bounding box for.

    Returns:
        overall_robot_bbox: The overall bounding box of the robot.
    """
    from pxr import Gf, UsdGeom, Usd

    # convert margin to the abs value
    arm_margin = abs(arm_margin)
    floor_margin = abs(floor_margin)

    robot_prim = None

    for prim in env.sim.stage.Traverse():
        if prim.GetName().lower() == "robot":
            robot_prim = prim

    robot_bbox = usd_utils.get_prim_aabb_bounding_box(robot_prim)
    min_point = robot_bbox.GetMidpoint()

    new_center = Gf.Vec3d(*robot_pos)

    diff = new_center - min_point
    new_min = robot_bbox.GetMin() + Gf.Vec3d(diff[0], diff[1], 0)
    new_max = robot_bbox.GetMax() + Gf.Vec3d(diff[0], diff[1], 0)

    floor_margin_vec = Gf.Vec3d(0, 0, floor_margin)  # floor margin in the z-axis
    arm_margin_vec = Gf.Vec3d(arm_margin, arm_margin, 0)  # arm margin in the x and y-axis
    new_bbox = Gf.Range3d(new_min + floor_margin_vec - arm_margin_vec, new_max + arm_margin_vec)

    return new_bbox


def detect_robot_out_of_scene(robot_bbox, scene_bbox):
    """
    Detect if the robot is out of the scene.

    Args:
        robot_bbox: The robot bounding box.
        scene_bbox: The scene bounding box.

    Returns:
        bool: True if the robot is out of the scene, False otherwise.
    """

    # Check if the robot bounding box is outside the scene bounding box, ignore the z-axis
    # since the robot can be above the scene
    robot_bbox_min = robot_bbox.GetMin()
    robot_bbox_max = robot_bbox.GetMax()
    scene_bbox_min = scene_bbox.GetMin()
    scene_bbox_max = scene_bbox.GetMax()

    return not (scene_bbox_min[0] < robot_bbox_min[0] and
                scene_bbox_max[0] > robot_bbox_max[0] and
                scene_bbox_min[1] < robot_bbox_min[1] and
                scene_bbox_max[1] > robot_bbox_max[1])


def check_valid_robot_pose(env, robot_pos, env_ids=None):
    """
    Check if the robot pose is valid.

    Args:
        env: The environment object.
        robot_pos: The robot position.
        env_ids: The environment IDs to check the robot pose for.

    Returns:
        bool: True if the robot pose is valid, False otherwise.
    """
    # TODO: disable collision check
    # 1. check if the robot is in collision with the environment
    # if detect_robot_collision(env, env_ids=env_ids):
    #     print(f"Robot pose: {robot_pos} is in collision with the environment")
    #     return False

    # 2. check if the robot is out of the scene
    robot_bbox = calculate_robot_bbox(env, robot_pos)
    scene_prim = None
    for prim in env.sim.stage.Traverse():
        if prim.IsValid() and prim.GetName().lower() == "scene":
            scene_prim = prim
    scene_bbox = usd_utils.get_prim_aabb_bounding_box(scene_prim) if scene_prim else None

    if scene_bbox is not None:
        if detect_robot_out_of_scene(robot_bbox, scene_bbox):
            print(f"Robot pose: {robot_pos} is out of the scene bounds: {scene_bbox.GetMin()} - {scene_bbox.GetMax()}")
            return False

    # 3. check if the robot is in overlap with the objects in the scene
    # all rigid prim in the scene except the robot
    for obs_prim in scene_prim.GetChildren():
        if not obs_prim.IsValid():
            continue
        # Compute the world-space bounding box for prim
        obs_bbox = usd_utils.get_prim_aabb_bounding_box(obs_prim)

        # Check if the robot bounding box overlaps with the current object bounding box
        if check_overlap(robot_bbox, obs_bbox):
            get_default_logger().info(f"Collision detected between robot and object: {obs_prim.GetPath()}")
            get_default_logger().info(f"Robot BBox: {robot_bbox.GetMin()} - {robot_bbox.GetMax()}")
            get_default_logger().info(f"Robot Size: {robot_bbox.GetSize()}")
            get_default_logger().info(f"Object BBox: {obs_bbox.GetMin()} - {obs_bbox.GetMax()}")
            get_default_logger().info(f"Object Prim Size: {obs_bbox.GetSize()}")
            return False

    return True


def detect_robot_collision(env, env_ids=None):
    # check if base contact is available
    if "base_contact" in env.scene.sensors:
        robot_contact = env.scene.sensors["base_contact"].data.net_forces_w[env_ids]
        return torch.all(torch.max(robot_contact, dim=-1).values > 0.0)
    else:
        return False


def sample_robot_base(
    env,
    anchor_pos,
    anchor_ori,
    rot_dev,
    pos_dev_x,
    pos_dev_y,
    env_ids=None,
    execute_mode=ExecuteMode.TELEOP,
):
    # random_rot = env.rng.uniform(-rot_dev, rot_dev)
    # random_quat = T.convert_quat(T.mat2quat(T.euler2mat(np.array([0, 0, random_rot]))), "wxyz")
    # init_pose_copy = env.scene.articulations["robot"].data.root_state_w[:, :7]

    found_valid = False

    if execute_mode in (ExecuteMode.REPLAY_ACTION, ExecuteMode.REPLAY_JOINT_TARGETS, ExecuteMode.REPLAY_STATE, ExecuteMode.EVAL):
        return anchor_pos

    cur_dev_pos_x = pos_dev_x
    cur_dev_pos_y = pos_dev_y
    while not found_valid:
        for attempt_position in range(50):
            robot_pos = generate_random_robot_pos(
                anchor_pos=anchor_pos,
                anchor_ori=anchor_ori,
                pos_dev_x=cur_dev_pos_x,
                pos_dev_y=cur_dev_pos_y,
            )
            set_robot_to_position(env, robot_pos, env_ids=env_ids)
            if check_valid_robot_pose(env, robot_pos, env_ids=env_ids):
                found_valid = True
                break
            # env.scene.articulations["robot"].write_root_pose_to_sim(init_pose_copy)
        # if valid position not found, increase range by 10 cm for x and 5 cm for y
        cur_dev_pos_x += 0.10
        cur_dev_pos_y += 0.05
    return robot_pos


from collections import deque


class ContactQueue:
    def __init__(self):
        self.queue = deque()

    def is_empty(self):
        return len(self.queue) == 0

    def add(self, contact_view):
        self.queue.append(contact_view)

    def pop(self):
        if self.is_empty():
            return None
        contact_view = self.queue.popleft()
        self.queue.append(contact_view)
        return contact_view

    def clear(self):
        self.queue.clear()


from . import monkey_patch
