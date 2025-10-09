from lwlab.core.checks.checker_factory import get_checkers_from_cfg


def get_task_desc(env_cfg):
    """
    Get the task description from the environment configuration.
    """
    base_desc = ""
    if hasattr(env_cfg, 'task_name') and hasattr(env_cfg, 'layout_id') and hasattr(env_cfg, 'style_id') and hasattr(env_cfg, 'get_ep_meta'):
        base_desc = "Task name: {}\nLayout id: {}\nStyle id: {}\nDesc: {}".format(env_cfg.task_name, env_cfg.layout_id, env_cfg.style_id, env_cfg.get_ep_meta()["lang"])
    elif hasattr(env_cfg, 'task_name') and hasattr(env_cfg, 'usd_path') and hasattr(env_cfg, 'get_ep_meta'):
        base_desc = "Task name: {}\nUSD path: {}\nDesc: {}".format(env_cfg.task_name, env_cfg.usd_path, env_cfg.get_ep_meta()["lang"])
    return base_desc


def setup_task_description_ui(env_cfg, env):
    """
    Set up UI for displaying task description in the overlay window.

    Args:
        env_cfg: Configuration for the environment.
        env: Environment object.

    Returns:
        tuple: (overlay_window, warning_label) for updating motion warnings.
    """
    import omni.ui as ui

    desc = None
    base_desc = get_task_desc(env_cfg)

    if base_desc is not None:
        desc = base_desc + "\nCheckpoints: not saved"

    if desc is None:
        return None, None

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

    checker_labels = {}
    checkers = None

    if hasattr(env_cfg, "checkers_cfg") and get_checkers_from_cfg is not None:
        try:
            filtered_cfg = {
                name: cfg
                for name, cfg in env_cfg.checkers_cfg.items()
                if cfg.get("warning_on_screen", False)
            }
            if filtered_cfg:
                checkers = get_checkers_from_cfg(filtered_cfg)
            else:
                checkers = None
        except Exception as e:
            print(f"Error getting checkers: {e}")
            checkers = None

    with overlay_window.frame:
        with ui.ZStack():
            ui.Spacer()
            with ui.VStack(style={"margin": 15}, spacing=5, alignment=ui.Alignment.LEFT_TOP):
                task_desc_label = ui.Label(
                    desc,
                    alignment=ui.Alignment.LEFT_TOP,
                    style={"color": 0xB300FF00,
                           "font_size": 18,
                           "background_color": 0x00000080,
                           "padding": 6,
                           "border_radius": 4}
                )

                warning_label = ui.Label(
                    "",
                    alignment=ui.Alignment.LEFT_TOP,
                    style={"color": 0xFF0000FF,
                           "font_size": 18,
                           "background_color": 0x00000080,
                           "padding": 6,
                           "border_radius": 4}
                )

                if checkers:
                    for checker in checkers:
                        label = ui.Label(
                            checker.type,
                            alignment=ui.Alignment.LEFT_TOP,
                            style={
                                "color": 0xB300FF00,
                                "font_size": 18,
                                "background_color": 0x00000080,
                                "padding": 6,
                                "border_radius": 4
                            }
                        )
                        checker_labels[checker.type] = label

    env.sim.render()

    try:
        setattr(env, "_task_desc_label", task_desc_label)
        setattr(env, "_warning_label", warning_label)
        setattr(env, "_checker_labels", checker_labels)
        setattr(env, "_base_desc", base_desc if base_desc is not None else desc)
    except Exception as e:
        print(f"Error setting attributes on env: {e}")
        pass

    return overlay_window


def update_task_desc(env, env_cfg):
    """
    Update the motion warning label with new warning text.

    Args:
        warning_label: The UI label to update
        warning_text: The warning text to display
    """
    if env._task_desc_label:
        env._task_desc_label.text = get_task_desc(env_cfg)


def update_checkers_status(env, warning_text: str):
    """
    Updates the color of checker labels based on the warning text.

    Args:
        env: Environment object containing the UI elements.
        warning_text (str): A string containing the names of checkers that have warnings.
    """
    if not hasattr(env, "_checker_labels"):
        return

    for checker_name, label in env._checker_labels.items():
        if warning_text and checker_name in warning_text:
            label.style = {**label.style, "color": 0xFF000080}
        else:
            label.style = {**label.style, "color": 0xB300FF00}


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


def create_and_dock_viewport(env, parent_window_name, position, ratio, camera_path, viewport=None):
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
    if viewport is None:
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
def setup_cameras(env, viewports=None):
    """
    Set up mulitiple viewports for the teleoperation.

    Args:
        env: Environment object
    Returns:
        viewports: Dictionary of created viewports with their names as keys
    """
    from pxr import UsdGeom
    import omni.ui as ui
    if viewports is None:
        viewports = {}
    camera_prims = []
    for prim in env.sim.stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            camera_prims.append(prim)
    left_hand_camera, right_hand_camera, left_shoulder_camera, right_shoulder_camera, eye_in_hand_camera, first_person_camera = None, None, None, None, None, None
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
        elif camera_prim.GetName().lower() == "first_person_camera":
            first_person_camera = camera_prim
    if first_person_camera is not None:
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_viewport_from_window_name()
        viewport.set_active_camera(first_person_camera.GetPath())
        viewport.set_texture_resolution((1280, 720))
    if eye_in_hand_camera is not None:
        viewport_eye_in_hand = viewports.get("eye_in_hand", None)
        viewport_eye_in_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.BOTTOM,
            0.25,
            eye_in_hand_camera.GetPath(),
            viewport=viewport_eye_in_hand

        )
        viewports["eye_in_hand"] = viewport_eye_in_hand
    if left_hand_camera is not None:
        viewport_left_hand = viewports.get("left_hand", None)
        viewport_left_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.LEFT,
            0.25,
            left_hand_camera.GetPath(),
            viewport=viewport_left_hand
        )
        viewports["left_hand"] = viewport_left_hand
    if right_hand_camera is not None:
        viewport_right_hand = viewports.get("right_hand", None)
        viewport_right_hand = create_and_dock_viewport(
            env,
            "DockSpace",
            ui.DockPosition.RIGHT,
            0.25,
            right_hand_camera.GetPath(),
            viewport=viewport_right_hand
        )
        viewports["right_hand"] = viewport_right_hand
    if left_shoulder_camera is not None:
        viewport_left_shoulder = viewports.get("left_shoulder", None)
        viewport_left_shoulder = create_and_dock_viewport(
            env,
            viewport_left_hand.name,
            ui.DockPosition.BOTTOM,
            0.5,
            left_shoulder_camera.GetPath(),
            viewport=viewport_left_shoulder
        )
        viewports["left_shoulder"] = viewport_left_shoulder
    if right_shoulder_camera is not None:
        viewport_right_shoulder = viewports.get("right_shoulder", None)
        viewport_right_shoulder = create_and_dock_viewport(
            env,
            viewport_right_hand.name,
            ui.DockPosition.BOTTOM,
            0.5,
            right_shoulder_camera.GetPath(),
            viewport=viewport_right_shoulder
        )
        viewports["right_shoulder"] = viewport_right_shoulder
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


def hide_ui_windows(sim_app):
    hide_window_names = []
    hide_window_names.extend(
        [
            "Console",
            "Main ToolBar",
            "Stage",
            "Layer",
            "Property",
            "Render Settings",
            "Content",
            "Flow",
            "Semantics Schema Editor",
            "VR",
            "Isaac Sim Assets [Beta]",
        ]
    )
    import omni.ui as ui
    for name in hide_window_names:
        window = ui.Workspace.get_window(name)
        if window is not None:
            window.visible = False
