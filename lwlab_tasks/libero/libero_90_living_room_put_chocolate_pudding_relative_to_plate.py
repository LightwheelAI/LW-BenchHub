import lwlab.utils.object_utils as OU
from lwlab.core.tasks.base import LwLabTaskBase
import torch
from lwlab.core.models.fixtures import FixtureType


class RelativePlacementBase(LwLabTaskBase):
    task_name: str = "RelativePlacementBase"

    obj_name: str = "obj"
    ref_name: str = "ref"
    obj_groups: list | tuple | None = None
    ref_groups: list | tuple | None = None
    obj_mjcf_path: str | None = None
    ref_mjcf_path: str | None = None
    relation: str = "left"  # left / right / front / behind
    dist_range: tuple[float, float] = (0.10, 0.35)
    lateral_tol: float = 0.20
    stable_vel_th: float = 0.5

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.init_robot_base_ref = self.counter
        # additional object names for consistency with other tasks
        self.porcelain_mug = "porcelain_mug"
        self.red_coffee_mug = "red_coffee_mug"

    def _get_obj_cfgs(self):
        cfgs = []

        placement = dict(
            fixture=self.counter,
            size=(0.8, 0.4),
            pos=(0.0, -0.6),
            ensure_valid_placement=True,
        )

        def add_cfg(name, obj_groups=None, graspable=True, mjcf_path=None):
            info = dict(mjcf_path=mjcf_path) if mjcf_path else None
            cfg = dict(name=name, obj_groups=obj_groups or name, graspable=graspable, placement=placement)
            if info:
                cfg["info"] = info
            cfgs.append(cfg)

        add_cfg(self.ref_name, self.ref_groups, False, self.ref_mjcf_path)
        add_cfg(self.obj_name, self.obj_groups, True, self.obj_mjcf_path)
        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the {self.obj_name.replace('_', ' ')} {self.relation} of the {self.ref_name.replace('_', ' ')}."
        return ep_meta

    def _check_success(self, env):
        # use check_place_obj1_side_by_obj2 for side-by-side placement check (no angle requirement)
        return OU.check_place_obj1_side_by_obj2(env, self.obj_name, self.ref_name, {
            "gripper_far": True,   # obj1 and obj2 should be far from the gripper
            "contact": False,   # obj1 should not be in contact with obj2
            "side": self.relation,    # relative position of obj1 to obj2
            "side_threshold": 0.25,    # threshold for distance between obj1 and obj2 in other directions
            "margin_threshold": [0.001, 0.1],    # threshold for distance between obj1 and obj2
            "stable_threshold": 0.5,    # threshold for stable, velocity vector length less than 0.5
        })


class L90L6PutTheChocolatePuddingToTheLeftOfThePlate(RelativePlacementBase):
    task_name: str = "L90L6PutTheChocolatePuddingToTheLeftOfThePlate"
    relation: str = "left"
    obj_name: str = "chocolate_pudding"
    ref_name: str = "plate"
    obj_groups = "chocolate_pudding"
    ref_groups = "plate"
    obj_mjcf_path = None
    ref_mjcf_path = None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the chocolate pudding to the left of the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos, size):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, init_robot_here=False, scale=1.0):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        object_scale=scale,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                        **({"init_robot_here": True} if init_robot_here else {}),
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        object_scale=scale,
                        graspable=graspable,
                        placement=placement,
                        **({"init_robot_here": True} if init_robot_here else {}),
                    )
                )

        # 初始布局：盘子在右、布丁在左，分开放置；干扰物放前缘
        plate_placement = get_placement((0.2, 0.10), (0.30, 0.30))
        pudding_placement = get_placement((0, -0.35), (0.30, 0.30))
        mug_l_placement = get_placement((0.55, -0.55), (0.25, 0.25))
        mug_r_placement = get_placement((0.20, -1.00), (0.25, 0.25))

        add_cfg(self.ref_name, self.ref_groups, False, plate_placement, init_robot_here=True, mjcf_path="/objects/lightwheel/plate/Plate039/model.xml")
        add_cfg(self.obj_name, self.obj_groups, False, pudding_placement, scale=0.7, mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml")
        # 杯子按 CSV 归属到 cup 类别，通过路径指定具体型号
        add_cfg(self.porcelain_mug, "cup", True, mug_l_placement, mjcf_path="/objects/lightwheel/cup/Cup012/model.xml")
        add_cfg(self.red_coffee_mug, "cup", True, mug_r_placement, mjcf_path="/objects/lightwheel/cup/Cup030/model.xml")

        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_side_by_obj2(env, self.obj_name, self.ref_name, {
            "gripper_far": True,   # obj1 and obj2 should be far from the gripper
            "contact": False,   # obj1 should not be in contact with obj2
            "side": 'left',    # relative position of obj1 to obj2
            "side_threshold": 0.25,    # threshold for distance between obj1 and obj2 in other directions
            "margin_threshold": [0.001, 0.1],    # threshold for distance between obj1 and obj2
            "stable_threshold": 0.5,    # threshold for stable, velocity vector length less than 0.5
        })


class L90L6PutTheChocolatePuddingToTheRightOfThePlate(RelativePlacementBase):
    task_name: str = "L90L6PutTheChocolatePuddingToTheRightOfThePlate"
    relation: str = "right"
    obj_name: str = "chocolate_pudding"
    ref_name: str = "plate"
    obj_groups = "chocolate_pudding"
    ref_groups = "plate"
    obj_mjcf_path = None
    ref_mjcf_path = None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the chocolate pudding to the right of the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        def get_placement(pos, size):
            return dict(
                fixture=self.counter,
                size=size,
                pos=pos,
                margin=0.02,
                ensure_valid_placement=True,
            )

        def add_cfg(name, obj_groups, graspable, placement, mjcf_path=None, init_robot_here=False):
            if mjcf_path is not None:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        info=dict(mjcf_path=mjcf_path),
                        placement=placement,
                        **({"init_robot_here": True} if init_robot_here else {}),
                    )
                )
            else:
                cfgs.append(
                    dict(
                        name=name,
                        obj_groups=obj_groups,
                        graspable=graspable,
                        placement=placement,
                        **({"init_robot_here": True} if init_robot_here else {}),
                    )
                )

        # 初始布局：盘子在左、布丁在右，分开放置；干扰物放前缘
        plate_placement = get_placement((0.2, 0.10), (0.50, 0.50))
        pudding_placement = get_placement((0, -0.35), (0.70, 0.70))
        mug_l_placement = get_placement((0.55, -0.55), (0.25, 0.25))
        mug_r_placement = get_placement((0.20, -1.00), (0.25, 0.25))

        add_cfg(self.ref_name, self.ref_groups, False, plate_placement, init_robot_here=True, mjcf_path="/objects/lightwheel/plate/Plate012/model.xml")
        add_cfg(self.obj_name, self.obj_groups, True, pudding_placement, mjcf_path="/objects/lightwheel/chocolate_pudding/ChocolatePudding001/model.xml")
        add_cfg(self.porcelain_mug, "cup", True, mug_l_placement, mjcf_path="/objects/lightwheel/cup/Cup012/model.xml")
        add_cfg(self.red_coffee_mug, "cup", True, mug_r_placement, mjcf_path="/objects/lightwheel/cup/Cup030/model.xml")

        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_side_by_obj2(env, self.obj_name, self.ref_name, {
            "gripper_far": True,   # obj1 and obj2 should be far from the gripper
            "contact": False,   # obj1 should not be in contact with obj2
            "side": 'right',    # relative position of obj1 to obj2
            "side_threshold": 0.25,    # threshold for distance between obj1 and obj2 in other directions
            "margin_threshold": [0.001, 0.1],    # threshold for distance between obj1 and obj2
            "stable_threshold": 0.5,    # threshold for stable, velocity vector length less than 0.5
        },
            gipper_th=0.35)
