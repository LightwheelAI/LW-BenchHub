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

        plate_placement = dict(
            fixture=self.counter,
            pos=(0.2, 0.10),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.30, 0.30),
        )
        pudding_placement = dict(
            fixture=self.counter,
            pos=(0, -0.35),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.30, 0.30),
        )
        mug_l_placement = dict(
            fixture=self.counter,
            pos=(0.55, -0.55),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.25, 0.25),
        )
        mug_r_placement = dict(
            fixture=self.counter,
            pos=(0.20, -1.00),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.25, 0.25),
        )

        cfgs.append(
            dict(
                name=self.ref_name,
                obj_groups=self.ref_groups,
                graspable=False,
                placement=plate_placement,
                init_robot_here=True,
                asset_name="Plate039.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.obj_name,
                obj_groups=self.obj_groups,
                graspable=False,
                placement=pudding_placement,
                object_scale=0.7,
                asset_name="ChocolatePudding001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups="cup",
                graspable=True,
                placement=mug_l_placement,
                asset_name="Cup012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups="cup",
                graspable=True,
                placement=mug_r_placement,
                asset_name="Cup030.usd",
            )
        )

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

        plate_placement = dict(
            fixture=self.counter,
            pos=(0.2, 0.10),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.50, 0.50),
        )
        pudding_placement = dict(
            fixture=self.counter,
            pos=(0, -0.35),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.70, 0.70),
        )
        mug_l_placement = dict(
            fixture=self.counter,
            pos=(0.55, -0.55),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.25, 0.25),
        )
        mug_r_placement = dict(
            fixture=self.counter,
            pos=(0.20, -1.00),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.25, 0.25),
        )

        cfgs.append(
            dict(
                name=self.ref_name,
                obj_groups=self.ref_groups,
                graspable=False,
                placement=plate_placement,
                init_robot_here=True,
                asset_name="Plate012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.obj_name,
                obj_groups=self.obj_groups,
                graspable=True,
                placement=pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.porcelain_mug,
                obj_groups="cup",
                graspable=True,
                placement=mug_l_placement,
                asset_name="Cup012.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.red_coffee_mug,
                obj_groups="cup",
                graspable=True,
                placement=mug_r_placement,
                asset_name="Cup030.usd",
            )
        )

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
