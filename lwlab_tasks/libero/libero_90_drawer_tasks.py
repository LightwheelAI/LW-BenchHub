import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class LiberoDrawerTasksBase(LwLabTaskBase):
    """
    LiberoDrawerTasksBase: base class for all libero drawer tasks
    """

    task_name: str = "LiberoDrawerTasksBase"
    enable_fixtures = ["storage_furniture", "winerack"]
    removable_fixtures = ["winerack"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.table))

        self.init_robot_base_ref = self.table
        # Define object names for drawer tasks
        self.akita_black_bowl = "akita_black_bowl"
        self.cream_cheese = "cream_cheese"
        self.plate = "plate"
        self.wine_bottle = "wine_bottle"
        self.butter = "butter"
        self.chocolate_pudding = "chocolate_pudding"
        self.ketchup = "ketchup"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the top drawer joint name (first joint)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
            self.drawer.set_joint_state(0.9, 1.0, env, [self.top_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.top_drawer_joint_name = "drawer_joint_1"

    def _get_obj_cfgs(self):
        cfgs = []
        return cfgs

    def _check_success(self, env):
        return torch.tensor([False], device=env.device)


class L90K5PutTheBlackBowlInTheTopDrawerOfTheCabinet(LiberoDrawerTasksBase):
    """
    L90K5PutTheBlackBowlInTheTopDrawerOfTheCabinet: put the black bowl in the top drawer of the cabinet

    Steps:
        open the top drawer of the cabinet
        pick up the black bowl
        put the black bowl in the top drawer of the cabinet

    """

    task_name: str = "L90K5PutTheBlackBowlInTheTopDrawerOfTheCabinet"
    enable_fixtures = ["storage_furniture"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl in the top drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):

        bowl_placement = dict(
            fixture=self.table,
            pos=(0.2, -0.60),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.6, 0.6),
        )
        ketchup_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.45, 0.45),
        )
        plate_placement = dict(
            fixture=self.table,
            pos=(0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )
        cfgs = []

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
                object_scale=0.8,
            )
        )
        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups="ketchup",
                graspable=True,
                placement=ketchup_placement,
                asset_name="Ketchup003.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        bowl_in_drawer = OU.obj_inside_of(env, self.akita_black_bowl, self.drawer)
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)
        return bowl_in_drawer & gripper_far


class L90K1OpenTheTopDrawerOfTheCabinetAndPutTheBowlInIt(LiberoDrawerTasksBase):
    """
    L90K1OpenTheTopDrawerOfTheCabinetAndPutTheBowlInIt: open the top drawer of the cabinet and put the bowl in it

    Steps:
        open the top drawer of the cabinet
        pick up the bowl
        put the bowl in the top drawer of the cabinet

    """

    task_name: str = "L90K1OpenTheTopDrawerOfTheCabinetAndPutTheBowlInIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Open the top drawer of the cabinet and put the bowl in it."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
            self.drawer.set_joint_state(0.0, 0.0, env, [self.top_drawer_joint_name])
        else:
            self.top_drawer_joint_name = "drawer_joint_1"

    def _get_obj_cfgs(self):
        cfgs = []

        bowl_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.60),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        plate_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
                object_scale=0.7,
            )
        )
        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        bowl_in_drawer = OU.obj_inside_of(env, self.akita_black_bowl, self.drawer)
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)
        return bowl_in_drawer & gripper_far


class L90K5CloseTheTopDrawerOfTheCabinet(LiberoDrawerTasksBase):
    """
    L90K5CloseTheTopDrawerOfTheCabinet: close the top drawer of the cabinet

    Steps:
        close the top drawer of the cabinet

    """

    task_name: str = "L90K5CloseTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the top drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        bowl_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.60),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        ketchup_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.45, 0.45),
        )
        plate_placement = dict(
            fixture=self.table,
            pos=(0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.7, 0.7),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups="ketchup",
                graspable=True,
                placement=ketchup_placement,
                asset_name="Ketchup003.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        cabinet_closed = self.drawer.is_closed(env, [self.top_drawer_joint_name])
        return cabinet_closed


class L90K10CloseTheTopDrawerOfTheCabinet(LiberoDrawerTasksBase):
    """
    L90K10CloseTheTopDrawerOfTheCabinet: close the top drawer of the cabinet (Scene10)

    Steps:
        close the top drawer of the cabinet

    """

    task_name: str = "L90K10CloseTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the top drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        bowl_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.60),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        butter_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        chocolate_pudding_placement = dict(
            fixture=self.table,
            pos=(0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups="chocolate_pudding",
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        cabinet_closed = self.drawer.is_closed(env, [self.top_drawer_joint_name])
        return cabinet_closed


class L90K4CloseTheBottomDrawerOfTheCabinet(LiberoDrawerTasksBase):
    """
    L90K4CloseTheBottomDrawerOfTheCabinet: close the bottom drawer of the cabinet

    Steps:
        close the bottom drawer of the cabinet

    """

    task_name: str = "L90K4CloseTheBottomDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the bottom drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        bowl_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
                object_scale=0.7,
            )
        )
        cfgs.append(
            dict(
                name=f"wine_bottle",
                obj_groups=["bottle"],
                graspable=True,
                washable=True,
                object_scale=0.8,
                asset_name="Bottle054.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(0.2, -0.3),
                ),
            )
        )

        return cfgs

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            joint_names = list(self.drawer._joint_infos.keys())
            self.bottom_drawer_joint_name = joint_names[-1]
            self.top_drawer_joint_name = joint_names[0]
            self.drawer.set_joint_state(0, 0, env, [self.top_drawer_joint_name])
            self.drawer.set_joint_state(0.8, 0.9, env, [self.bottom_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.bottom_drawer_joint_name = "drawer_joint_2"

    def _check_success(self, env):
        cabinet_closed = self.drawer.is_closed(env, [self.bottom_drawer_joint_name])
        return cabinet_closed & OU.gripper_obj_far(env, self.drawer.name)


class L90K4CloseTheBottomDrawerOfTheCabinetAndOpenTheTopDrawer(L90K4CloseTheBottomDrawerOfTheCabinet):
    task_name: str = "L90K4CloseTheBottomDrawerOfTheCabinetAndOpenTheTopDrawer"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the bottom drawer of the cabinet and open the top drawer."
        return ep_meta

    def _check_success(self, env):
        cabinet_closed = self.drawer.is_closed(env, [self.bottom_drawer_joint_name])
        top_open = self.drawer.is_open(env, [self.top_drawer_joint_name], th=0.95)
        return cabinet_closed & top_open & OU.gripper_obj_far(env, self.drawer.name, th=0.35)


class L90K4PutTheBlackBowlInTheBottomDrawerOfTheCabinet(L90K4CloseTheBottomDrawerOfTheCabinet):
    task_name: str = "L90K4PutTheBlackBowlInTheBottomDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl in the bottom drawer of the cabinet."
        return ep_meta

    def _check_success(self, env):
        bowl_success = OU.obj_inside_of(env, "akita_black_bowl", self.drawer)
        return bowl_success & OU.gripper_obj_far(env, "akita_black_bowl")


class L10K4PutTheBlackBowlInTheBottomDrawerOfTheCabinetAndCloseIt(L90K4CloseTheBottomDrawerOfTheCabinet):
    task_name: str = "L10K4PutTheBlackBowlInTheBottomDrawerOfTheCabinetAndCloseIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl in the bottom drawer of the cabinet and close it."
        return ep_meta

    def _check_success(self, env):
        bowl_success = OU.obj_inside_of(env, "akita_black_bowl", self.drawer)
        return bowl_success & OU.gripper_obj_far(env, "akita_black_bowl") & self.drawer.is_closed(env, [self.bottom_drawer_joint_name])


class L90K4PutTheBlackBowlOnTopOfTheCabinet(L90K4CloseTheBottomDrawerOfTheCabinet):
    task_name: str = "L90K4PutTheBlackBowlOnTopOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl on top of the cabinet."
        return ep_meta

    def _check_success(self, env):
        bowl_success = OU.obj_inside_of(env, self.akita_black_bowl, self.drawer)
        return bowl_success & OU.gripper_obj_far(env, self.akita_black_bowl)


class Libero90PutBowlONCabinetTopDrawer(LwLabTaskBase):
    task_name: str = "Libero90PutBowlONCabinetTopDrawer"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl on top of the cabinet."
        return ep_meta

    def _check_success(self, env):
        bowl_success = OU.obj_inside_of(env, "akita_black_bowl", self.drawer, partial_check=True)
        return bowl_success & OU.gripper_obj_far(env, "akita_black_bowl")


class L90K4PutTheWineBottleInTheBottomDrawerOfTheCabinet(L90K4CloseTheBottomDrawerOfTheCabinet):
    task_name: str = "L90K4PutTheWineBottleInTheBottomDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the wine bottle in the bottom drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        bowl_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.30),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
                object_scale=0.7,
            )
        )
        cfgs.append(
            dict(
                name=f"wine_bottle",
                obj_groups=["bottle"],
                graspable=True,
                washable=True,
                object_scale=0.7,
                asset_name="Bottle054.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(0.2, -0.3),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        bottle_success = OU.check_obj_fixture_contact(env, "wine_bottle", self.drawer)
        bottle_stable = OU.check_object_stable(env, "wine_bottle")
        bottle_gripper_far = OU.gripper_obj_far(env, "wine_bottle")
        return bottle_success & bottle_stable & bottle_gripper_far


class L90K10CloseTheTopDrawerOfTheCabinetAndPutTheBlackBowlOnTopOfIt(LiberoDrawerTasksBase):
    """
    L90K10CloseTheTopDrawerOfTheCabinetAndPutTheBlackBowlOnTopOfIt: close the top drawer of the cabinet and put the black bowl on top of it

    Steps:
        1. close the top drawer of the cabinet
        2. put the black bowl on top of the drawer

    """

    task_name: str = "L90K10CloseTheTopDrawerOfTheCabinetAndPutTheBlackBowlOnTopOfIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Close the top drawer of the cabinet and put the black bowl on top of it."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, butter, chocolate_pudding
        bowl_placement = dict(
            fixture=self.table,
            pos=(0.0, -0.60),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        butter_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        chocolate_pudding_placement = dict(
            fixture=self.table,
            pos=(0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups="chocolate_pudding",
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )

        return cfgs

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the top drawer joint name (first joint)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
            # Set the top drawer to semi-open state
            self.drawer.set_joint_state(0.1, 0.2, env, [self.top_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.top_drawer_joint_name = "drawer_joint_1"

    def _check_success(self, env):
        # Check if the top drawer is closed
        drawer_closed = self.drawer.is_closed(env, [self.top_drawer_joint_name])
        bowl_pos = env.scene.rigid_objects[self.akita_black_bowl].data.root_pos_w[0, :].cpu().numpy()
        bowl_on_drawer = OU.point_in_fixture(bowl_pos, self.drawer, only_2d=True)
        bowl_on_drawer_tensor = torch.tensor(bowl_on_drawer, dtype=torch.bool, device=env.device).repeat(env.num_envs)
        # Check if gripper is far from the bowl
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)

        return drawer_closed & bowl_on_drawer_tensor & gripper_far


class _BaseDrawerTasksWithoutWineRack(LwLabTaskBase):
    """
    Base class for drawer tasks that don't need wine_rack
    """

    task_name: str = "_BaseDrawerTasksWithoutWineRack"
    enable_fixtures = ["storage_furniture"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.table = self.register_fixture_ref(
            "table", dict(id=FixtureType.TABLE)
        )
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE, ref=self.table))

        self.init_robot_base_ref = self.table
        # Define object names for drawer tasks
        self.akita_black_bowl = "akita_black_bowl"
        self.butter_1 = "butter_1"
        self.butter_2 = "butter_2"
        self.chocolate_pudding = "chocolate_pudding"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        # Get the top drawer joint name (first joint)
        if hasattr(self.drawer, '_joint_infos') and self.drawer._joint_infos:
            self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
            self.drawer.set_joint_state(0.45, 0.5, env, [self.top_drawer_joint_name])
        else:
            # Use default joint name if joint info is not available
            self.top_drawer_joint_name = "drawer_joint_1"

    def _get_obj_cfgs(self):
        cfgs = []
        return cfgs

    def _check_success(self, env):
        return torch.tensor([False], device=env.device)


class L90K10PutTheBlackBowlInTheTopDrawerOfTheCabinet(_BaseDrawerTasksWithoutWineRack):
    """
    L90K10PutTheBlackBowlInTheTopDrawerOfTheCabinet: put the black bowl in the top drawer of the cabinet (Scene10)

    Steps:
        1. open the top drawer of the cabinet
        2. pick up the black bowl
        3. put the black bowl in the top drawer of the cabinet

    """

    task_name: str = "L90K10PutTheBlackBowlInTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl in the top drawer of the cabinet."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, butter, chocolate_pudding
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        butter_placement_1 = dict(
            fixture=self.table,
            pos=(0.3, -0.80),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        butter_placement_2 = dict(
            fixture=self.table,
            pos=(0.3, -0.3),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        chocolate_pudding_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.3),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
                object_scale=0.7,
            )
        )
        cfgs.append(
            dict(
                name=self.butter_1,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement_1,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter_2,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement_2,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups="chocolate_pudding",
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )
        return cfgs

    def _check_success(self, env):
        bowl_in_drawer = OU.obj_inside_of(env, self.akita_black_bowl, self.drawer)
        gripper_far = OU.gripper_obj_far(env, self.akita_black_bowl)
        return bowl_in_drawer & gripper_far


class L90K10PutTheButterAtTheFrontInTheTopDrawerOfTheCabinetAndCloseIt(_BaseDrawerTasksWithoutWineRack):
    """
    L90K10PutTheButterAtTheFrontInTheTopDrawerOfTheCabinetAndCloseIt: put the butter at the front in the top drawer of the cabinet and close it

    Steps:
        1. open the top drawer of the cabinet
        2. pick up the butter
        3. put the butter at the front in the top drawer of the cabinet
        4. close the top drawer

    """

    task_name: str = "L90K10PutTheButterAtTheFrontInTheTopDrawerOfTheCabinetAndCloseIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the butter at the front in the top drawer of the cabinet and close it."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        # According to task.csv, need: akita_black_bowl, butter, chocolate_pudding
        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )
        butter_placement_1 = dict(
            fixture=self.table,
            pos=(0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.2, 0.2),
        )
        butter_placement_2 = dict(
            fixture=self.table,
            pos=(0.3, 0.0),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.2, 0.2),
        )
        chocolate_pudding_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.3),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter_1,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement_1,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter_2,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement_2,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups="chocolate_pudding",
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if at least one butter is inside the drawer
        # butter_1_in_drawer = OU.obj_inside_of(env, self.butter_1, self.drawer)
        butter_2_in_drawer = OU.obj_inside_of(env, self.butter_2, self.drawer)
        # any_butter_in_drawer = butter_1_in_drawer or butter_2_in_drawer

        # Check if the top drawer is closed
        drawer_closed = self.drawer.is_closed(env, [self.top_drawer_joint_name])

        # Check if gripper is far from both butters
        # gripper_far_1 = OU.gripper_obj_far(env, self.butter_1)
        gripper_far_2 = OU.gripper_obj_far(env, self.butter_2)
        # gripper_far = gripper_far_1 and gripper_far_2

        # Convert to boolean and combine results
        return butter_2_in_drawer & drawer_closed & gripper_far_2


class L90K10PutTheChocolatePuddingInTheTopDrawerOfTheCabinetAndCloseIt(_BaseDrawerTasksWithoutWineRack):
    """
    L90K10PutTheChocolatePuddingInTheTopDrawerOfTheCabinetAndCloseIt: put the chocolate pudding in the top drawer of the cabinet and close it

    Steps:
        1. open the top drawer of the cabinet
        2. pick up the chocolate pudding
        3. put the chocolate pudding in the top drawer of the cabinet
        4. close the top drawer

    """
    task_name: str = "L90K10PutTheChocolatePuddingInTheTopDrawerOfTheCabinetAndCloseIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the chocolate pudding in the top drawer of the cabinet and close it."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        bowl_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        butter_placement_1 = dict(
            fixture=self.table,
            pos=(0.3, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        butter_placement_2 = dict(
            fixture=self.table,
            pos=(0.3, -0.3),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.3, 0.3),
        )
        chocolate_pudding_placement = dict(
            fixture=self.table,
            pos=(-0.3, -0.3),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.5, 0.5),
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter_1,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement_1,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.butter_2,
                obj_groups="butter",
                graspable=True,
                placement=butter_placement_2,
                asset_name="Butter001.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.chocolate_pudding,
                obj_groups="chocolate_pudding",
                graspable=True,
                placement=chocolate_pudding_placement,
                asset_name="ChocolatePudding001.usd",
                object_scale=0.7,
            )
        )

        return cfgs

    def _check_success(self, env):
        # Check if the chocolate pudding is inside the drawer
        chocolate_pudding_in_drawer = OU.obj_inside_of(env, self.chocolate_pudding, self.drawer)

        # Check if the top drawer is closed
        drawer_closed = self.drawer.is_closed(env, [self.top_drawer_joint_name])

        # Check if gripper is far from the chocolate pudding
        gripper_far = OU.gripper_obj_far(env, self.chocolate_pudding)

        return chocolate_pudding_in_drawer & drawer_closed & gripper_far
