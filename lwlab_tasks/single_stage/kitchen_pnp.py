import torch
import numpy as np
from lwlab.core.tasks.base import LwLabTaskBase
from dataclasses import MISSING
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import lwlab.utils.place_utils.env_utils as EnvUtils
from lwlab.core.models import fixtures
from lwlab.core.models.fixtures.others import Floor, Wall


class PnP(LwLabTaskBase):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """

    obj_groups: str = "all"
    exclude_obj_groups: str | None = None

    def _get_obj_cfgs(self):
        raise NotImplementedError


class PnPCounterToCabinet(PnP):
    """
    Class encapsulating the atomic counter to cabinet pick and place task

    Args:
        cab_id (str): The cabinet fixture id to place the object.

        obj_groups (str): Object groups to sample the target object from.
    """
    cab_id: FixtureType = FixtureType.CABINET
    task_name: str = "PnPCounterToCabinet"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to cabinet pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the cabinet."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to cabinet pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the back area of the cabinet.

        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(0.60, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.10),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(None, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the counter to cabinet pick and place task is successful.
        Checks if the object is inside the cabinet and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_inside_cab = OU.obj_inside_of(env, "obj", self.cab)
        gripper_obj_far = OU.gripper_obj_far(env)
        return obj_inside_cab & gripper_obj_far


class PnPCabinetToCounter(PnP):
    """
    Class encapsulating the atomic cabinet to counter pick and place task

    Args:
        cab_id (str): The cabinet fixture id to pick the object from.

        obj_groups (str): Object groups to sample the target object from.
    """
    cab_id: FixtureType = FixtureType.CABINET
    task_name: str = "PnPCabinetToCounter"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the cabinet to counter pick and place task:
        The cabinet to pick object from and the counter to place it on
        """
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref(
            "cab",
            dict(id=self.cab_id),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.cab),
        )
        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        """
        Get the episode metadata for the cabinet to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the cabinet and place it on the counter."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the cabinet to counter pick and place task.
        Puts the target object in the front area of the cabinet. Puts a distractor object on the counter
        and the back area of the cabinet.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.50, 0.20),
                    pos=(0, -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.cab,
                    ),
                    size=(1.0, 0.30),
                    pos=(0.0, 1.0),
                    offset=(0.0, -0.05),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_cab",
                obj_groups="all",
                placement=dict(
                    fixture=self.cab,
                    size=(1.0, 0.20),
                    pos=(0.0, 1.0),
                    offset=(0.0, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the cabinet to counter pick and place task is successful.
        Checks if the object is on the counter and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        gripper_obj_far = OU.gripper_obj_far(env)
        obj_on_counter = OU.check_obj_fixture_contact(env, "obj", self.counter)
        return obj_on_counter & gripper_obj_far


class PnPCounterToSink(PnP):
    """
    Class encapsulating the atomic counter to sink pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    obj_groups: str = "all"
    task_name: str = "PnPCounterToSink"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the counter to sink pick and place task:
        The sink to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to sink pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the counter and place it in the sink"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to sink pick and place task.
        Puts the target object in the front area of the counter. Puts a distractor object on the counter
        and the sink.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )
        cfgs.append(
            dict(
                name="distr_sink",
                obj_groups="all",
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the counter to sink pick and place task is successful.
        Checks if the object is inside the sink and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_sink = OU.obj_inside_of(env, "obj", self.sink, partial_check=True)
        gripper_obj_far = OU.gripper_obj_far(env)
        return obj_in_sink & gripper_obj_far


class PnPSinkToCounter(PnP):
    """
    Class encapsulating the atomic sink to counter pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    obj_groups: str = "food"
    task_name: str = "PnPSinkToCounter"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the sink to counter pick and place task:
        The sink to pick object from and the counter to place it on
        """
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref(
            "sink",
            dict(id=FixtureType.SINK),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.sink),
        )
        self.init_robot_base_ref = self.sink

    def get_ep_meta(self):
        """
        Get the episode metadata for the sink to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="obj")
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"pick the {obj_lang} from the sink and place it on the {cont_lang} located on the counter"
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the sink to counter pick and place task.
        Puts the target object in the sink. Puts a distractor object on the counter
        and places a container on the counter for the target object to be placed on.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                washable=True,
                placement=dict(
                    fixture=self.sink,
                    size=(0.25, 0.25),
                    pos=(0.0, 1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups="container",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.35, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    offset=(0.0, 0.30),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the sink to counter pick and place task is successful.
        Checks if the object is in the container, the container on the counter, and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_recep = OU.check_obj_in_receptacle(env, "obj", "container")
        recep_on_counter = OU.check_contact(env, self.objects["container"], self.counter)
        gripper_obj_far = OU.gripper_obj_far(env)
        return obj_in_recep & recep_on_counter & gripper_obj_far


class PnPCounterToMicrowave(PnP):
    """
    Class encapsulating the atomic counter to microwave pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]
    obj_groups: str = "food"
    task_name: str = "PnPCounterToMicrowave"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the counter to microwave pick and place task:
        The microwave to place object on, the counter to initialize it/the container on, and a distractor counter
        """
        super()._setup_kitchen_references(scene)
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.microwave.open_door(env=env, env_ids=env_ids)

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to microwave pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the microwave."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to microwave pick and place task.
        Puts the target object in a container on the counter. Puts a distractor object on the distractor
        counter and places another container in the microwave.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the counter to microwave pick and place task is successful.
        Checks if the object is inside the microwave and on the container and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj = self.objects["obj"]
        container = self.objects["container"]

        obj_container_contact = OU.check_contact(env, obj, container)
        container_micro_contact = OU.check_contact(env, container, self.microwave)
        gripper_obj_far = OU.gripper_obj_far(env)
        return obj_container_contact & container_micro_contact & gripper_obj_far


class PnPMicrowaveToCounter(PnP):
    """
    Class encapsulating the atomic microwave to counter pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    # exclude layout 9 because the microwave is far from counters
    EXCLUDE_LAYOUTS = [9]
    obj_groups: str = "food"
    task_name: str = "PnPMicrowaveToCounter"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the microwave to counter pick and place task:
        The microwave to pick object from, the counter to place it on, and a distractor counter
        """
        super()._setup_kitchen_references(scene)
        self.microwave = self.register_fixture_ref(
            "microwave",
            dict(id=FixtureType.MICROWAVE),
        )
        self.counter = self.register_fixture_ref(
            "counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.distr_counter = self.register_fixture_ref(
            "distr_counter",
            dict(id=FixtureType.COUNTER, ref=self.microwave),
        )
        self.init_robot_base_ref = self.microwave

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.microwave.open_door(env=env, env_ids=env_ids)

    def get_ep_meta(self):
        """
        Get the episode metadata for the microwave to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the microwave and place it on {cont_lang} located on the counter."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the microwave to counter pick and place task.
        Puts the target object in a container in the microwave. Puts a distractor object on the distractor
        counter and places another container on the counter."""
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                microwavable=True,
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="container",
                ),
            )
        )
        cfgs.append(
            dict(
                name="container",
                obj_groups=("container"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="distr_counter",
                obj_groups="all",
                placement=dict(
                    fixture=self.distr_counter,
                    sample_region_kwargs=dict(
                        ref=self.microwave,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", 1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the microwave to counter pick and place task is successful.
        Checks if the object is inside the container and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_container_contact = OU.check_obj_in_receptacle(env, "obj", "container")
        gripper_obj_far = OU.gripper_obj_far(env)
        return obj_container_contact & gripper_obj_far


class PnPCounterToOven(PnP):
    """
    Class encapsulating the counter to oven pick and place atomic task
    """

    EXCLUDE_LAYOUTS = LwLabTaskBase.OVEN_EXCLUDED_LAYOUTS  # TODO
    task_name: str = "PnPCounterToOven"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.oven)
        )
        if "rack_level" in scene._ep_meta:
            self.rack_level = scene._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        if self.oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Place the {obj_lang} on the {rack_pos} rack of the oven."
        else:
            ep_meta["lang"] = f"Place the {obj_lang} on the rack of the oven."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.oven.open_door(env=env, env_ids=env_ids)
        self.oven.slide_rack(env=env, rack_level=self.rack_level, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("oven_ready"),
                graspable=True,
                init_robot_here=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.oven,
                    ),
                    size=(0.45, 0.40),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        cfgs.append(
            dict(
                name="oven_tray",
                obj_groups=("oven_tray"),
                object_scale=0.9,
                placement=dict(
                    fixture=self.oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(1.0, 0.45),
                    pos=(0, -1.0),
                    offset=(0, -0.325),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        obj_container_contact = OU.check_obj_in_receptacle(env, "obj", "oven_tray")
        on_rack = self.oven.check_rack_contact(  # TODO: implement this
            env, "oven_tray", rack_level=self.rack_level
        )
        gripper_far = OU.gripper_obj_far(env, "obj")
        return on_rack & obj_container_contact & gripper_far


class PnPOvenToCounter(PnP):
    """
    Class encapsulating the oven to counter pick and place atomic task
    """

    EXCLUDE_LAYOUTS = LwLabTaskBase.OVEN_EXCLUDED_LAYOUTS
    task_name: str = "PnPOvenToCounter"

    def _load_model(self, *args, **kwargs):
        super()._load_model(*args, **kwargs)
        self._place_robot()

    def _place_robot(self):
        x_ofs = (self.oven.width / 2) + 0.27
        TEST_OFS = 0.23
        inits = []

        # compute where the robot placement if it is to the left of the oven
        (
            robot_base_pos_left,
            robot_base_ori_left,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.oven, offset=(-x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture.
        test_pos_left, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.oven, offset=(-x_ofs - TEST_OFS, -0.10)
        )

        # check if the robot will be in contact with any fixture or wall during initialization
        if not self.check_fxtr_contact(test_pos_left) and not self._point_outside_scene(
            test_pos_left
        ):
            # oven is to the right of the robot
            inits.append((robot_base_pos_left, robot_base_ori_left, "right"))

        # compute where the robot placement if it is to the right of the oven
        (
            robot_base_pos_right,
            robot_base_ori_right,
        ) = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.oven, offset=(x_ofs, -0.10)
        )
        # get a test point to check if the robot is in contact with any fixture if initialized to the right of the oven
        test_pos_right, _ = EnvUtils.compute_robot_base_placement_pose(
            self, ref_fixture=self.oven, offset=(x_ofs + TEST_OFS, -0.10)
        )

        if not self.check_fxtr_contact(
            test_pos_right
        ) and not self._point_outside_scene(test_pos_right):
            inits.append((robot_base_pos_right, robot_base_ori_right, "left"))

        if len(inits) == 0:
            return False
        random_index = self.rng.integers(len(inits))
        robot_base_pos, robot_base_ori, side = inits[random_index]
        self.oven_side = side
        if hasattr(self, "init_robot_base_pos_anchor"):
            self.init_robot_base_pos_anchor[:2] = robot_base_pos[:2]
            self.init_robot_base_ori_anchor[:2] = robot_base_ori[:2]
        else:
            self.init_robot_base_pos_anchor = robot_base_pos
            self.init_robot_base_ori_anchor = robot_base_ori
        return True

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.oven = self.register_fixture_ref("oven", dict(id=FixtureType.OVEN))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.oven)
        )
        if "rack_level" in scene._ep_meta:
            self.rack_level = scene._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0
        self.init_robot_base_ref = self.oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        if self.oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the {rack_pos} rack in the oven and place it on the plate on the counter."
        else:
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the oven and place it on the plate on the counter."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def check_fxtr_contact(self, pos):
        """
        Check if the point is in contact with any fixture

        Args:
            pos (tuple): The position of the point to check

        Returns:
            bool: True if the point is in contact with any fixture, False otherwise
        """
        fxtrs = [
            fxtr
            for fxtr in self.fixtures.values()
            if isinstance(fxtr, fixtures.Counter)
            or isinstance(fxtr, fixtures.Stove)
            or isinstance(fxtr, fixtures.Stovetop)
            or isinstance(fxtr, fixtures.HousingCabinet)
            or isinstance(fxtr, fixtures.SingleCabinet)
            or isinstance(fxtr, fixtures.HingeCabinet)
            or isinstance(fxtr, fixtures.Fridge)
            or (isinstance(fxtr, Wall) and not isinstance(fxtr, Floor))
        ]

        for fxtr in fxtrs:
            # get bounds of fixture
            if OU.point_in_fixture(point=pos, fixture=fxtr, only_2d=True):
                return True
        return False

    def _point_outside_scene(self, pos):
        walls = [
            fxtr for (name, fxtr) in self.fixtures.items() if isinstance(fxtr, Floor)
        ]
        return not any(
            [
                OU.point_in_fixture(point=pos, fixture=wall, only_2d=True)
                for wall in walls
            ]
        )

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.oven.open_door(env=env, env_ids=env_ids)
        self.oven.slide_rack(env=env, rack_level=self.rack_level, env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("oven_ready"),
                graspable=True,
                placement=dict(
                    fixture=self.oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(1.0, 0.45),
                    pos=(0, -1.0),
                    offset=(0, -0.325),
                    try_to_place_in="oven_tray",
                    try_to_place_in_kwargs=dict(
                        object_scale=0.85,
                    ),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.oven,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        obj_in_recep = OU.check_obj_in_receptacle(env, "obj", "container")
        recep_on_counter = OU.check_contact(env, self.objects["container"], self.counter)
        return obj_in_recep & recep_on_counter & OU.gripper_obj_far(env, "obj")


class PnPCounterToStove(PnP):  # DONE
    """
    Class encapsulating the atomic counter to stove pick and place task

    Args:
        obj_groups (str): Object groups to sample the target object from.
    """

    obj_groups: str = "food"
    task_name: str = "PnPCounterToStove"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the counter to stove pick and place task:
        The stove to place object on and the counter to initialize it/container on
        """
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the counter to stove pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        cont_lang = self.get_obj_lang(obj_name="container")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the plate and place it in the {cont_lang}."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the counter to stove pick and place task.
        Puts the target object in a container on the counter and places pan on the stove.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="container",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                ),
            )
        )

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="container",
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the counter to stove pick and place task is successful.
        Checks if the object is on the pan and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_container = OU.check_obj_in_receptacle(env, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(env)

        return obj_in_container & gripper_obj_far


class PnPStoveToCounter(PnP):  # DONE
    """
    Class encapsulating the atomic stove to counter pick and place task
    """

    obj_groups: str = "food"
    task_name: str = "PnPStoveToCounter"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the stove to counter pick and place task:
        The counter to place object/container on and the stove to initialize it/the pan on
        """
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove, size=[0.30, 0.40])
        )
        self.init_robot_base_ref = self.stove

    def get_ep_meta(self):
        """
        Get the episode metadata for the stove to counter pick and place task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        obj_cont_lang = self.get_obj_lang(obj_name="obj_container")
        cont_lang, preposition = self.get_obj_lang(
            obj_name="container", get_preposition=True
        )
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the {obj_cont_lang} and place it {preposition} the {cont_lang}."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the stove to counter pick and place task.
        Puts the target object in a pan on the stove and places a container on the counter.
        """
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=self.obj_groups,
                exclude_obj_groups=self.exclude_obj_groups,
                graspable=True,
                cookable=True,
                max_size=(0.15, 0.15, None),
                placement=dict(
                    fixture=self.stove,
                    ensure_object_boundary_in_range=False,
                    size=(0.02, 0.02),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    try_to_place_in="pan",
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate", "bowl"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        """
        Check if the stove to counter pick and place task is successful.
        Checks if the object is inside the container on the counter and the gripper far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_in_container = OU.check_obj_in_receptacle(env, "obj", "container", th=0.07)
        gripper_obj_far = OU.gripper_obj_far(env)

        return obj_in_container & gripper_obj_far


class PnPToasterToCounter(PnP):  # DONE
    """
    Class encapsulating the toaster to counter pick and place atomic task
    """

    task_name: str = "PnPToasterToCounter"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the toaster to plate task
        """
        super()._setup_kitchen_references(scene)
        self.toaster = self.get_fixture(FixtureType.TOASTER)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster)
        )
        self.init_robot_base_ref = self.toaster

    def get_ep_meta(self):
        """
        Get the episode metadata for the toaster to plate task.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Place the toasted item on a plate."
        return ep_meta

    def _get_obj_cfgs(self):
        """
        Get the object configurations for the toaster to plate task.
        Places a toasted item in the toaster and a plate on the counter.
        """
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("sandwich_bread",),
                rotate_upright=True,
                placement=dict(
                    fixture=self.toaster,
                    rotation=(0, 0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="plate",
                obj_groups="plate",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster,
                    ),
                    size=(0.80, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        """
        Check if the toaster to plate task is successful.
        Checks if the object is on the plate and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """
        obj_on_plate = OU.check_obj_in_receptacle(env, "obj", "plate")
        gripper_obj_far = OU.gripper_obj_far(env)
        return obj_on_plate & gripper_obj_far


class PnPCounterToToasterOven(PnP):  # DONE
    """
    Class encapsulating the counter to toaster oven pick and place atomic task
    """

    task_name: str = "PnPCounterToToasterOven"
    enable_fixtures: list[str] = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        if "rack_level" in scene._ep_meta:
            self.rack_level = scene._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        receptacle_type = "rack" if "rack" in self.chosen_toaster_receptacle else "tray"
        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Place the {obj_lang} on the {rack_pos} {receptacle_type} of the toaster oven."
        else:
            ep_meta[
                "lang"
            ] = f"Place the {obj_lang} on the {receptacle_type} of the toaster oven."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        # TODO: this is being called after get_ep_meta, which is not correct
        self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
            env=env, rack_level=self.rack_level, env_ids=env_ids
        )

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread_food"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.45, 0.30),
                    pos=("ref", -1.0),
                    try_to_place_in="plate",
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        on_rack = self.toaster_oven.check_rack_contact(
            env, "obj", rack_level=self.rack_level
        )
        gripper_far = OU.gripper_obj_far(env, "obj")
        return on_rack & gripper_far


class PnPToasterOvenToCounter(PnP):  # DONE
    """
    Class encapsulating the toaster oven to counter pick and place atomic task
    """
    task_name: str = "PnPToasterOvenToCounter"
    enable_fixtures: list[str] = ["toaster_oven"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.toaster_oven = self.register_fixture_ref(
            "toaster_oven", dict(id=FixtureType.TOASTER_OVEN)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.toaster_oven)
        )
        if "rack_level" in scene._ep_meta:
            self.rack_level = scene._ep_meta["rack_level"]
        else:
            self.rack_level = 1 if self.rng.random() > 0.5 else 0
        self.init_robot_base_ref = self.toaster_oven

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        receptacle_type = "rack" if "rack" in self.chosen_toaster_receptacle else "tray"
        obj_lang = self.get_obj_lang()
        if self.toaster_oven.has_multiple_rack_levels():
            rack_pos = "top" if self.rack_level == 1 else "bottom"
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the {rack_pos} {receptacle_type} and place it on the plate on the counter."
        else:
            ep_meta[
                "lang"
            ] = f"Pick the {obj_lang} from the {receptacle_type} and place it on the plate on the counter."
        ep_meta["rack_level"] = self.rack_level
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.chosen_toaster_receptacle = self.toaster_oven.slide_rack(
            env=env, rack_level=self.rack_level, env_ids=env_ids
        )

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("bread_food"),
                graspable=True,
                placement=dict(
                    fixture=self.toaster_oven,
                    sample_region_kwargs=dict(
                        rack_level=self.rack_level,
                    ),
                    size=(0.50, 0.40),
                    pos=(0, -1.0),
                    offset=(0, -0.20),
                ),
            )
        )

        cfgs.append(
            dict(
                name="container",
                obj_groups=("plate"),
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.toaster_oven,
                        loc="left_right",
                    ),
                    size=(0.40, 0.40),
                    pos=("ref", -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        obj_in_recep = OU.check_obj_in_receptacle(env, "obj", "container")
        recep_on_counter = OU.check_contact(env, self.objects["container"], self.counter)
        return obj_in_recep & recep_on_counter & OU.gripper_obj_far(env, "obj")


class PnPCounterToStandMixer(PnP):  # DONE
    """
    Class encapsulating the task of placing food items in the stand mixer bowl.
    """

    task_name: str = "PnPCounterToStandMixer"
    enable_fixtures: list[str] = ["stand_mixer"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stand_mixer = self.register_fixture_ref(
            "stand_mixer", dict(id=FixtureType.STAND_MIXER)
        )
        self.counter = self.get_fixture(
            FixtureType.COUNTER_NON_CORNER, ref=self.stand_mixer
        )

        self.init_robot_base_ref = self.stand_mixer

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang()
        ep_meta["lang"] = f"Place the {obj_lang} in the stand mixer bowl."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.stand_mixer.set_head_pos(env)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("cheese", "bread", "cake"),
                init_robot_here=True,
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.stand_mixer,
                        loc="left_right",
                    ),
                    size=(0.30, 0.30),
                    pos=("ref", -1.0),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        """
        Check if the food item is inside the stand mixer bowl.

        Returns:
            bool: True if the food item is inside the bowl, False otherwise.
        """
        return self.stand_mixer.check_item_in_bowl(env, "obj") & OU.gripper_obj_far(env, "obj")
