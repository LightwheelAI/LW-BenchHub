import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU


class SeasoningSpiceSetup(LwLabTaskBase):
    """
    Seasoning Spice Setup: composite task for Setting The Table activity.

    Simulates the task of setting the table with seasoning and spices.

    Steps:
        Move the seasoning and spices from the cabinet directly in front to the
        dining counter.

    Restricted to layouts which have a dining table (long counter area with
    stools).

    Args:
        cab_id (int): Enum which serves as a unique identifier for different
            cabinet types. Used to choose the cabinet from which the seasoning
            and spices are picked.
    """

    layout_registry_names: list[int] = [FixtureType.CABINET, FixtureType.COUNTER, FixtureType.STOOL]
    task_name: str = "SeasoningSpiceSetup"
    EXCLUDE_LAYOUTS = LwLabTaskBase.STOOL_EXCLUDED_LAYOUT
    cab_id: FixtureType = FixtureType.CABINET

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.COUNTER, ref=FixtureType.STOOL, size=(0.75, 0.2)),
        )

        self.init_robot_base_ref = self.cab

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        condiment1_name = self.get_obj_lang("condiment1")
        condiment2_name = self.get_obj_lang("condiment2")
        ep_meta[
            "lang"
        ] = f"Move the {condiment1_name} and {condiment2_name} from the cabinet to the dining counter."
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.cab.open_door(env=env, env_ids=env_ids)

    def _reset_internal(self, env, env_ids=None):
        return super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="condiment1",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(-0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="condiment2",
                obj_groups="condiment",
                graspable=True,
                placement=dict(
                    fixture=self.cab,
                    size=(0.4, 0.20),
                    pos=(0.5, -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dstr_dining2",
                obj_groups="all",
                placement=dict(
                    fixture=self.dining_table,
                    size=(1, 0.30),
                    pos=(0, 0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):
        gripper_condiment1_far = OU.gripper_obj_far(env, obj_name="condiment1")
        gripper_condiment2_far = OU.gripper_obj_far(env, obj_name="condiment2")
        condiment1_on_counter = OU.check_obj_fixture_contact(
            env, "condiment1", self.dining_table
        )
        condiment2_on_counter = OU.check_obj_fixture_contact(
            env, "condiment2", self.dining_table
        )
        return gripper_condiment1_far & gripper_condiment2_far & condiment1_on_counter & condiment2_on_counter
