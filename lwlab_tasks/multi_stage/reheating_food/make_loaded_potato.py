import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class MakeLoadedPotato(LwLabTaskBase):
    """
    Make Loaded Potato: composite task for Reheating Food activity.

    Simulates the task of making a loaded potato.

    Steps:
        Retrieve the reheated potato from the microwave, then place it on the
        cutting board along with cheese and a bottle of condiment.
    """

    layout_registry_names: list[int] = [FixtureType.COUNTER, FixtureType.MICROWAVE]
    task_name: str = "MakeLoadedPotato"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)

        self.microwave = self.register_fixture_ref(
            "microwave", dict(id=FixtureType.MICROWAVE)
        )
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, size=(0.6, 0.6), ref=self.microwave)
        )
        self.init_robot_base_ref = self.microwave

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"Retrieve the reheated potato from the microwave, then place it on "
            "the cutting board along with cheese and a bottle of condiment."
        )
        return ep_meta

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        # Initialize potato in the microwave
        cfgs.append(
            dict(
                name="potato",
                obj_groups="potato",
                placement=dict(
                    fixture=self.microwave,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    try_to_place_in="plate",
                ),
            )
        )

        # Cutting board at the center of the counter
        cfgs.append(
            dict(
                name="cutting_board",
                obj_groups="cutting_board",
                placement=dict(
                    fixture=self.counter,
                    size=(0.05, 0.05),
                    ensure_object_boundary_in_range=False,
                    pos=(0, 0),
                    rotation=np.pi / 2,
                ),
            )
        )

        # Cheese and condiment to be placed on the cutting board
        cfgs.append(
            dict(
                name="condiment",
                obj_groups="condiment",
                placement=dict(fixture=self.counter, size=(0.6, 0.5), pos=(0, -1)),
            )
        )
        cfgs.append(
            dict(
                name="cheese",
                obj_groups="cheese",
                placement=dict(fixture=self.counter, size=(0.6, 0.5), pos=(0, -1)),
            )
        )
        return cfgs

    def _check_success(self, env):
        gripper_far = (
            OU.gripper_obj_far(env, "potato")
            & OU.gripper_obj_far(env, "condiment")
            & OU.gripper_obj_far(env, "cheese")
        )
        objects_in_place = (
            OU.check_obj_in_receptacle(env, "potato", "cutting_board")
            & OU.check_obj_in_receptacle(env, "condiment", "cutting_board")
            & OU.check_obj_in_receptacle(env, "cheese", "cutting_board")
        )
        return gripper_far & objects_in_place
