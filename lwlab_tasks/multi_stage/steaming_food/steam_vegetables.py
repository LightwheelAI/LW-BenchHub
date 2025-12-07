import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
import numpy as np


class SteamVegetables(LwLabTaskBase):

    layout_registry_names: list[int] = [FixtureType.STOVE, FixtureType.COUNTER]

    """
    Steam Vegetables: composite task for Steaming Food activity.

    Simulates the task of steaming vegetables based on their cooking time.

    Steps:
        Place vegetables into the pot based on the amount of time it would take to
        steam each. e.g. potatoes and carrots would take the longest. Then, turn
        off the burner beneath the pot.

    Args:
        knob_id (str): The id of the knob who's burner the pot will be placed on.
            If "random", a random knob is chosen.
    """

    task_name: str = "SteamVegetables"
    knob_id = "random"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.stove)
        )
        self.init_robot_base_pos = self.stove

        if "refs" in scene._ep_meta:
            self.knob = scene._ep_meta["refs"]["knob"]
        else:
            valid_knobs = self.stove.valid_locations
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Place vegetables into the pot based on the amount of time it would take to steam each, "
            "e.g. potatoes and carrots would take the longest. "
            "Then turn off the burner beneath the pot."
        )
        ep_meta["knob"] = self.knob
        return ep_meta

    def _reset_internal(self, env, env_ids):
        super()._reset_internal(env, env_ids)
        self.stove.set_knob_state(env=env, knob=self.knob, mode="on", env_ids=env_ids)

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)
        self.stove.set_knob_state(env=env, knob=self.knob, mode="on", env_ids=env_ids)

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="vegetable_hard",
                obj_groups=["potato", "carrot"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.50),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    pos=("ref", -1.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="vegetable_easy",
                obj_groups="vegetable",
                exclude_obj_groups=["potato", "carrot"],
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.50),
                    sample_region_kwargs=dict(
                        ref=self.stove,
                    ),
                    pos=("ref", -1.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="pot",
                obj_groups="pot",
                placement=dict(
                    fixture=self.stove,
                    # ensure_object_boundary_in_range=False because the pans handle is a part of the
                    # bounding box making it hard to place it if set to True
                    ensure_object_boundary_in_range=False,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    rotation=[(-3 * np.pi / 8, -np.pi / 4), (np.pi / 4, 3 * np.pi / 8)],
                    size=(0.02, 0.02),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        # Must place vegetables into pot in sequence
        hard_in_pot = OU.check_obj_in_receptacle(env, "vegetable_hard", "pot")
        easy_in_pot = OU.check_obj_in_receptacle(env, "vegetable_easy", "pot")
        # Check wrong order for each environment individually
        wrong_order_mask = easy_in_pot & ~hard_in_pot
        vegetables_in_pot = hard_in_pot & easy_in_pot

        knobs_state = self.stove.get_knobs_state(env=env)
        knob_value = knobs_state[self.knob]
        knob_off = ~((0.35 <= torch.abs(knob_value)) & (torch.abs(knob_value) <= 2 * torch.pi - 0.35))

        gripper_far = (
            OU.gripper_obj_far(env, "vegetable_hard")
            & OU.gripper_obj_far(env, "vegetable_easy")
            & OU.gripper_obj_far(env, "pot")
        )
        pot_on_stove = OU.check_obj_fixture_contact(
            env, "pot", self.stove
        )

        success = knob_off & gripper_far & pot_on_stove & vegetables_in_pot
        # Mark environments with wrong order as failed (check each environment individually)
        success = success & ~wrong_order_mask
        return success
