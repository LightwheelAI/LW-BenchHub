import numpy as np
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from typing import Union
import torch


class FryingPanAdjustment(LwLabTaskBase):
    """
    Frying Pan Adjustment: composite task for Frying activity.

    Simulates the task of adjusting the frying pan on the stove.

    Steps:
        Move the pan from the current burner to another burner and turn on
        the burner.
    """

    layout_registry_names: list[int] = [FixtureType.STOVE]
    task_name: str = "FryingPanAdjustment"
    knob_id: str = "random"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.stove

        if "refs" in scene._ep_meta:
            self.knob = scene._ep_meta["refs"]["knob"]
        else:
            valid_knobs = self.stove.valid_locations
            if self.knob_id == "random":
                self.knob = self.rng.choice(list(valid_knobs))
            else:
                assert self.knob_id in valid_knobs
                self.knob = self.knob

    def _reset_internal(self, env, env_ids):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal(env, env_ids)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs)

    def _setup_scene(self, env, env_ids=None):
        super()._setup_scene(env, env_ids)

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="obj",
                obj_groups=("pan"),
                placement=dict(
                    fixture=self.stove,
                    sample_region_kwargs=dict(
                        locs=[self.knob],
                    ),
                    ensure_object_boundary_in_range=False,
                    size=(0.05, 0.05),
                ),
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick and place the pan from the current burner to another burner and turn the burner on."
        return ep_meta

    def _check_success(self, env):
        curr_loc = self.stove.check_obj_location_on_stove(env, "obj", need_knob_on=True)
        other_knobs = set([k for k in self.stove.valid_locations if k != self.knob])

        # Check if pan is on a burner and not at start location
        pan_on_stove = torch.tensor([
            loc is not None and len(loc) > 0 and loc[0] is not None
            for loc in curr_loc
        ], device=env.device)
        not_start_loc = torch.tensor([
            (loc and len(loc) > 0 and loc[0] is not None and loc[0] in other_knobs) if loc else False
            for loc in curr_loc
        ], device=env.device)

        return pan_on_stove & not_start_loc
