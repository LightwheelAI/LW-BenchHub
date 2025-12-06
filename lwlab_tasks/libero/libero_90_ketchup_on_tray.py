import copy
from lwlab.core.tasks.base import LwLabTaskBase
import re
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from .libero_90_tomoto_sauce_on_tray import L90L3PickUpTheTomatoSauceAndPutItInTheTray


class L90L3PickUpTheKetchupAndPutItInTheTray(L90L3PickUpTheTomatoSauceAndPutItInTheTray):
    """
    L90L3PickUpTheKetchupAndPutItInTheTray: pick up the ketchup and put it in the wooden_tray

    Steps:
        pick up the ketchup
        put the ketchup in the wooden_tray

    """

    task_name: str = "L90L3PickUpTheKetchupAndPutItInTheTray"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the ketchup, and put it in the wooden_tray."
        return ep_meta

    def _check_success(self, env):
        if self.is_replay_mode:
            self._get_obj_cfgs()
        return OU.check_place_obj1_on_obj2(
            env,
            self.ketchup,
            self.wooden_tray,
            th_z_axis_cos=0,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
