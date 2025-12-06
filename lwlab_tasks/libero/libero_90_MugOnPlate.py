import copy
from lwlab.core.tasks.base import LwLabTaskBase
import re
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from .libero_10_MugOnAndChocolateRightPlate import L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import lwlab.core.mdp as mdp
import torch


@configclass
class VisualObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_target_pos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
        left_hand_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_hand_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        right_hand_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_hand_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        eye_in_hand_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("eye_in_hand_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class L90L6PutTheRedMugOnThePlate(L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate):
    """
    L90L6PutTheRedMugOnThePlate: put the red mug on the right plate
    """

    task_name: str = "L90L6PutTheRedMugOnThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the red mug and put it on the plate"
        return ep_meta

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.red_coffee_mug,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )


class L90L5PutTheRedMugOnTheRightPlate(L10L6PutWhiteMugOnPlateAndPutChocolatePuddingToRightPlate):

    """
    L90L5PutTheRedMugOnTheRightPlate: put the red mug on the right plate
    """

    task_name: str = "L90L5PutTheRedMugOnTheRightPlate"
    # observations: VisualObservationsCfg = VisualObservationsCfg()
    # resample_objects_placement_on_reset: bool = True

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the red mug and put it on the plate"
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] == self.chocolate_pudding:
                cfgs[cfg_index]['name'] = self.white_yellow_mug
                cfgs[cfg_index]['asset_name'] = "Cup014.usd"
                break
            cfg_index += 1
        return cfgs

    def _check_success(self, env):
        return OU.check_place_obj1_on_obj2(
            env,
            self.red_coffee_mug,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
