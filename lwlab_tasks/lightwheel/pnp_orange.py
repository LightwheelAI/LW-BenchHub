import torch
import numpy as np
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from dataclasses import MISSING
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
import lwlab.core.mdp as mdp


@configclass
class LeRobotVisualObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_target_pos = ObsTerm(func=mdp.get_target_qpos, params={"action_name": 'arm_action'})
        hand_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("hand_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        global_camera = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("global_camera"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


class PnPOrange(LwLabTaskBase):
    task_name: str = "PnPOrange"
    observations: LeRobotVisualObservationsCfg = LeRobotVisualObservationsCfg()
    resample_objects_placement_on_reset: bool = True

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.sink = self.register_fixture_ref("sink", dict(id=FixtureType.SINK))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.sink)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        obj_lang = self.get_obj_lang(obj_name="orange")
        ep_meta[
            "lang"
        ] = f"Pick the {obj_lang} from the counter and place it in the bowl."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="orange",
                obj_groups="orange",
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right",
                    ),
                    size=(0.60, 0.20),
                    offset=(0.0, -0.15),
                ),
            )
        )

        # distractors
        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.sink, loc="right",
                    ),
                    size=(0.60, 0.20),
                    offset=(0.0, -0.15),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        orange_in_bowl = OU.check_obj_in_receptacle(
            env, "orange", "bowl"
        )
        gripper_orange_far = OU.gripper_obj_far(env, obj_name="orange", eef_name="tool_gripper", th=0.10)
        gripper_bowl_far = OU.gripper_obj_far(env, obj_name="bowl", eef_name="tool_gripper", th=0.10)
        return orange_in_bowl & gripper_orange_far & gripper_bowl_far
