import torch
from lwlab.core.tasks.base import BaseTaskEnvCfg
from lwlab.core.scenes.kitchen.libero import LiberoEnvCfg
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K1PutTheBlackBowlOnThePlate(LiberoEnvCfg, BaseTaskEnvCfg):

    task_name: str = "L90K1PutTheBlackBowlOnThePlate"

    def __post_init__(self):
        self.init_dish_rack_pos = None
        self.obj_name = []
        return super().__post_init__()

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.init_robot_base_ref = self.dining_table

    def _load_model(self):
        super()._load_model()
        for cfg in self.object_cfgs:
            self.obj_name.append(cfg["name"])

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the black bowl on the plate."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups=["bowl"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/bowl/Bowl008/model.xml"
                ),
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(0.0, -0.7),
                ),
            )
        )

        cfgs.append(
            dict(
                name=f"plate",
                obj_groups=["plate"],
                graspable=True,
                washable=True,
                info=dict(
                    mjcf_path="/objects/lightwheel/plate/Plate012/model.xml"
                ),
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.5),
                    margin=0.02,
                    pos=(-0.3, -0.6)
                ),
            )
        )
        return cfgs

    def _check_success(self):
        th = self.env.cfg.isaac_arena_env.task.objects["plate"].horizontal_radius
        bowl_in_plate = OU.check_obj_in_receptacle_no_contact(self.env, "akita_black_bowl", "plate", th)
        far_from_objects = self._gripper_obj_farfrom_objects()
        return bowl_in_plate & far_from_objects

    def _gripper_obj_farfrom_objects(self):
        gripper_far_tensor = torch.tensor([True], device=self.env.device).repeat(self.env.num_envs)
        for obj_name in self.obj_name:
            gripper_far_tensor = gripper_far_tensor & OU.gripper_obj_far(self.env, obj_name)
        return gripper_far_tensor
