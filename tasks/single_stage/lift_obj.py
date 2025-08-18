# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from lwlab.core.models.fixtures import FixtureType
from lwlab.core.scenes.kitchen.kitchen import RobocasaKitchenEnvCfg
from lwlab.core.tasks.base import BaseTaskEnvCfg

##
# Scene definition
##
# Increase PhysX GPU aggregate pairs capacity to avoid simulation errors
sim_utils.simulation_context.gpu_total_aggregate_pairs_capacity = 160000


@configclass
class LiftObj(BaseTaskEnvCfg, RobocasaKitchenEnvCfg):
    """
    Class encapsulating the atomic pick and place tasks.

    Args:
        obj_groups (str): Object groups to sample the target object from.

        exclude_obj_groups (str): Object groups to exclude from sampling the target object.
    """
    counter_id: FixtureType = FixtureType.COUNTER
    task_name: str = "LiftObj"

    def _setup_kitchen_references(self):
        """
        Setup the kitchen references for the counter to cabinet pick and place task:
        The cabinet to place object in and the counter to initialize it on
        """
        super()._setup_kitchen_references()

        self.counter = self.register_fixture_ref("counter", dict(id=self.counter_id, fix_id=1))
        self.useful_fixture_names = [self.counter.name]
        self.init_robot_base_ref = self.counter

    def _get_obj_cfgs(self):

        cfgs = []

        cfgs.append(
            dict(
                name="object",
                obj_groups=os.path.abspath("apple_11"),
                graspable=True,
                placement=dict(
                    fixture=self.counter,
                    size=(0.30, 0.30),
                    pos=(0, -1.0),
                ),
            )
        )

        return cfgs

    def _check_success(self):
        """
        Check if the counter to cabinet pick and place task is successful.
        Checks if the object is inside the cabinet and the gripper is far from the object.

        Returns:
            bool: True if the task is successful, False otherwise
        """

        object_height = self.env.scene['object'].data.root_pos_w[:, 2]
        is_height_sufficient = (object_height >= 1.0)
        object_pos = self.env.scene['object'].data.root_pos_w                        # [N, 3]
        ee_pos = self.env.scene["ee_frame"].data.target_pos_w[..., 0, :]  # [N, 3]
        gripper_force = torch.linalg.norm(self.env.scene.sensors["right_gripper_contact"]._data.net_forces_w[:, 0, :], dim=1)
        object_force = torch.linalg.norm(self.env.scene.sensors["object_contact"]._data.net_forces_w[:, 0, :], dim=1)
        distance = torch.norm(object_pos - ee_pos, dim=1)
        is_gripper_force = gripper_force > 0.1
        is_object_force = object_force > 0.1
        is_near_gripper = distance < 0.03
        is_grasping = is_gripper_force & is_object_force & is_near_gripper
        return is_grasping & is_height_sufficient
