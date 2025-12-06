import torch
from lwlab.core.tasks.base import LwLabTaskBase
from lwlab.core.models.fixtures import FixtureType
import lwlab.utils.object_utils as OU
from lwlab.core.models.fixtures.counter import Counter
import numpy as np
import copy


class L90K4PutTheWineBottleOnTheWineRack(LwLabTaskBase):
    task_name: str = "L90K4PutTheWineBottleOnTheWineRack"
    enable_fixtures = ["winerack", "storage_furniture"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref(
            "dining_table",
            dict(id=FixtureType.TABLE, size=(1.0, 0.35)),
        )
        self.winerack = self.register_fixture_ref(
            "winerack",
            dict(id=FixtureType.WINE_RACK),
        )
        self.init_robot_base_ref = self.dining_table

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the wine bottle on the wine rack."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name=f"akita_black_bowl",
                obj_groups="bowl",
                graspable=True,
                washable=True,
                asset_name="Bowl008.usd",
                init_robot_here=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.50, 0.35),
                    margin=0.02,
                    pos=(-0.3, -0.7),
                    ensure_object_boundary_in_range=False
                ),
            )
        )
        cfgs.append(
            dict(
                name=f"wine_bottle",
                obj_groups="bottle",
                graspable=True,
                washable=True,
                asset_name="Bottle054.usd",
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.30, 0.35),
                    margin=0.02,
                    pos=(0.1, -0.6),
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        wine_bottle_pos = OU.get_object_pos(env, "wine_bottle")
        gripper_far = OU.gripper_obj_far(env, "wine_bottle", th=0.3)

        # Check if wine bottle is stable (not moving) - more relaxed threshold
        bottle_stable = OU.check_object_stable(env, "wine_bottle", threshold=0.3)

        result_tensor = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        for i in range(env.num_envs):
            # Check 1: Wine bottle is in wine rack area (xy check)
            in_rack = OU.point_in_fixture(wine_bottle_pos[i], self.winerack, only_2d=True)

            # All conditions must be met - convert numpy bool to tensor bool
            in_rack_tensor = torch.tensor(bool(in_rack), dtype=torch.bool, device=env.device)
            result_tensor[i] = in_rack_tensor & gripper_far[i] & bottle_stable[i]

        return result_tensor
