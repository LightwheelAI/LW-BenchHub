# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class LiberoBlackBowlAndPlateBase(LwTaskBase):
    """
    LiberoBlackBowlAndPlateBase: base class for all libero black bowl and plate tasks
    """

    task_name: str = "LiberoBlackBowlAndPlateBase"
    enable_fixtures = ['storage_furniture']
    fix_object_pose_cfg: dict = None

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("table", dict(id=FixtureType.TABLE))
        self.drawer = self.register_fixture_ref("storage_furniture", dict(id=FixtureType.STORAGE_FURNITURE))
        self.init_robot_base_ref = self.counter
        self.akita_black_bowl_front = "akita_black_bowl_front"
        self.akita_black_bowl_middle = "akita_black_bowl_middle"
        self.akita_black_bowl_back = "akita_black_bowl_back"
        self.plate = "plate"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.top_joint_name = list(self.drawer._joint_infos.keys())[0]

    def _get_obj_cfgs(self):
        cfgs = []

        base_x = self.rng.uniform(-0.3, -0.15)
        middle_y = self.rng.uniform(-0.6, -0.4)
        spacing = self.rng.uniform(0.6, 1.0)

        front_pos = (base_x, middle_y + spacing)
        middle_pos = (base_x, middle_y)
        back_pos = (base_x, middle_y - spacing)

        plate_placement = dict(
            fixture=self.counter,
            size=(0.28, 0.28),
            pos=(0.15, -0.2),
            margin=0.02,
            ensure_valid_placement=True,
        )
        black_bowl_front_placement = dict(
            fixture=self.counter,
            size=(0.22, 0.22),
            pos=front_pos,
            margin=0.02,
            ensure_valid_placement=True,
        )
        black_bowl_middle_placement = dict(
            fixture=self.counter,
            size=(0.22, 0.22),
            pos=middle_pos,
            margin=0.02,
            ensure_valid_placement=True,
        )
        black_bowl_back_placement = dict(
            fixture=self.counter,
            size=(0.22, 0.22),
            pos=back_pos,
            margin=0.02,
            ensure_valid_placement=True,
        )

        cfgs.append(
            dict(
                name=self.akita_black_bowl_back,
                obj_groups='akita_black_bowl',
                graspable=True,
                placement=black_bowl_back_placement,
                asset_name='Bowl008.usd',
                object_scale=0.8,
            )
        )
        cfgs.append(
            dict(
                name=self.plate,
                obj_groups=self.plate,
                graspable=True,
                placement=plate_placement,
                asset_name='Plate012.usd',
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl_middle,
                obj_groups='akita_black_bowl',
                graspable=True,
                placement=black_bowl_middle_placement,
                asset_name='Bowl008.usd',
                object_scale=0.8,
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl_front,
                obj_groups='akita_black_bowl',
                graspable=True,
                placement=black_bowl_front_placement,
                asset_name='Bowl008.usd',
                object_scale=0.8,
            )
        )
        return cfgs


class L90K2PutTheMiddleBlackBowlOnThePlate(LiberoBlackBowlAndPlateBase):
    """
    L90K2PutTheMiddleBlackBowlOnThePlate: put the black bowl in the middle on the plate

    Steps:
        pick up the black bowl
        put the black bowl in the middle on the plate

    """

    task_name: str = "L90K2PutTheMiddleBlackBowlOnThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the black bowl in the middle on the plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl_middle,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.7,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        return success


class L90K2PutTheBlackBowlAtTheFrontOnThePlate(LiberoBlackBowlAndPlateBase):
    """
    L90K2PutTheBlackBowlAtTheFrontOnThePlate: put the black bowl in the front on the plate

    Steps:
        pick up the black bowl
        put the black bowl in the front on the plate

    """

    task_name: str = "L90K2PutTheBlackBowlAtTheFrontOnThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the black bowl and put it in the front on the plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl_front,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.7,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        return success


class L90K2PutTheBlackBowlAtTheBackOnThePlate(LiberoBlackBowlAndPlateBase):
    task_name: str = "L90K2PutTheBlackBowlAtTheBackOnThePlate"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Put the black bowl at the front on the plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl_front,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        success1 = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl_back,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        print(f"success1 state: {success1}")
        success2 = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl_middle,
            self.plate,
            th_z_axis_cos=0.95,  # verticality
            th_xy_dist=0.5,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )
        print(f"success2 state: {success2}")
        return success | success1 | success2


class L90K2OpenTheTopDrawerOfTheCabinet(L90K2PutTheBlackBowlAtTheBackOnThePlate):
    task_name: str = "L90K2OpenTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Open the top drawer of the cabinet."
        return ep_meta

    def _check_success(self, env):
        return self.drawer.is_open(env, [self.top_joint_name], th=0.5) & OU.gripper_obj_far(env, self.drawer.name, th=0.4)


class L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl(LiberoBlackBowlAndPlateBase):
    """
    L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl: put the black bowl in the middle on the front bowl

    Steps:
        pick up the black bowl
        put the black bowl in the middle on the back bowl

    """

    task_name: str = "L90K2StackTheMiddleBlackBowlOnTheBackBlackBowl"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Stack the black bowl in the middle on the black bowl at the back."
        return ep_meta

    def _check_success(self, env):

        return OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl_middle,
            self.akita_black_bowl_back,
            th_z_axis_cos=0.5,   # verticality - allows up to 60 degree tilt
            th_xy_dist=1.0,      # xy distance threshold - very relaxed, within bowl diameter
            th_xyz_vel=0.5,      # velocity threshold - relaxed
            gipper_th=0.3        # gripper distance threshold - more relaxed
        )


class L90K5PutTheBlackBowlOnThePlate(LiberoBlackBowlAndPlateBase):
    """
    L90K5PutTheBlackBowlOnThePlate: put the black bowl on the plate

    Steps:
        pick up the black bowl
        put the black bowl on the plate

    """

    task_name: str = "L90K5PutTheBlackBowlOnThePlate"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        # names used in success checks
        self.akita_black_bowl = "akita_black_bowl"
        self.plate = "plate"
        self.ketchup = "ketchup"

    def _setup_scene(self, env, env_ids=None):
        """
        Resets simulation internal configurations.
        """
        super()._setup_scene(env, env_ids)
        self.top_drawer_joint_name = list(self.drawer._joint_infos.keys())[0]
        self.drawer.set_joint_state(0.8, 1.0, env, [self.top_drawer_joint_name])

    def _get_obj_cfgs(self):
        cfgs = []

        plate_placement = dict(
            fixture=self.counter,
            pos=(0.55, -0.85),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.35, 0.35),
        )
        bowl_placement = dict(
            fixture=self.counter,
            pos=(-0.05, -0.25),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.35, 0.35),
        )
        ketchup_placement = dict(
            fixture=self.counter,
            pos=(0.20, -0.8),
            margin=0.02,
            ensure_valid_placement=True,
            size=(0.25, 0.25),
        )

        cfgs.append(
            dict(
                name=self.plate,
                obj_groups="plate",
                graspable=False,
                placement=plate_placement,
                asset_name="Plate012.usd",
                init_robot_here=True,
            )
        )
        cfgs.append(
            dict(
                name=self.akita_black_bowl,
                obj_groups="bowl",
                graspable=True,
                placement=bowl_placement,
                asset_name="Bowl008.usd",
            )
        )
        cfgs.append(
            dict(
                name=self.ketchup,
                obj_groups="ketchup",
                graspable=True,
                placement=ketchup_placement,
                asset_name="Ketchup003.usd",
            )
        )

        return cfgs

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl on the plate."
        return ep_meta

    def _check_success(self, env):
        success = OU.check_place_obj1_on_obj2(
            env,
            self.akita_black_bowl,
            self.plate,
            th_z_axis_cos=0.95,
            th_xy_dist=0.25,
            th_xyz_vel=0.5,
        )
        return success


class L90K5PutTheBlackBowlOnTopOfTheCabinet(L90K5PutTheBlackBowlOnThePlate):
    task_name: str = "L90K5PutTheBlackBowlOnTopOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the black bowl on top of the cabinet."
        return ep_meta

    def _check_success(self, env):
        import torch
        bowl_poses = OU.get_object_pos(env, self.akita_black_bowl)
        bowl_success_tensor = torch.tensor([False] * env.num_envs, device=env.device)
        for i, bowl_pos in enumerate(bowl_poses):
            bowl_success = OU.point_in_fixture(bowl_pos, self.drawer, only_2d=True)
            bowl_success_tensor[i] = torch.as_tensor(bowl_success, dtype=torch.bool, device=env.device)

        result = bowl_success_tensor & OU.gripper_obj_far(env, self.akita_black_bowl)
        return result


class L90K5PutTheKetchupInTheTopDrawerOfTheCabinet(L90K5PutTheBlackBowlOnThePlate):
    task_name: str = "L90K5PutTheKetchupInTheTopDrawerOfTheCabinet"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Put the ketchup in the top drawer of the cabinet."
        return ep_meta

    def _check_success(self, env):
        ketchup_success = OU.obj_inside_of(env, self.ketchup, self.drawer)
        return ketchup_success & OU.gripper_obj_far(env, self.ketchup)
