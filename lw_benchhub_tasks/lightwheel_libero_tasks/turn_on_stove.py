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

import numpy as np
import torch

import lw_benchhub.utils.object_utils as OU
from lw_benchhub.core.models.fixtures import FixtureType
from lw_benchhub.core.tasks.base import LwTaskBase


class L90K3TurnOnTheStove(LwTaskBase):
    task_name: str = "L90K3TurnOnTheStove"

    enable_fixtures = ["stove", "mokapot_1"]
    removable_fixtures = ["mokapot_1"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("island", dict(id=FixtureType.TABLE))
        self.mokapot = self.register_fixture_ref("mokapot", dict(id=FixtureType.MOKA_POT))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn on the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        placement = dict(
            fixture=self.counter,
            size=(0.6, 0.5),
            pos=(-0.5, -0.2),
            ensure_object_boundary_in_range=False,
        )
        cfgs.append(
            dict(
                name="chefmate_8_frypan",
                obj_groups="pot",
                graspable=True,
                asset_name="Pot086.usd",
                placement=placement,
            )
        )
        return cfgs

    def _check_success(self, env):
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_on = (abs_knob >= lower) & (abs_knob <= upper)
            knob_success = knob_success | knob_on
        return knob_success & OU.gripper_obj_far(env, self.stove.name, th=0.4)


class L90K9TurnOnTheStove(LwTaskBase):
    task_name: str = "L90K9TurnOnTheStove"

    enable_fixtures = ["stove"]

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref("island", dict(id=FixtureType.TABLE))
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))

        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn on the stove."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        placement = dict(
            fixture=self.counter,
            size=(0.6, 0.5),
            pos=(-0.5, 0.2),
            rotation=np.pi,
            ensure_object_boundary_in_range=False,
        )
        cfgs.append(
            dict(
                name="wooden_two_layer_shelf",
                obj_groups="shelf",
                graspable=True,
                object_scale=0.8,
                asset_name="Shelf073.usd",
                placement=dict(
                    fixture=self.counter,
                    rotation=-np.pi / 2,
                    size=(0.80, 0.80),
                    pos=(0.3, 0.0),
                ),
            )
        )
        cfgs.append(
            dict(
                name="chefmate_8_frypan",
                obj_groups="pot",
                graspable=True,
                asset_name="Pot086.usd",
                placement=dict(
                    fixture=self.counter,
                    size=(0.6, 0.5),
                    pos=(-0.5, 0.2),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        cfgs.append(
            dict(
                name="white_bowl",
                obj_groups="bowl",
                graspable=True,
                asset_name="Bowl011.usd",
                placement=placement,
            )
        )
        return cfgs

    def _check_success(self, env):
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_on = (abs_knob >= lower) & (abs_knob <= upper)
            knob_success = knob_success | knob_on
        return knob_success & OU.gripper_obj_far(env, self.stove.name, th=0.4)


class L90K3TurnOnTheStoveAndPutTheFryingPanOnIt(L90K3TurnOnTheStove):
    task_name: str = "L90K3TurnOnTheStoveAndPutTheFryingPanOnIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn on the stove and put the frying pan on it."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        placement = dict(
            fixture=self.counter,
            size=(0.6, 0.5),
            pos=(-0.5, 0.0),
            ensure_object_boundary_in_range=False,
        )
        cfgs.append(
            dict(
                name="chefmate_8_frypan",
                obj_groups="pot",
                graspable=True,
                asset_name="Pot086.usd",
                placement=placement,
            )
        )
        return cfgs

    def _check_success(self, env):
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_on = (abs_knob >= lower) & (abs_knob <= upper)
            knob_success = knob_success | knob_on
        pot_success = torch.tensor([False] * env.num_envs, device=env.device)
        for i in range(env.num_envs):
            pot_success[i] = torch.as_tensor(
                OU.point_in_fixture(OU.get_object_pos(env, "chefmate_8_frypan")[i], self.stove, only_2d=True),
                dtype=torch.bool,
                device=env.device,
            )
        return knob_success & pot_success & OU.gripper_obj_far(env, "chefmate_8_frypan", 0.35)


class L10K3TurnOnTheStoveAndPutTheMokaPotOnIt(L90K3TurnOnTheStove):
    task_name: str = "L10K3TurnOnTheStoveAndPutTheMokaPotOnIt"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Turn on the stove and put the moka pot on it."
        return ep_meta

    def _check_success(self, env):
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_on = (abs_knob >= lower) & (abs_knob <= upper)
            knob_success = knob_success | knob_on
        mokapot_pos = env.scene.articulations[self.mokapot.name].data.root_pos_w[0, :].cpu().numpy()
        moka_success = OU.point_in_fixture(mokapot_pos, self.stove, only_2d=True)
        moka_success = torch.tensor([moka_success], device=env.device).repeat(env.num_envs)
        return knob_success & moka_success & OU.gripper_obj_far(env, "mokapot_1_front_group_1", 0.35)


class L90K9TurnOnTheStoveAndPutTheFryingPanOnIt(LwTaskBase):
    task_name: str = 'L90K9TurnOnTheStoveAndPutTheFryingPanOnIt'
    EXCLUDE_LAYOUTS: list = [63, 64]
    enable_fixtures: list[str] = ["stovetop"]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"put the frying pan on top of the cabinet."
        return ep_meta

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.dining_table = self.register_fixture_ref("dining_table", dict(id=FixtureType.TABLE, size=(1.0, 0.35)),)
        self.stove = self.register_fixture_ref("stove", dict(id=FixtureType.STOVE))
        self.init_robot_base_ref = self.dining_table
        self.shelf = "shelf"
        self.frying_pan = "frying_pan"
        self.bowl = "bowl"

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name=self.shelf,
                obj_groups="shelf",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(-0.75, 0.25),
                    rotation=np.pi / 2,
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Shelf073.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.bowl,
                obj_groups="bowl",
                graspable=True,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.25, 0.25),
                    pos=(1.0, -0.0),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Bowl009.usd",
            )
        )

        cfgs.append(
            dict(
                name=self.frying_pan,
                obj_groups="pot",
                graspable=True,
                object_scale=0.8,
                placement=dict(
                    fixture=self.dining_table,
                    size=(0.5, 0.25),
                    pos=(0.5, -0.25),
                    ensure_object_boundary_in_range=False,
                ),
                asset_name="Pot086.usd",
            )
        )

        return cfgs

    def _check_success(self, env):
        knobs_state = self.stove.get_knobs_state(env=env)
        knob_success = torch.tensor([False], device=env.device).repeat(env.num_envs)
        for knob_name, knob_value in knobs_state.items():
            abs_knob = torch.abs(knob_value)
            lower, upper = 0.35, 2 * np.pi - 0.35
            knob_on = (abs_knob >= lower) & (abs_knob <= upper)
            knob_success = knob_success | knob_on
        pot_success = torch.tensor([False] * env.num_envs, device=env.device)
        for i in range(env.num_envs):
            pot_success[i] = torch.as_tensor(
                OU.point_in_fixture(OU.get_object_pos(env, self.frying_pan)[i], self.stove, only_2d=True),
                dtype=torch.bool,
                device=env.device,
            )
        return knob_success & pot_success & OU.gripper_obj_far(env, self.frying_pan, 0.35)
