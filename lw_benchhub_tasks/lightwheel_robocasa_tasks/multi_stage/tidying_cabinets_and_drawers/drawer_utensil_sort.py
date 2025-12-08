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
from lw_benchhub_tasks.lightwheel_robocasa_tasks.single_stage.kitchen_drawer import ManipulateDrawer


class DrawerUtensilSort(ManipulateDrawer):
    """
    Drawer Utensil Sort: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of placing utensils in the drawer.

    Steps:
        Open the drawer and push the utensils inside it.

    Args:
        drawer_id (int): Enum which serves as a unique identifier for different
            drawer types. Used to choose the drawer in which the utensils are
            placed. TOP_DRAWER specifies the uppermost drawer.
    """

    task_name: str = "DrawerUtensilSort"
    behavior: str = "open"
    draw_id: int = FixtureType.TOP_DRAWER

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer, size=(0.2, 0.2))
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"{self.behavior} the {self.drawer_side} drawer and push the utensils inside it."
        ep_meta["lang"] = ep_meta["lang"][0].capitalize() + ep_meta["lang"][1:]
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []

        cfgs.append(
            dict(
                name="utensil1",
                obj_groups="utensil",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.3, 0.4),
                    pos=("ref", -1.0),
                    offset=(-0.05, 0.0),
                ),
            )
        )

        cfgs.append(
            dict(
                name="utensil2",
                obj_groups="utensil",
                graspable=False,
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(
                        ref=self.drawer,
                    ),
                    size=(0.3, 0.4),
                    pos=("ref", -1.0),
                    offset=(0.05, 0.0),
                ),
            )
        )

        return cfgs

    def _check_success(self, env):

        # need to make sure utensils not on the counter. Since the top drawer (when closed) is so close to the counter,
        # its bounding box may overlap with the top of the counter. So when the utensil is on the counter it may be falsely identified as inside the drawer
        utensil1_inside_drawer = OU.obj_inside_of(
            env, "utensil1", self.drawer
        ) & ~ OU.check_obj_fixture_contact(env, "utensil1", self.counter)
        utensil2_inside_drawer = OU.obj_inside_of(
            env, "utensil2", self.drawer
        ) & ~ OU.check_obj_fixture_contact(env, "utensil2", self.counter)

        gripper_obj_far = OU.gripper_obj_far(
            env, obj_name="utensil1"
        ) & OU.gripper_obj_far(env, obj_name="utensil2")
        return utensil1_inside_drawer & utensil2_inside_drawer & gripper_obj_far
