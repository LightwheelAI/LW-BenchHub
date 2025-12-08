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


class SnackSorting(ManipulateDrawer):
    """
    Snack Sorting: composite task for Tidying Cabinets And Drawers activity.

    Simulates the task of placing snacks in the bowl.

    Steps:
        Place the bar in the bowl and close the drawer.
    """

    task_name: str = "SnackSorting"
    behavior: str = "close"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.drawer)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Place the bar in the bowl and close the drawer."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = []
        cfgs.append(
            dict(
                name="bar",
                obj_groups="bar",
                graspable=True,
                # have to make sure that the sampled object can fit inside the drawer hence the max_size being 0.1 in the z axis
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.20, 0.25),
                    # put object towards the front of the drawer
                    pos=(None, -0.75),
                ),
            )
        )

        cfgs.append(
            dict(
                name="dist",
                obj_groups="all",
                max_size=(None, None, 0.10),
                placement=dict(
                    fixture=self.drawer,
                    size=(0.30, 0.30),
                    pos=(None, 1),
                ),
            )
        )

        cfgs.append(
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.counter,
                    sample_region_kwargs=dict(ref=self.drawer),
                    size=(0.15, 0.10),
                    offset=(0.0, 0.075),
                    pos=("ref", -1.0),
                    ensure_object_boundary_in_range=False,
                ),
            )
        )
        return cfgs

    def _check_success(self, env):
        bars_in_bowl = OU.check_obj_in_receptacle(env, "bar", "bowl")

        # user super class to make sure that the drawer is closed
        door_closed = super()._check_success(env)

        return bars_in_bowl & door_closed
