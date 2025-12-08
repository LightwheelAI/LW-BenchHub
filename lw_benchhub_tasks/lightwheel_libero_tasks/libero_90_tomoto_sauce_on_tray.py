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

from .libero_10_put_in_basket import Libero10PutInBasket


class L90L3PickUpTheTomatoSauceAndPutItInTheTray(Libero10PutInBasket):
    """
    L90L3PickUpTheTomatoSauceAndPutItInTheTray: pick up the tomato_sauce and put it in the wooden_tray

    Steps:
        pick up the tomato_sauce
        put the tomato_sauce in the wooden_tray

    """

    task_name: str = "L90L3PickUpTheTomatoSauceAndPutItInTheTray"

    def _setup_kitchen_references(self, scene):
        super()._setup_kitchen_references(scene)
        # Define wooden_tray early so it's available in _check_success
        self.wooden_tray = "wooden_tray"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = f"Pick up the tomato_sauce, and put it in the wooden_tray."
        return ep_meta

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        pop_objs = [self.milk, self.orange_juice]
        cfg_index = 0
        while cfg_index < len(cfgs):
            if cfgs[cfg_index]['name'] == self.basket:
                # Replace basket with wooden_tray
                cfgs[cfg_index]['name'] = self.wooden_tray
                cfgs[cfg_index]['asset_name'] = "Tray016.usd"
                cfgs[cfg_index]['object_scale'] = 0.6
            if cfgs[cfg_index]['name'] in pop_objs:
                cfgs.pop(cfg_index)
            else:
                cfg_index += 1
        return cfgs

    def _check_success(self, env):
        if self.is_replay_mode:
            self._get_obj_cfgs()
        success_tomato_sauce = OU.check_place_obj1_on_obj2(
            env,
            self.tomato_sauce,
            self.wooden_tray,
            th_z_axis_cos=0.8,  # verticality
            th_xy_dist=0.4,    # within 0.4 diameter
            th_xyz_vel=0.5     # velocity vector length less than 0.5
        )

        return success_tomato_sauce
