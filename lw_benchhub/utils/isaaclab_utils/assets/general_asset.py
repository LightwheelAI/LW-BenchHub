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


from isaaclab.assets.articulation import Articulation
from isaaclab.assets.asset_base import AssetBase
from isaaclab.assets.rigid_object import RigidObject

from lw_benchhub.utils.usd_utils import OpenUsd as usd


class GeneralAsset(AssetBase):

    def __init__(self, cfg, env_regex_ns):
        super().__init__(cfg)
        self.cfg = cfg
        self.rigid_objects = list()
        self.articulations = list()
        self.env_regex_ns = env_regex_ns
        self.stage = usd.get_stage(self.cfg.spawn.usd_path)
        self._parse_usd_and_create_subassets()

    def _make_articulation_cfg(self, prim):
        from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
        pos, quat, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat is None:
            print(f"GeneralAsset: {prim.GetName()} none pos or quat")
            return None
        joints = usd.get_all_joints_without_fixed(prim)
        if not joints:
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
        prim_path = prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        return ArticulationCfg(prim_path=prim_path,
                               spawn=None,
                               init_state=ArticulationCfg.InitialStateCfg(
                                   pos=pos,
                                   rot=quat,
                               ),
                               actuators={},
                               )

    def _make_rigidbody_cfg(self, prim):
        from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
        pos, quat, _ = usd.get_prim_pos_rot_in_world(prim)
        if pos is None or quat is None:
            print(f"GeneralAsset: {prim.GetName()} none pos or quat")
            return None
        orin_prim_path = prim.GetPath().pathString
        name = orin_prim_path.split("/")[-1]
        sub_prim_path = orin_prim_path[orin_prim_path.find('/', 1) + 1:]
        prim_path = f"{{ENV_REGEX_NS}}/Scene/{sub_prim_path}"
        prim_path = prim_path.format(ENV_REGEX_NS=self.env_regex_ns)
        return RigidObjectCfg(prim_path=prim_path,
                              spawn=None,
                              init_state=RigidObjectCfg.InitialStateCfg(
                                  pos=pos,
                                  rot=quat,
                              ),
                              )

    def _parse_usd_and_create_subassets(self):

        prims = usd.get_all_prims(self.stage)
        articulation_sub_prims = list()
        for prim in prims:
            if usd.is_articulation_root(prim):
                art_cfg = self._make_articulation_cfg(prim)
                if art_cfg is None:
                    continue
                self.articulations.append(Articulation(art_cfg))
                articulation_sub_prims.extend(usd.get_all_prims(self.stage, prim))
        for prim in prims:
            if usd.is_rigidbody(prim):
                if prim in articulation_sub_prims:
                    continue
                rb_cfg = self._make_rigidbody_cfg(prim)
                if rb_cfg is None:
                    continue
                self.rigid_objects.append(RigidObject(rb_cfg))

    def reset(self, env_ids=None):
        for art in self.articulations:
            art.reset(env_ids)
        for rb in self.rigid_objects:
            rb.reset(env_ids)

    def write_data_to_sim(self):
        for art in self.articulations:
            art.write_data_to_sim()
        for rb in self.rigid_objects:
            rb.write_data_to_sim()

    def update(self, dt):
        for art in self.articulations:
            art.update(dt)
        for rb in self.rigid_objects:
            rb.update(dt)

    def get_all_joint_names(self):
        names = []
        for art in self.articulations:
            names.extend(art.data.joint_names)
        return names

    def get_all_body_names(self):
        names = []
        for rb in self.rigid_objects:
            names.extend(rb.data.body_names)
        for art in self.articulations:
            names.extend(art.data.body_names)
        return names

    def _initialize_impl(self):
        pass

    @property
    def data(self):
        pass

    @property
    def num_instances(self):
        pass
