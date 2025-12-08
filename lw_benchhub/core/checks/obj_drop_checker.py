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

import torch

from lw_benchhub.core.checks.base_checker import BaseChecker


class ObjDropChecker(BaseChecker):
    type = "obj_drop"

    def __init__(self, warning_on_screen=False, velocity_threshold=1.0):
        super().__init__(warning_on_screen)
        self.velocity_threshold = velocity_threshold
        self._init_state()

    def _init_state(self):
        self._obj_drop_warning_frame_count = 0
        self._obj_drop_warning_text = ""
        self._obj_drop_violation_counts = {}
        self._prev_obj_poses = {}
        self._warned_objects = set()
        self.objects = {}
        self._objects_initialized = False
        self._frame_counter = 0
        self._last_result = {"success": True, "metrics": {}, "warning_text": ""}

    def reset(self):
        self._init_state()

    def _collect_objects_once(self, env):
        scene = env.scene
        obj_map, obj_getters = {}, {}

        for name, obj in scene.rigid_objects.items():
            obj_map[name] = "rigid"
            obj_getters[name] = lambda o=obj: o.data.body_com_pos_w[:, 0:1, 2]

        for name, obj in scene.deformable_objects.items():
            obj_map[name] = "deformable"
            obj_getters[name] = lambda o=obj: o.data.nodal_pos_w[:, 0:1, 2]

        for name, fixture_ref in env.cfg.fixture_refs.items():
            ref_name = fixture_ref.name
            try:
                if (
                    "articulation" in scene.state
                    and ref_name in scene.state["articulation"]
                    and "root_pose" in scene.state["articulation"][ref_name]
                ):
                    obj_map[name] = "fixture"
                    obj_getters[name] = lambda ref_name=ref_name, scene=scene: \
                        scene.state["articulation"][ref_name]["root_pose"][:, 2:3].unsqueeze(1)
            except Exception as e:
                print(f"[ObjDropChecker] Skip fixture {ref_name}: {e}")
                continue

        for name in env.scene.rigid_objects:
            if name in scene.rigid_objects:
                obj_getters[name] = lambda o=scene.rigid_objects[name]: o.data.body_com_pos_w[:, 0:1, 2]
            elif name in scene.deformable_objects:
                obj_getters[name] = lambda o=scene.deformable_objects[name]: o.data.nodal_pos_w[:, 0:1, 2]
            obj_map[name] = "cfg_object"

        self.objects = obj_map
        self._obj_z_getters = obj_getters
        self._objects_initialized = True

    def _get_obj_position_z(self, obj_name):
        return self._obj_z_tensors.get(obj_name, None)

    def _check(self, env):
        self._frame_counter += 1

        if self._frame_counter % 2 != 0:
            return self._last_result

        if self._frame_counter <= 60:
            return self._last_result

        if self._frame_counter <= 60:
            return self._last_result

        result = self._check_obj_drop(env)
        self._last_result = result
        return result

    @torch.no_grad()
    def _check_obj_drop(self, env):
        if not self._objects_initialized:
            self._collect_objects_once(env)

        if 0 < self._obj_drop_warning_frame_count < 50:
            self._obj_drop_warning_frame_count += 1
        else:
            self._obj_drop_warning_frame_count = 0
            self._obj_drop_warning_text = ""

        if not self.objects:
            return {"success": True, "metrics": {}, "warning_text": ""}

        dt = env.step_dt * 2
        velocity_threshold = -self.velocity_threshold

        obj_names = list(self._obj_z_getters.keys())

        current_zs = torch.stack([getter().view(-1) for getter in self._obj_z_getters.values()], dim=0)

        if hasattr(self, "_prev_zs_tensor"):
            prev_zs = self._prev_zs_tensor
        else:
            prev_zs = current_zs.clone()

        velocity_z = (current_zs - prev_zs) / dt
        dropped_mask = velocity_z < velocity_threshold

        self._prev_zs_tensor = current_zs.detach()

        drop_indices = torch.where(dropped_mask.any(dim=1))[0]
        newly_dropped = [obj_names[i] for i in drop_indices.tolist()]

        for name in newly_dropped:
            self._obj_drop_violation_counts[name] = (
                self._obj_drop_violation_counts.get(name, 0) + 1
            )

        if newly_dropped:
            if len(newly_dropped) == 1:
                msg = f"obj_drop Warning: <<{newly_dropped[0]}>> falling"
            else:
                msg = f"obj_drop Warning: {', '.join(newly_dropped[:3])} falling"
            self._obj_drop_warning_text = msg
            self._obj_drop_warning_frame_count = 1

        success = not any(v > 0 for v in self._obj_drop_violation_counts.values())
        metrics = self.get_obj_drop_metrics()
        metrics["success"] = success

        return {
            "success": success,
            "warning_text": self._obj_drop_warning_text,
            "metrics": metrics,
        }

    def get_obj_drop_metrics(self):
        return {
            name: count
            for name, count in self._obj_drop_violation_counts.items()
            if count > 0
        }
