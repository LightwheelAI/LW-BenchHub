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

from lw_benchhub.core.checks.base_checker import BaseChecker
from lw_benchhub.utils.object_utils import calculate_contact_force


class ClippingChecker(BaseChecker):
    type = "clipping"

    def __init__(self, warning_on_screen=False):
        super().__init__(warning_on_screen)
        self._init_state()

    def _init_state(self, w=3, spike_th=100, mean_force_th=150, stable_var_th=5.0):
        self._clipping_warning_frame_count = 0
        self._clipping_warning_text = ""
        self._clipping_counts = 0
        self._clipping_frames = []
        self.forces = []
        self.spike_th = spike_th
        self.mean_force_th = mean_force_th
        self.stable_var_th = stable_var_th
        self.w = w
        self.last_force = 0
        self._frame_count = 0
        self._force_events = []
        self._objects_initialized = False
        self._last_result = {"success": True, "metrics": {}, "warning_text": ""}
        self.history_metrics = {
            # "force": [],
            # "mean_force": [],
            # "force_variance": [],
            # "delta_force": [],
            "clipping_times": [],
            "clipping_frames": [],
        }

    def reset(self):
        self._init_state()

    def _check(self, env):
        self._frame_count += 1

        if self._frame_count % 2 != 0:
            return self._last_result

        result = self._check_clipping(env)
        self._last_result = result
        return result

    def _check_clipping(self, env):
        """
        Check if the clipping happens.
        Calculates the clipping status.

        Returns:
            dict: Dictionary containing total clipping times.
        """
        if self._clipping_counts is None:
            self._clipping_counts = 0

        if self._clipping_warning_frame_count is not None and 50 > self._clipping_warning_frame_count > 0:
            self._clipping_warning_frame_count += 1
        else:
            self._clipping_warning_frame_count = 0
            self._clipping_warning_text = ""

        try:
            left_gripper = "left_gripper"
            right_gripper = "right_gripper"

            current_frame = self._frame_count

            left_force = calculate_contact_force(env, left_gripper)
            right_force = calculate_contact_force(env, right_gripper)
            # Handle both scalar and multi-environment tensors
            if left_force.dim() == 0:
                left_force_val = left_force.item()
            else:
                left_force_val = left_force[0].item()

            if right_force.dim() == 0:
                right_force_val = right_force.item()
            else:
                right_force_val = right_force[0].item()

            left_force_val = left_force.item() if left_force.dim() == 0 else left_force[0].item()
            right_force_val = right_force.item() if right_force.dim() == 0 else right_force[0].item()
            force = max(left_force_val, right_force_val)

            self.forces.append(force)
            if len(self.forces) > self.w:
                self.forces.pop(0)

            mu = sum(self.forces) / len(self.forces)
            var = sum((x - mu)**2 for x in self.forces) / len(self.forces)

            deltaF = abs(force - getattr(self, "last_force", 0.0))
            self.last_force = force

            last_event = self._force_events[-1] if self._force_events else None
            if (last_event is None) or (last_event.get("end") is not None):
                if force > 2:
                    self._force_events.append({"start": current_frame, "end": None, "triggered": False, "candidates": []})
            else:
                if force < 2 and last_event.get("end") is None:
                    last_event["end"] = current_frame
                    for f in last_event["candidates"]:
                        if last_event["start"] + 50 <= f <= last_event["end"] - 35:
                            if (self._clipping_warning_frame_count == 0):
                                self._clipping_warning_text = "clipping Warning: Contact forces too high, there may be << Clipping >> happens"
                                self._clipping_counts += 1
                                self._clipping_frames.append(current_frame)
                                self._clipping_warning_frame_count += 1
                                last_event["triggered"] = True
                                break
                    last_event["candidates"] = []

            condition_now = (force > 100 or deltaF > self.spike_th or var > self.stable_var_th)

            for ev in self._force_events:
                if ev.get("triggered"):
                    continue

                start = ev["start"]
                end = ev.get("end")
                window_start = start + 35

                if condition_now and current_frame >= window_start:
                    if end is not None:
                        if current_frame <= end - 35:
                            if not ev["candidates"] or ev["candidates"][-1] != current_frame:
                                ev["candidates"].append(current_frame)
                    else:
                        if not ev["candidates"] or ev["candidates"][-1] != current_frame:
                            ev["candidates"].append(current_frame)

                if ev["candidates"]:
                    if end is None:
                        f = ev["candidates"][0]
                        if current_frame >= f + 35:
                            if self._clipping_warning_frame_count == 0:
                                self._clipping_warning_text = "clipping Warning: Contact forces too high, there may be << Clipping >> happens"
                                self._clipping_counts += 1
                                self._clipping_frames.append(current_frame)
                                self._clipping_warning_frame_count += 1
                                ev["triggered"] = True
                                break
                            else:
                                ev["triggered"] = True
                                break
                    else:
                        for f in ev["candidates"]:
                            if start + 35 <= f <= end - 35:
                                if self._clipping_warning_frame_count == 0:
                                    self._clipping_warning_text = "clipping Warning: Contact forces too high, there may be << Clipping >> happens"
                                    self._clipping_counts += 1
                                    self._clipping_frames.append(current_frame)
                                    self._clipping_warning_frame_count += 1
                                    ev["triggered"] = True
                                    break
                                else:
                                    ev["triggered"] = True
                                    break
                        if ev.get("end") is not None:
                            ev["candidates"] = []

            if len(self._force_events) > 500:
                self._force_events = self._force_events[-200:]

            success = self._clipping_counts == 0

            # self.history_metrics["force"].append(float(force))
            # self.history_metrics["mean_force"].append(float(mu))
            # self.history_metrics["force_variance"].append(float(var))
            # self.history_metrics["delta_force"].append(float(deltaF))
            self.history_metrics["clipping_times"] = int(self._clipping_counts)
            self.history_metrics["clipping_frames"] = list(self._clipping_frames)
            self.history_metrics["success"] = success

            result = {
                "success": success,
                "warning_text": self._clipping_warning_text,
                "metrics": self.history_metrics
            }

            return result
        except Exception as e:
            import traceback
            print(f"Error in _check_clipping: {traceback.format_exc()}")
            return {"success": False, "warning_text": None, "metrics": {}}
