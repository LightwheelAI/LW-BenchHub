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

class MotionMetric():
    def __init__(self):
        pass

    def _compute_episode_metrics(self, env, episode_info):
        return episode_info["articulation"]["robot"]["joint_velocity"][0]

    @classmethod
    def validate_episode(
            cls,
            episode_metrics,
            vel_limit=None,
    ):
        results = dict()
        robot_joint_velocity = episode_metrics["articulation"]["robot"]["joint_velocity"][0]
        for vel in robot_joint_velocity:
            if vel > vel_limit:
                results["joint_velocity"] = {"success": False, "feedback": f"Robot's joint velocity is too high ({vel}), must be <= {vel_limit}"}
                break
        if "joint_velocity" not in results:
            results["joint_velocity"] = {"success": True, "feedback": None}

        return results
