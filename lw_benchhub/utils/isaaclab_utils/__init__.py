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

import typing
if typing.TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def get_robot_joint_target_from_scene(scene):  # : InteractiveScene):
    # only articulations have joints
    return {
        "joint_pos_target": scene.articulations["robot"].data.joint_pos_target.clone(),
        "joint_vel_target": scene.articulations["robot"].data.joint_vel_target.clone(),
        "joint_effort_target": scene.articulations["robot"].data.joint_effort_target.clone()
    }


def update_sensors(env: "ManagerBasedEnv", dt: float) -> None:
    for sensor in env.scene.sensors.values():
        sensor.update(dt, force_recompute=not env.scene.cfg.lazy_sensor_update)


class NoDeepcopyMixin:
    def __deepcopy__(self, memo):
        return self
