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

def rl_on(task=None, embodiment=None):
    from lw_benchhub.core.tasks.base import LwTaskBase
    from lw_benchhub.core.robots.robot_arena_base import LwEmbodimentBase

    if task is not None:
        if not issubclass(task, LwTaskBase):
            raise TypeError(f"task must be a subclass of LwTaskBase, got {type(task)}")

    if embodiment is not None:
        if not issubclass(embodiment, LwEmbodimentBase):
            raise TypeError(f"embodiment must be a subclass of LwEmbodimentBase, got {type(embodiment)}")

    def wrapper(cls):
        if task:
            cls._rl_on_tasks.append(task)
        if embodiment:
            cls._rl_on_embodiments.append(embodiment)
        return cls

    return wrapper
