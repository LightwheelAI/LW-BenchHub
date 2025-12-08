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

class BaseChecker:
    type = "base"

    def __init__(self, warning_on_screen=False):
        self.warning_on_screen = warning_on_screen

    def check(self, env):
        result = self._check(env)
        if self.warning_on_screen:
            self.show_warning(result)
        return result

    def _check(self, env):
        return {"success": True, "warning_text": None}

    def reset(self):
        pass

    def show_warning(self, result):
        if result.get("warning_text"):
            return result.get("warning_text")
        else:
            return None

    def get_metrics(self, result):
        if result.get("metrics"):
            return result.get("metrics")
        else:
            return {}
