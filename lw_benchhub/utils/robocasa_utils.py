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

def convert_fixture_to_name(d) -> dict:
    if not isinstance(d, dict):
        # Check if it is a fixture type
        if hasattr(d, "__class__") and "lw_benchhub.core.models.fixtures" in d.__class__.__module__:
            return d.name
        return d
    result = {}
    for k, v in d.items():
        result[k] = convert_fixture_to_name(v)
    return result
