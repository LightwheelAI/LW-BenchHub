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

from .fixture import Fixture
from .fixture_types import FixtureType


class FloorLayout(Fixture):
    fixture_types = [FixtureType.FLOOR_LAYOUT]

    def get_reset_regions(self, env=None, reset_region_names=None, z_range=(0.0, 1.50)):
        # floor obj z range is start from 0.0
        return super().get_reset_regions(env, reset_region_names, z_range)
