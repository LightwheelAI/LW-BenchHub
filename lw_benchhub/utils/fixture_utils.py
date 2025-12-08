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

from lw_benchhub.core.models.fixtures import FixtureType


def fixture_is_type(fixture, fixture_type):
    """
    Check if a fixture is of a certain type

    Args:
        fixture (Fixture): The fixture to check

        fixture_type (FixtureType): The type to check against
    """
    if fixture is None:
        return False
    elif isinstance(fixture, FixtureType):
        return fixture == fixture_type
    else:
        return fixture._is_fixture_type(fixture_type)


def is_fxtr_valid(env, fxtr, size):
    """
    checks if counter is valid for object placement by making sure it is large enough

    Args:
        fxtr (Fixture): fixture to check
        size (tuple): minimum size (x,y) that the counter region must be to be valid

    Returns:
        bool: True if fixture is valid, False otherwise
    """
    for region in fxtr.get_reset_regions(env).values():
        if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
            return True
    return False
