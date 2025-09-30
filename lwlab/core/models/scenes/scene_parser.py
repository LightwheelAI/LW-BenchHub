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

# base class for kitchen arena

import numpy as np
from lwlab.core.models.fixtures.fixture import FIXTURES
from lwlab.utils.usd_utils import OpenUsd as usd


def parse_fixtures(stage, num_envs, seed, device):
    """
    Parses fixtures from the given stage

    Args:
        stage (Usd.Stage): stage to parse fixtures from

    Returns:
        list: list of fixture infos{name: fixture object}
    """
    fixtures = {}
    root_prim = stage.GetPseudoRoot().GetChildren()[0]
    xform_infos = usd.get_child_xform_infos(root_prim)
    for info in xform_infos:
        size_attr = info["prim"].GetAttribute("size").Get()
        if size_attr is None or np.fromstring(size_attr, sep=',').size == 0:
            continue
        fixture_type = info["type"] if info["type"] in FIXTURES else "Accessory"
        fixtures[info["name"]] = FIXTURES[fixture_type](info["name"], info["prim"], num_envs, seed=seed, device=device)

    return fixtures
