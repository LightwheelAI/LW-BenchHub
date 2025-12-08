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


import os

import numpy as np

from lw_benchhub.core.models.fixtures.fixture import FIXTURES
from lw_benchhub.utils.usd_utils import OpenUsd as usd


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
        if info['type'] == "Wall_obj" or info['type'] == "Floor_obj":
            continue
        size_attr = info["prim"].GetAttribute("size").Get()
        if size_attr is None or np.fromstring(size_attr, sep=',').size == 0:
            continue
        fixture_type = info["type"] if info["type"] in FIXTURES else "Accessory"
        fixtures[info["name"]] = FIXTURES[fixture_type](info["name"], info["prim"], num_envs, seed=seed, device=device)

    return fixtures


def get_fixture_cfgs(env):
    """
    Returns config data for all fixtures in the arena

    Returns:
        list: list of fixture configurations
    """
    fixture_cfgs = []
    for (name, fxtr) in env.fixtures.items():
        cfg = {}
        cfg["name"] = name
        cfg["model"] = fxtr
        cfg["type"] = "fixture"
        if hasattr(fxtr, "_placement"):
            cfg["placement"] = fxtr._placement

        fixture_cfgs.append(cfg)
    return fixture_cfgs


def register_fixture_from_obj(obj, prim, fixtures_ref, num_envs, pos, rot):
    """
    Register a fixture if the given object corresponds to an articulated fixture.

    Args:
        obj: The object model instance (from self.objects)
        prim: The prim of placed fixture
        fixtures_ref (dict): the fixtures dictionary to add into
        num_envs (int): number of environments (for fixture init)

    Returns:
        bool: True if fixture successfully registered, False otherwise
    """
    fixture_path = os.path.splitext(os.path.basename(obj["info"]["obj_path"]))[0]
    fixture_name = ""
    for c in fixture_path:
        if c == "_":
            break
        elif c.isalpha():
            fixture_name += c

    fixture_type = fixture_name if fixture_name in FIXTURES else "Accessory"

    if obj["name"] in fixtures_ref:
        return False  # already exists

    try:
        fixtures_ref[obj["name"]] = FIXTURES[fixture_type](
            name=obj["name"],
            prim=prim,
            num_envs=num_envs,
            pos=pos,
            rot=rot
        )
        print(f"[Fixture Placed] {obj['name']} ({fixture_type})")
        return True
    except Exception as e:
        raise RuntimeError(f"Fixture registration failed for {obj['name']}: {e}")
