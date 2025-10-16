from functools import cached_property
from lwlab.core.models.fixtures.fixture import Fixture
import os
from lwlab.utils.usd_utils import OpenUsd as usd
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from lwlab.core.models.fixtures.fixture_types import FixtureType


class FloorLayout(Fixture):
    fixture_types = [FixtureType.FLOOR_LAYOUT]

    def get_reset_regions(self, env=None, reset_region_names=None, z_range=(0.0, 1.50)):
        # floor obj z range is start from 0.0
        return super().get_reset_regions(env, reset_region_names, z_range)
