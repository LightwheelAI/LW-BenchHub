from .fixture import Fixture
from .fixture_types import FixtureType


class DishTowel(Fixture):
    fixture_types = [FixtureType.DISHTOWEL]

    def __init__(self, name, prim, num_envs, **kwargs):
        super().__init__(name, prim, num_envs, **kwargs)
