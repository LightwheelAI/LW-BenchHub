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

import numpy as np

from .fixture import Fixture
from .fixture_types import FixtureType


class Accessory(Fixture):
    """
    Base class for all accessories/Miscellaneous objects

    Args:
        name (str): name of the object
        prim: USD prim for the accessory
        pos (list): position of the object
    """

    def __init__(self, name, prim, num_envs, pos=None, *args, **kwargs):
        super().__init__(
            name=name,
            prim=prim,
            num_envs=num_envs,
            pos=pos,
            *args,
            **kwargs
        )


class WallAccessory(Fixture):
    """
    Class for wall accessories. These are objects that are attached to walls,
    such as outlets, clocks, paintings, etc.

    Args:
        name (str): name of the object
        prim: USD prim for the accessory
        pos (list): position of the object
        attach_to (Wall): The wall to attach the object to
        protrusion (float): How much to protrude out of the wall when placing the object
    """

    def __init__(
        self,
        name,
        prim,
        num_envs,
        pos=None,
        attach_to=None,
        protrusion=0.02,
        *args,
        **kwargs
    ):
        super().__init__(
            name=name,
            prim=prim,
            num_envs=num_envs,
            pos=pos,
            *args,
            **kwargs
        )

        # The wall to attach accessory to
        self.wall = attach_to
        # How much to protrude out of wall
        if self.wall is not None:
            if protrusion is not None:
                self.protrusion = protrusion
            else:
                # Calculate protrusion based on accessory dimensions
                self.protrusion = (
                    self.height / 2
                    if hasattr(self.wall, 'wall_side') and self.wall.wall_side == "floor"
                    else self.depth / 2 if hasattr(self, 'depth') else 0.02
                )
                if hasattr(self.wall, 'size') and self.wall.size is not None:
                    self.protrusion += self.wall.size[2] if len(self.wall.size) > 2 else 0
        else:
            self.protrusion = None

        # Place the accessory after initialization
        if pos is not None:
            self._place_accessory()

    def _place_accessory(self):
        """
        Place the accessory on the wall using USD transformations
        """
        if self.wall is None:
            # Absolute position was specified
            return

        if not hasattr(self, 'pos') or self.pos is None:
            return

        x, y, z = self.pos

        # Update position and rotation based on the wall it attaches to
        if hasattr(self.wall, 'wall_side'):
            if self.wall.wall_side == "back":
                y = self.wall.pos[1] - self.protrusion
            elif self.wall.wall_side == "front":
                self.set_euler([0, 0, self.rot + np.pi])
                y = self.wall.pos[1] + self.protrusion
            elif self.wall.wall_side == "right":
                x = self.wall.pos[0] - self.protrusion
                self.set_euler([0, 0, self.rot - np.pi / 2])
            elif self.wall.wall_side == "left":
                x = self.wall.pos[0] + self.protrusion
                self.set_euler([0, 0, self.rot + np.pi / 2])
            elif self.wall.wall_side == "floor":
                z = self.wall.pos[2] + self.protrusion
            else:
                raise ValueError(f"Unknown wall side: {self.wall.wall_side}")


class Stool(WallAccessory):
    """
    Stool accessory that can be placed on walls or floors
    """
    fixture_types = [FixtureType.STOOL]

    def __init__(
        self,
        name,
        prim,
        num_envs,
        pos=None,
        attach_to=None,
        protrusion=None,
        z_rot=None,
        *args,
        **kwargs
    ):
        super().__init__(
            name=name,
            prim=prim,
            num_envs=num_envs,
            pos=pos,
            attach_to=attach_to,
            protrusion=protrusion,
            *args,
            **kwargs
        )

        if z_rot is not None:
            self.set_euler([0, 0, z_rot])

    @property
    def nat_lang(self):
        return "stool"
