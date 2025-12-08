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

import torch

import lw_benchhub.utils.object_utils as OU
import lw_benchhub.utils.place_utils.env_utils as EnvUtils
from lw_benchhub.core.tasks.base import LwTaskBase


class NavigateKitchen(LwTaskBase):
    """
    Class encapsulating the atomic navigate kitchen tasks.
    Involves navigating the robot to a target fixture.
    """

    task_name: str = "NavigateKitchen"

    def _setup_kitchen_references(self, scene):
        """
        Setup the kitchen references for the navigate kitchen tasks.
        If not already chosen, selects a random start and destination fixture for the robot to navigate from/to.
        """
        super()._setup_kitchen_references(scene)
        if "src_fixture" in self.fixture_refs:
            self.src_fixture = self.fixture_refs["src_fixture"]
            self.target_fixture = self.fixture_refs["target_fixture"]
        else:
            # choose a valid random start and destination fixture
            fixtures = list(self.fixtures.values())
            valid_src_fixture_classes = [
                "CoffeeMachine",
                "Toaster",
                "Stove",
                "Stovetop",
                "SingleCabinet",
                "HingeCabinet",
                "OpenCabinet",
                "Drawer",
                "Microwave",
                "Sink",
                "Hood",
                "Oven",
                "Fridge",
                "Dishwasher",
            ]
            # keep choosing src fixture until it is a valid fixture
            while True:
                self.src_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.src_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break

            fxtr_classes = [type(fxtr).__name__ for fxtr in fixtures]
            valid_target_fxtr_classes = [
                cls
                for cls in fxtr_classes
                if fxtr_classes.count(cls) == 1
                and cls
                in [
                    "CoffeeMachine",
                    "Toaster",
                    "Stove",
                    "Stovetop",
                    "OpenCabinet",
                    "Microwave",
                    "Sink",
                    "Hood",
                    "Oven",
                    "Fridge",
                    "Dishwasher",
                ]
            ]

            while True:
                self.target_fixture = self.rng.choice(fixtures)
                fxtr_class = type(self.target_fixture).__name__
                if (
                    self.target_fixture == self.src_fixture
                    or fxtr_class not in valid_target_fxtr_classes
                ):
                    continue
                if fxtr_class == "Accessory":
                    continue
                # don't sample closeby fixtures
                if (
                    OU.fixture_pairwise_dist(self.src_fixture, self.target_fixture)
                    <= 1.0
                ):
                    continue
                break

            self.fixture_refs["src_fixture"] = self.src_fixture
            self.fixture_refs["target_fixture"] = self.target_fixture

        self.target_pos, self.target_ori = EnvUtils.compute_robot_base_placement_pose(
            scene, self, self.target_fixture
        )

        self.init_robot_base_ref = self.src_fixture

    def get_ep_meta(self):
        """
        Get the episode metadata for the navigate kitchen tasks.
        This includes the language description of the task.
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"Navigate to the {self.target_fixture.nat_lang}."
        return ep_meta

    def _check_success(self, env):
        """
        Check if the navigation task is successful.
        This is done by checking if the robot is within a certain distance of the target fixture and the robot is facing the fixture.

        Returns:
            bool: True if the task is successful, False otherwise.
        """
        base_id = env.scene.articulations['robot'].data.body_names.index(env.cfg.isaaclab_arena_env.embodiment.robot_base_link)
        base_pos = env.scene.articulations['robot'].data.body_com_pos_w[..., base_id, :]
        target_pos = torch.tensor(self.target_pos, device=base_pos.device, dtype=base_pos.dtype).unsqueeze(0).repeat(env.num_envs, 1)
        pos_check = torch.linalg.norm(target_pos[:, :2] - base_pos[:, :2], dim=-1) <= 0.20
        base_ori = env.scene.articulations['robot'].data.body_com_quat_w[..., base_id, :]
        base_ori = self.quat2euler_torch(base_ori)
        ori_check = torch.cos(torch.tensor(self.target_ori[2], device=env.device).repeat(env.num_envs) - base_ori[:, 2]) >= 0.98

        return pos_check & ori_check

    def quat2euler_torch(self, quat):
        """
        Convert a batch of quaternions to euler angles (roll, pitch, yaw).
        Input shape: (N, 4) or (4,), quaternion format [w, x, y, z].
        Output shape matches input: (N, 3) or (3,).
        """
        # Ensure input is a torch tensor
        if not isinstance(quat, torch.Tensor):
            quat = torch.tensor(quat, dtype=torch.float32)
        # Track if input is 1D for output shape alignment
        squeeze_flag = False
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
            squeeze_flag = True
        # Extract quaternion components
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        # Compute yaw (z-axis rotation)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        # Compute pitch (y-axis rotation)
        pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        # Compute roll (x-axis rotation)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        # Stack results to shape (N, 3)
        euler = torch.stack([roll, pitch, yaw], dim=-1)
        # If input was 1D, return 1D output
        if squeeze_flag:
            return euler.squeeze(0)
        return euler
