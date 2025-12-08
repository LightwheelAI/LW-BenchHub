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

from lw_benchhub.utils.env import get_default_logger


class SineWaveActionGenerator:
    """
    Generates complex, multi-stage sinusoidal actions for debugging robot arm control.
    """

    def __init__(self, is_enabled: bool = False, freq: float = 0.5, amplitude: float = 0.15):
        """
        Initializes the SineWaveActionGenerator.

        Args:
            is_enabled: Flag to enable or disable the sine wave generation.
            freq: Frequency of the sine wave in Hz.
            amplitude: Amplitude of the sine wave in meters.
        """
        self.is_enabled = is_enabled
        self.freq = freq
        self.amplitude = amplitude

        # Internal state variables
        self._start_time = None
        self._initial_left_pos = None
        self._initial_right_pos = None

    def reset(self):
        """Resets the internal state of the generator."""
        self._start_time = None
        self._initial_left_pos = None
        self._initial_right_pos = None

    def generate(self, action: dict, device) -> dict:
        """
        If enabled, modifies the input action dictionary with sine wave motions.

        Args:
            action: The original action dictionary from the teleop device.
            device: The simulation device object, used to get simulation time.

        Returns:
            The modified action dictionary.
        """
        if not self.is_enabled:
            return action

        if self._start_time is None:
            self._start_time = device.env.sim._current_time
            self._initial_left_pos = action["left_arm_abs"][:3].clone()
            self._initial_right_pos = action["right_arm_abs"][:3].clone()
            get_default_logger().info("Sine wave generator started.")

        elapsed_time = device.env.sim._current_time - self._start_time

        # Define motion phases, each lasting 15 seconds, for a 45-second cycle
        phase_duration = 15.0
        total_cycle_duration = 3 * phase_duration
        time_in_cycle = elapsed_time % total_cycle_duration

        new_left_pos = self._initial_left_pos.clone()
        new_right_pos = self._initial_right_pos.clone()

        # Phase 1 (0-15s): Synchronized forward/backward motion (X-axis)
        if time_in_cycle < phase_duration:
            phase_time = time_in_cycle
            sine_input = torch.tensor(2 * torch.pi * self.freq * phase_time, device=device.env.device)
            offset = self.amplitude * torch.sin(sine_input)
            new_left_pos[0] = self._initial_left_pos[0] + offset
            new_right_pos[0] = self._initial_right_pos[0] + offset

        # Phase 2 (15-30s): Asymmetrical circles (Left: XY plane, Right: YZ plane)
        elif time_in_cycle < 2 * phase_duration:
            phase_time = time_in_cycle - phase_duration
            angle = 2 * torch.pi * self.freq * phase_time
            radius = self.amplitude
            angle_tensor = torch.tensor(angle, device=device.env.device)
            # Left arm: Horizontal circle (XY Plane)
            new_left_pos[0] = self._initial_left_pos[0] + radius * torch.cos(angle_tensor)
            new_left_pos[1] = self._initial_left_pos[1] + radius * torch.sin(angle_tensor)
            # Right arm: Vertical circle (YZ Plane)
            new_right_pos[1] = self._initial_right_pos[1] + radius * torch.sin(angle_tensor)
            new_right_pos[2] = self._initial_right_pos[2] + radius * torch.cos(angle_tensor)

        # Phase 3 (30-45s): Orthogonal linear motion (Left: Y-axis, Right: Z-axis)
        else:
            phase_time = time_in_cycle - (2 * phase_duration)
            sine_input = torch.tensor(2 * torch.pi * self.freq * phase_time, device=device.env.device)
            offset = self.amplitude * torch.sin(sine_input)
            new_left_pos[1] = self._initial_left_pos[1] + offset
            new_right_pos[2] = self._initial_right_pos[2] + offset

        # Overwrite the original action with the newly generated positions
        action["left_arm_abs"][:3] = new_left_pos
        action["right_arm_abs"][:3] = new_right_pos

        return action
