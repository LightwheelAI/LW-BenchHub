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

import threading
import time

import numpy as np


class Runner_online_real:
    """Runner for real robot"""

    def __init__(self, robot_cfg, robotInterface):
        """Initialize the runner

        Args:
            robot_cfg: robot configuration
            robotInterface: robot interface
        """
        self.robot_cfg = robot_cfg
        self.robotInterface = robotInterface
        self.lock = threading.Lock()

        # State variables
        self.low_state = None
        self.ang_vel = np.zeros(3)  # Angular velocity
        self.prop_updated = False

        # Command variables
        self.cmd = {}
        self.cmd_ready = False

    def refresh_prop(self):
        """Update robot state

        Returns:
            Whether the state was updated successfully
        """
        if not self.robotInterface.is_alive():
            print("Robot interface disconnected")
            return False

        try:
            self.lock.acquire()
            # Update robot state
            self.low_state = self.robotInterface.receive_low_state()

            # Get angular velocity from IMU
            if hasattr(self.low_state, "imu") and hasattr(self.low_state.imu, "gyroscope"):
                self.ang_vel = self.low_state.imu.gyroscope

                # If quaternion data exists, update to leg_position_action
                if hasattr(self.robot_cfg, "leg_action_term") and hasattr(self.low_state.imu, "orientation"):
                    quaternion = self.low_state.imu.orientation
                    gyroscope = self.low_state.imu.gyroscope

                    # Update IMU data in LegPositionAction
                    self.robot_cfg.leg_action_term.update_imu_data(quaternion, gyroscope)

            self.prop_updated = True
            return True
        except Exception as e:
            print(f"Failed to refresh robot state: {e}")
            return False
        finally:
            self.lock.release()

    def update_cmd(self, cmd):
        """Update control command

        Args:
            cmd: control command
        """
        try:
            self.lock.acquire()
            self.cmd = cmd
            self.cmd_ready = True
        finally:
            self.lock.release()

    def send_cmd(self):
        """Send control command

        Returns:
            Whether the command was sent successfully
        """
        if not self.robotInterface.is_alive():
            print("Robot interface disconnected")
            return False

        try:
            self.lock.acquire()
            if not self.cmd_ready:
                return False

            # Send command
            self.robotInterface.send_command(self.cmd)
            self.cmd_ready = False
            return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False
        finally:
            self.lock.release()

    def run_step(self):
        """Run one step

        Returns:
            Whether the step ran successfully
        """
        # Update robot state
        if not self.refresh_prop():
            return False

        # Send control command
        if not self.send_cmd():
            return False

        return True

    def run(self, freq=100):
        """Run continuously

        Args:
            freq: run frequency (Hz)
        """
        period = 1.0 / freq

        while self.robotInterface.is_alive():
            start_time = time.time()

            # Run one step
            if not self.run_step():
                print("Run step failed")
                break

            # Control frequency
            elapsed = time.time() - start_time
            if elapsed < period:
                time.sleep(period - elapsed)
            else:
                print(f"Warning: run period overtime ({elapsed:.4f}s > {period:.4f}s)")

    def close(self):
        """Close the runner"""
        if self.robotInterface.is_alive():
            self.robotInterface.close()
