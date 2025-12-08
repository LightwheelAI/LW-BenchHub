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
# See the License for the specifi

import json
import os
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from lw_benchhub.utils.lerobot_common.cameras.camera import Camera
from lw_benchhub.utils.lerobot_common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lw_benchhub.utils.lerobot_common.cameras.utils import make_cameras_from_configs
from lw_benchhub.utils.lerobot_common.common import flatten_state_dict, to_cpu_tensor, to_tensor
from lw_benchhub.utils.lerobot_common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lw_benchhub.utils.lerobot_common.motors import FeetechMotorsBus, Motor, MotorCalibration, MotorNormMode, OperatingMode


class SO101Follower():
    """A SO101 Leader device for SE(3) control.
    """

    def __init__(self, port: str = '/dev/ttyACM0', recalibrate: bool = False, calibration_file_name: str = 'so101_follower.json',
                 camera_index=-1, use_degrees=False):
        super().__init__()
        self.port = port
        self.cameras = {}
        if camera_index > -1:
            self.cameras_cfg = {"global_camera": OpenCVCameraConfig(index_or_path=camera_index, fps=30, width=640, height=480)}
            self._sensor_names = list(self.cameras_cfg.keys())
            self.cameras = make_cameras_from_configs(self.cameras_cfg)
        norm_mode_body = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
        # calibration
        self.calibration_path = os.path.join(os.path.dirname(__file__), ".cache", calibration_file_name)
        if not os.path.exists(self.calibration_path) or recalibrate:
            self.calibrate(norm_mode_body)
        calibration = self._load_calibration()

        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )
        self._motor_keys = list(self._bus.motors.keys())

        # connect
        self.connect()

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

    def __str__(self) -> str:
        """Returns: A string containing the information of so101 leader."""
        msg = "SO101-Follower device for SE(3) control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove SO101-Follower to control SO101-Follower\n"
        msg += "\tIf SO101-Follower can't synchronize with SO101-Follower, please add --recalibrate and rerun to recalibrate SO101-Follower.\n"
        return msg

    def _get_raw_data(self):
        return self._bus.sync_read("Present_Position")

    def reset(self, qpos):
        qpos = to_cpu_tensor(qpos)
        freq = 30
        target_pos = self.qpos
        max_rad_per_step = 0.025
        for _ in range(int(20 * freq)):
            start_loop_t = time.perf_counter()
            delta_step = (qpos - target_pos).clip(
                min=-max_rad_per_step, max=max_rad_per_step
            )
            if np.linalg.norm(delta_step) <= 1e-4:
                break
            target_pos += delta_step

            self.set_target_qpos(target_pos)
            dt_s = time.perf_counter() - start_loop_t
            time.sleep(1 / freq - dt_s)

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    @property
    def started(self) -> bool:
        return self._started

    @property
    def reset_state(self) -> bool:
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state

    @property
    def motor_limits(self) -> Dict[str, Tuple[float, float]]:
        return self._motor_limits

    @property
    def is_connected(self) -> bool:
        return self._bus.is_connected

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("SO101-Follower is not connected.")
        self._bus.disconnect()
        print("SO101-Follower disconnected.")

    def connect(self):
        if self.is_connected:
            raise DeviceAlreadyConnectedError("SO101-Follower is already connected.")
        self._bus.connect()
        self.configure()
        print("SO101-Follower connected.")

        for cam in self.cameras.values():
            if cam.is_connected:
                continue
            cam.connect()
        print("Cameras connected.")

    def configure(self) -> None:
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self._bus.write("P_Coefficient", motor, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self._bus.write("I_Coefficient", motor, 0)
            self._bus.write("D_Coefficient", motor, 32)

        self._bus.write("P_Coefficient", "shoulder_lift", 50)
        self._bus.write("P_Coefficient", "elbow_flex", 50)
        self._bus.write("P_Coefficient", "gripper", 50)

    @property
    def qpos(self):
        return self.get_qpos()

    def get_qpos(self):
        # NOTE (stao): the slowest part of inference is reading the qpos from the robot. Each time it takes about 5-6 milliseconds, meaning control frequency is capped at 200Hz.
        # and if you factor in other operations like policy inference etc. the max control frequency is typically more like 30-60 Hz.
        # Moreover on the rare occassions reading qpos can take 40 milliseconds which causes the control step to fall behind the desired control frequency.
        qpos_deg = self._bus.sync_read("Present_Position")

        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        qpos_deg = flatten_state_dict(qpos_deg)
        qpos = torch.deg2rad(torch.tensor(qpos_deg)).unsqueeze(0)
        return qpos

    def calibrate(self, norm_mode_body):
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )

        self.connect()

        print("\n Running calibration of SO101-Follower")
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move SO101-Follower to the middle of its range of motion and press ENTER...")
        homing_offset = self._bus.set_half_turn_homings()
        print("Move all joints sequentially through their entire ranges of motion.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = self._bus.record_ranges_of_motion()

        calibration = {}
        for motor, m in self._bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_path}")

        self.disconnect()

    def _load_calibration(self) -> Dict[str, MotorCalibration]:
        with open(self.calibration_path, "r") as f:
            json_data = json.load(f)
        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    def set_target_qpos(self, qpos):
        qpos = to_cpu_tensor(qpos).flatten()
        qpos = torch.rad2deg(qpos)
        qpos = {f"{self._motor_keys[i]}.pos": qpos[i] for i in range(len(qpos))}
        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        self.send_action(qpos)

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Send goal position to the arm
        self._bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _save_calibration(self, calibration: Dict[str, MotorCalibration]):
        save_calibration = {k: {
            "id": v.id,
            "drive_mode": v.drive_mode,
            "homing_offset": v.homing_offset,
            "range_min": v.range_min,
            "range_max": v.range_max,
        } for k, v in calibration.items()}
        if not os.path.exists(os.path.dirname(self.calibration_path)):
            os.makedirs(os.path.dirname(self.calibration_path))
        with open(self.calibration_path, 'w') as f:
            json.dump(save_calibration, f, indent=4)

    def get_sensor_images(self):
        # used by render_sensors
        return self._get_obs_sensor_data()

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        # note apply_texture_transforms is not used for real envs, data is expected to already be transformed to standard texture names, types, and shapes.
        self.capture_sensor_data(self._sensor_names)
        data = self.get_sensor_data(self._sensor_names)
        # observation data needs to be processed to be the same shape in simulation
        # default strategy is to do a center crop to the same shape as simulation and then resize image to the same shape as simulation
        data = self.preprocess_sensor_data(data)
        return data

    def get_sensor_data(self, sensor_names: Optional[List[str]] = None):
        if self._captured_sensor_data is None:
            raise RuntimeError(
                "No sensor data captured yet. Please call capture_sensor_data() first."
            )
        if sensor_names is None:
            return self._captured_sensor_data
        else:
            return {
                k: v for k, v in self._captured_sensor_data.items() if k in sensor_names
            }

    def capture_sensor_data(self, sensor_names: Optional[List[str]] = None):
        sensor_obs = dict()
        cameras: dict[str, Camera] = self.cameras
        if sensor_names is None:
            sensor_names = list(cameras.keys())
        for name in sensor_names:
            data = cameras[name].async_read()
            # until https://github.com/huggingface/lerobot/issues/860 is resolved we temporarily assume this is RGB data only otherwise need to write a few extra if statements to check
            # if isinstance(cameras[name], IntelRealSenseCamera):
            sensor_obs[name] = dict(rgb=(to_tensor(data)).unsqueeze(0))
        self._captured_sensor_data = sensor_obs

    def preprocess_sensor_data(
        self, sensor_data: Dict, sensor_names: Optional[List[str]] = None, target_h: int = 224, target_w: int = 224,
    ):
        import cv2

        if sensor_names is None:
            sensor_names = list(sensor_data.keys())
        for sensor_name in sensor_names:
            real_sensor_data = sensor_data[sensor_name]

            # crop to same aspect ratio
            for key in ["rgb", "depth"]:
                if key in real_sensor_data:
                    img = real_sensor_data[key][0].numpy()
                    xy_res = img.shape[:2]
                    crop_res = np.min(xy_res)
                    cutoff = (np.max(xy_res) - crop_res) // 2

                    if xy_res[0] == xy_res[1]:
                        pass
                    elif np.argmax(xy_res) == 0:
                        img = img[cutoff:-cutoff, :, :]
                    else:
                        img = img[:, cutoff:-cutoff, :]
                    real_sensor_data[key] = to_tensor(
                        cv2.resize(img, (target_w, target_h))
                    ).unsqueeze(0)

            sensor_data[sensor_name] = real_sensor_data
        return sensor_data
