# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ArgsConfig:
    """Args Config for running the data collection loop."""

    def update(self, config_dict: dict, strict: bool = False, skip_keys: list[str] = []):
        for k, v in config_dict.items():
            if k in skip_keys:
                continue
            if strict and not hasattr(self, k):
                raise ValueError(f"Config {k} not found in {self.__class__.__name__}")
            if not strict and not hasattr(self, k):
                continue
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, config_dict: dict, strict: bool = False, skip_keys: list[str] = []):
        instance = cls()
        instance.update(config_dict=config_dict, strict=strict, skip_keys=skip_keys)
        return instance

    def to_dict(self):
        return asdict(self)


def override_wbc_config(
    wbc_config: dict, config: "BaseConfig", missed_keys_only: bool = False
) -> dict:
    """Override WBC YAML values with dataclass values.

    Args:
        wbc_config: The loaded WBC YAML configuration dictionary
        config: The BaseConfig dataclass instance with override values
        missed_keys_only: If True, only add keys that don't exist in wbc_config.
                         If False, validate all keys exist and override all.

    Returns:
        Updated wbc_config dictionary with overridden values

    Raises:
        KeyError: If any required keys are missing from the WBC YAML configuration
                  (only when missed_keys_only=False)
    """
    # Override yaml values with dataclass values
    key_to_value = {
        "ENV_TYPE": config.env_type,
        "VERSION": config.wbc_version,
        "SIMULATOR": config.simulator,
        "model_path": config.wbc_model_path,
        "verbose": config.verbose,
        "upper_body_max_joint_speed": config.upper_body_joint_speed,
    }

    if missed_keys_only:
        # Only add keys that don't exist in wbc_config
        for key in key_to_value:
            if key not in wbc_config:
                wbc_config[key] = key_to_value[key]
    else:
        # Set all keys (overwrite existing)
        for key in key_to_value:
            wbc_config[key] = key_to_value[key]

    return wbc_config


@dataclass
class BaseConfig(ArgsConfig):
    """Base config inherited by all G1 control loops"""

    # WBC Configuration
    wbc_version: Literal["homie", "homie_v2"] = "homie"
    """Version of the whole body controller."""

    wbc_model_path: str = "models/t1.onnx"
    """Path to WBC model file"""
    """homie model path: models/t1.onnx"""

    wbc_policy_class: str = "G1DecoupledWholeBodyPolicy"
    """Whole body policy class."""

    # System Configuration

    env_type: str = "sim"

    simulator: str = "isaaclab"
    """Simulator to use."""

    verbose: bool = True
    """Whether to print verbose output."""

    env_name: str = "default"
    """Environment name."""

    # Robot Configuration
    enable_waist: bool = False
    """Whether to include waist joints in IK."""

    upper_body_joint_speed: float = 5
    """Upper body joint speed."""

    # Policy settings
    enable_upper_body_operation: bool = True
    """Enable upper body operation"""

    upper_body_operation_mode: Literal["teleop", "inference"] = "teleop"
    """Upper body operation mode"""

    view_camera: bool = True
    """Enable camera viewer"""

    def load_wbc_yaml(self) -> dict:
        """Load and merge wbc yaml with dataclass overrides"""
        # Get the base path to groot and convert to Path object
        groot_path = Path(os.path.dirname(__file__)).parent

        if self.wbc_version == "homie":
            config_path = str(groot_path / "config/g1_29dof_homie.yaml")
        elif self.wbc_version == "homie_v2":
            config_path = str(groot_path / "config/g1_29dof_homie_v2.yaml")
        else:
            raise ValueError(
                f"Invalid wbc_version: {self.wbc_version}, please use one of: "
                f"homie, homie_v2"
            )

        with open(config_path) as file:
            wbc_config = yaml.load(file, Loader=yaml.FullLoader)

        # Override yaml values with dataclass values
        wbc_config = override_wbc_config(wbc_config, self)

        return wbc_config
