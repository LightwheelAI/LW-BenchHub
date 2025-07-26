<h1 align="center">
    LwLab
</h1>

For detail documentation, please refer to https://lwlab-docs.lightwheel.net/

# Installation

For detailed installation instructions, please refer to the official documentation: https://lwlab-docs.lightwheel.net/usage/install.html

# Project Structure and Development Guidelines

## configs
This directory contains configuration files related to data collection, as well as the training and evaluation of reinforcement learning tasks.

## lwlab
This module provides core functionalities, including simulation scene generation, asset logic control, robot control, entry-point scripts, and utility functions. The `core` submodule offers standardized interfaces for common tasks, such as success criteria evaluation (`check_success_api`) and reward function design (`reward_design_api`).

## policy
This directory focuses on the implementation of policy algorithms, covering both imitation learning (IL) and reinforcement learning (RL) strategies. The codebase is designed for modular experimentation and systematic benchmarking of various policy architectures.

## third_party
This folder contains third-party environment dependencies, such as IsaacLab, Robocasa, and Robosuite. To ensure reproducibility and maintainability, these environments are preserved in their original form as much as possible.

## tasks
This directory defines task specifications. Each task, such as `OpenOven`, includes its own success criteria, task-related asset control and item placement, as well as a detailed task description.

## Launch Scripts

- `teleop.sh`: Launches the teleoperation mode, allowing real-time robot control via VR controllers or other input devices. Useful for data collection, demonstration, or manual intervention scenarios.
- `train.sh`: Starts the training process for reinforcement learning or imitation learning. This script automatically loads configuration files, initializes environments and policies, and begins the training loop.
- `eval.sh`: Evaluates trained policies or models. Supports performance testing across different tasks and environments, and outputs evaluation metrics.
- `install.sh`: Installs all required dependencies for the project, including Python packages, third-party libraries, and some system dependencies, ensuring a consistent development and runtime environment.