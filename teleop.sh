#!/bin/bash

task_config=teleop_base

python ./lwlab/scripts/teleop/teleop_main.py \
    --task_config="$task_config"
