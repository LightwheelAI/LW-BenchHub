#!/bin/bash

task_config=g1-controller
python ./lwlab/scripts/teleop/teleop_main.py \
    --task_config="$task_config"
