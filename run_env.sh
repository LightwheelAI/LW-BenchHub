#!/bin/bash

task_config=lift-cube
python ./lwlab/scripts/env_server.py \
    --task_config="$task_config" \
    --headless
