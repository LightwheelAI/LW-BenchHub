#!/bin/bash

task_config=lift-cube
python ./lw_benchhub/scripts/env_server.py \
    --task_config="$task_config" \
    --headless
