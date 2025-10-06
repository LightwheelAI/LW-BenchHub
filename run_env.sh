#!/bin/bash

task_config=lift-cube
export LW_API_ENDPOINT="https://api-dev.lightwheel.net"
python ./lwlab/scripts/env_server.py \
    --task_config="$task_config" \
    --headless
