#!/bin/bash
task_config=lerobot_liftobj_state

python ./lw_benchhub/scripts/rl/train.py \
    --task_config="$task_config" \
    --headless \
    # --enable_cameras
