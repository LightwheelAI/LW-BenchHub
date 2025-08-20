#!/bin/bash
task_config=lerobot_liftobj_state
env_gpu=0
policy_gpu=0

if [ "$env_gpu" -eq "$policy_gpu" ]; then
    export CUDA_VISIBLE_DEVICES=${env_gpu}
    export ENV_GPU=0
    export POLICY_GPU=0
else
    export CUDA_VISIBLE_DEVICES="${env_gpu},${policy_gpu}"
    export ENV_GPU=0
    export POLICY_GPU=1
fi

python ./lwlab/scripts/rl/train.py \
    --task_config="$task_config" \
    --headless \
    # --enable_cameras
