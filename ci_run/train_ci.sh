#!/bin/bash

task_config=$1

if [ -n "$CONDA_DIR" ] && [ -n "$ENV_NAME" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

DATASET_DIR=/output
OUTPUT_JSON="${DATASET_DIR}/result.json"
pt_found=0
success=True
success_rate=0
if [ "$task_config" == "ci_lerobot_liftobj_state" ] || [ "$task_config" == "ci_g1_liftobj_state" ] || [ "$task_config" == "ci_daily_lerobot_liftobj_state" ]; then
    rm -rf /workspace/lw_benchhub/policy/maniskill_ppo/logs/*
    python3 /workspace/lw_benchhub/policy/maniskill_ppo/train.py \
        --task_config="$task_config" \
        --headless 2>&1

    pt_file=$(find /workspace/lw_benchhub/policy/maniskill_ppo/logs/ -type f -path "*/*.pt" | head -n 1)
    if [ "$task_config" == "ci_daily_lerobot_liftobj_state" ]; then
        json_file=$(find /workspace/lw_benchhub/policy/maniskill_ppo/logs/ -type f -path "*/result.json")
        if [ -n "$json_file" ]; then
            success_rate=$(python3 -c "import json; print(json.load(open('$json_file')).get('success_rate', ''))")
            success=$(python3 -c "import json; print(json.load(open('$json_file')).get('ci_success', ''))")
        else
            success=False
        fi
    fi
    
elif [ "$task_config" == "ci_open_draw" ]; then
    rm -rf /workspace/lw_benchhub/policy/skrl/logs/*
    python3 /workspace/lw_benchhub/lw_benchhub/scripts/rl/train.py \
        --task_config="$task_config" \
        --headless 2>&1

    pt_file=$(find /workspace/lw_benchhub/policy/skrl/logs/ -type f -path "*/checkpoints/*.pt" | head -n 1)
fi

PYTHON_EXIT_CODE=$?

if [ -n "$pt_file" ]; then
    pt_found=1
fi

if [ ${PYTHON_EXIT_CODE} -eq 0 ] && [ $pt_found -eq 1 ]; then
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as out_f:
    json.dump({'success': ${success}, 'desc':'RL training ${task_config}, only daily ci check success_rate: ${success_rate}', 'error': None}, out_f, indent=4, ensure_ascii=False)
"
else
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump({
        'success': False,
        'desc': 'RL training failed',
        'error': '${PYTHON_EXIT_CODE} or not success save pt file'
    }, f, indent=4, ensure_ascii=False)
"
    exit 1
fi
