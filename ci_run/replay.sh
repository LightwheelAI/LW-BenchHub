#!/bin/bash

DATASET_FILE=$1
REPLAY_MODE=$2
export LW_API_ENDPOINT="https://api-dev.lightwheel.net"
HDF5_FILENAME=$(basename "${DATASET_FILE}")
DATASET_DIR=/output

cp "${DATASET_FILE}" "${DATASET_DIR}"
DATASET_FILE="${DATASET_DIR}/${HDF5_FILENAME}"

OUTPUT_JSON="${DATASET_DIR}/result.json"

python3 /workspace/lwlab/lwlab/scripts/teleop/replay_action_demo.py \
    --dataset_file=${DATASET_FILE} \
    --num_envs=1 \
    --width=480 \
    --height=480 \
    --replay_mode=${REPLAY_MODE} \
    --enable_cameras \
    --headless \
    --without_image \
    --device=cpu 2>&1

PYTHON_EXIT_CODE=$?

REPLAY_JSON_PATH="${DATASET_DIR}/isaac_replay_action_${REPLAY_MODE}.json"

if [ ${PYTHON_EXIT_CODE} -eq 0 ] && [ -f "${REPLAY_JSON_PATH}" ]; then
    python3 -c "
import json
with open('${REPLAY_JSON_PATH}', 'r') as f:
    data = json.load(f)
success = False
for k, v in data.items():
    if isinstance(v, dict) and 'success' in v:
        success = v['success']
        break
with open('${OUTPUT_JSON}', 'w') as out_f:
    json.dump({'success': success, 'desc':'${HDF5_FILENAME}--${REPLAY_MODE}', 'error': None}, out_f, indent=4, ensure_ascii=False)
"
else
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump({
        'success': False,
        'desc': '${HDF5_FILENAME}--${REPLAY_MODE}',
        'error': '${PYTHON_EXIT_CODE}'
    }, f, indent=4, ensure_ascii=False)
"
    exit 1
fi
