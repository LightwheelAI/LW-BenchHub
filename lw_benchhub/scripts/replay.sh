#!/bin/bash

DATASET_FILE=$1
REPLAY_MODE=$2
HDF5_FILENAME=$(basename "${DATASET_FILE}")
DATASET_DIR=/output

cp "${DATASET_FILE}" "${DATASET_DIR}"
DATASET_FILE="${DATASET_DIR}/${HDF5_FILENAME}"

OUTPUT_JSON="${DATASET_DIR}/result.json"

python3 /workspace/lw_benchhub/lw_benchhub/scripts/teleop/replay_action_demo.py \
    --dataset_file=${DATASET_FILE} \
    --num_envs=1 \
    --width=640 \
    --height=360 \
    --replay_mode=${REPLAY_MODE} \
    --enable_cameras \
    --headless \
    --device=cpu \
    --without_image 2>&1

PYTHON_EXIT_CODE=$?

REPLAY_JSON_PATH="${DATASET_DIR}/isaac_replay_action_${REPLAY_MODE}.json"

if [ ${PYTHON_EXIT_CODE} -eq 0 ] && [ -f "${REPLAY_JSON_PATH}" ]; then
    python3 -c "
import json, os
with open('${REPLAY_JSON_PATH}', 'r') as f:
    data = json.load(f)
success = False
for k, v in data.items():
    if isinstance(v, dict) and 'success' in v:
        success = v['success']
        break

result = {
    'success': success,
    'desc': '${HDF5_FILENAME}--${REPLAY_MODE}',
    'error': None,
    'checkers': {}
}

metrics_path = os.path.join('${DATASET_DIR}', 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r') as mf:
            metrics = json.load(mf)
        for checker_name, checker_data in metrics.items():
            if isinstance(checker_data, dict) and 'success' in checker_data:
                result['checkers'][checker_name] = bool(checker_data['success'])
    except Exception as e:
        pass

with open('${OUTPUT_JSON}', 'w') as out_f:
    json.dump(result, out_f, indent=4, ensure_ascii=False)
"
else
    python3 -c "
import json, os
result = {
    'success': False,
    'desc': '${HDF5_FILENAME}--${REPLAY_MODE}',
    'error': '${PYTHON_EXIT_CODE}',
    'checkers': {}
}

metrics_path = os.path.join('${DATASET_DIR}', 'metrics.json')
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, 'r') as mf:
            metrics = json.load(mf)
        for checker_name, checker_data in metrics.items():
            if isinstance(checker_data, dict) and 'success' in checker_data:
                result['checkers'][checker_name] = bool(checker_data['success'])
    except Exception as e:
        pass

with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
"
    exit 1
fi
