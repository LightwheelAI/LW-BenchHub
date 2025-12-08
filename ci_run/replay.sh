#!/bin/bash
DATASET_FILE=$1
REPLAY_MODE=$2
HDF5_FILENAME=$(basename "${DATASET_FILE}")
DATASET_DIR=/output

cp "${DATASET_FILE}" "${DATASET_DIR}"
DATASET_FILE="${DATASET_DIR}/${HDF5_FILENAME}"

OUTPUT_JSON="${DATASET_DIR}/result.json"
PYTHON_LOG="${DATASET_DIR}/python_run.log"
STACK_TRACE=""

if [ -n "$CONDA_DIR" ] && [ -n "$ENV_NAME" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

check_for_stack_trace() {
    local file_path="$1"
    STACK_TRACE=""
    
    if [ ! -f "$file_path" ]; then
        echo "error: log file not exist: $file_path"
        return 1
    fi
    
    if grep -q -E "Traceback.*most recent call last" "$file_path"; then
        #use awk to extract from Traceback to first empty line or line containing "ms]""
        # STACK_TRACE=$(awk '/Traceback/ {capture=1} capture {print; if (/^$/ || /ms\]/) {exit}}' "$file_path")
        STACK_TRACE=$(awk '
            /Traceback/ {capture=1}
            capture && !/^$/ && !/ms\]/ {
            print
            if (getline > 0 && (/^$/ || /ms\]/)) {
                exit
            }
        }' "$file_path")
    fi
}

echo "Start run replay_action_demo.py."
# run the Python script and capture both stdout and stderr
python3 /workspace/lw_benchhub/lw_benchhub/scripts/teleop/replay_action_demo.py \
    --dataset_file=${DATASET_FILE} \
    --num_envs=1 \
    --width=480 \
    --height=480 \
    --replay_mode=${REPLAY_MODE} \
    --headless \
    --without_image \
    --device=cpu 2>&1 | tee "${PYTHON_LOG}"
    # --enable_cameras \
PYTHON_EXIT_CODE=$?

REPLAY_JSON_PATH="${DATASET_DIR}/isaac_replay_action_${REPLAY_MODE}.json"

if [ ${PYTHON_EXIT_CODE} -eq 0 ] && [ -f "${REPLAY_JSON_PATH}" ]; then
    echo "Python script executed successfully and replay JSON found."
    python3 -c "
import json
with open('${REPLAY_JSON_PATH}', 'r') as f:
    data = json.load(f)
success = False
desc_str = 'I2I result assert failed'
for k, v in data.items():
    if isinstance(v, dict) and 'success' in v:
        success = v['success']
        break
if success:
    desc_str = '${HDF5_FILENAME}--${REPLAY_MODE}: i2i success.'
with open('${OUTPUT_JSON}', 'w') as out_f:
    json.dump({'success': success, 'desc': desc_str, 'error': None}, out_f, indent=4, ensure_ascii=False)
"
elif [ ${PYTHON_EXIT_CODE} -ne 0 ]; then
    echo "Python script exited with code ${PYTHON_EXIT_CODE}."
    check_for_stack_trace "${PYTHON_LOG}"
    STACK_SHORT=""
    if [ "$STACK_TRACE" = "" ]; then
        echo "No stack trace found in log."
        last_three_lines=$(tail -n 3 ${PYTHON_LOG})
        STACK_TRACE="Python script exited with code ${PYTHON_EXIT_CODE}, but no stack trace found in log. Last three lines of log:\n${last_three_lines}"
        STACK_SHORT=$STACK_TRACE
    elif [ $(echo "$STACK_TRACE" | wc -l) -ge 3 ]; then
        STACK_SHORT=$(echo "$STACK_TRACE" | tail -3)
    else
        STACK_SHORT="$STACK_TRACE"
    fi

    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump({
        'success': False,
        'desc': '${STACK_SHORT}',
        'error': '${STACK_SHORT}'
    }, f, indent=4, ensure_ascii=False)
"
    exit 1
else
    echo "Replay JSON not found at expected path: ${REPLAY_JSON_PATH}."
    python3 -c "
import json
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump({
        'success': False,
        'desc': '${REPLAY_JSON_PATH} not generated!',
        'error': '${REPLAY_JSON_PATH} not generated!'
    }, f, indent=4, ensure_ascii=False)
"
    exit 1
fi