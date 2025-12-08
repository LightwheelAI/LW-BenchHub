#!/bin/bash

DATASET_FILE=$1
HDF5_FILENAME=$(basename "${DATASET_FILE}")
DATASET_DIR=/output

cp "${DATASET_FILE}" "${DATASET_DIR}"
DATASET_FILE="${DATASET_DIR}/${HDF5_FILENAME}"

OUTPUT_JSON="${DATASET_DIR}/result.json"

python3 /workspace/lw_benchhub/lw_benchhub/scripts/teleop/replay_demos.py \
    --dataset_file=${DATASET_FILE} \
    --num_envs=1 \
    --width=1280 \
    --height=720 \
    --enable_cameras \
    --headless \
    --without_image 2>&1
