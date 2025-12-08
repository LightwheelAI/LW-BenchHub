#!/bin/bash

# CI script to update workspace to a specific commit and run a command
# Usage: ./run_ci.sh <commit_id> <command>

set -e  # Exit on any error

unset LW_SDK_HEADERS_X_FROM_LIGHTWHEEL_CLOUD

if [ -n "$CONDA_DIR" ] && [ -n "$ENV_NAME" ]; then
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi

COMMAND="$@"

# Run the provided command
eval "$COMMAND"

EXIT_CODE=$?

echo "=== Command completed with exit code: $EXIT_CODE ==="

exit $EXIT_CODE
