#!/bin/bash

# CI script to update workspace to a specific commit and run a command
# Usage: ./run_ci.sh <commit_id> <command>

set -e  # Exit on any error

# Check if both arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <commit_id> <command>"
    echo "Example: $0 abc123 'python test.py'"
    exit 1
fi

COMMIT_ID="$1"
shift  # Remove first argument, leaving the command and its arguments
COMMAND="$@"

echo "=== CI Script Started ==="
echo "Commit ID: $COMMIT_ID"
echo "Command: $COMMAND"
echo "Current directory: $(pwd)"

# Navigate to the workspace based on $ENV_NAME
cd /workspace/"$ENV_NAME"

if [[ $COMMIT_ID =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "=== Updating to commit $COMMIT_ID ==="

    # Check if we're in a git repository
    if [ ! -d ".git" ]; then
        echo "Error: /workspace/$ENV_NAME is not a git repository"
        exit 1
    fi

    # Fetch latest changes
    echo "Fetching latest changes..."
    git fetch --all

    # Check if the commit exists
    if ! git rev-parse --verify "$COMMIT_ID" >/dev/null 2>&1; then
        echo "Error: Commit $COMMIT_ID not found"
        exit 1
    fi

    # Reset to the specified commit (hard reset to ensure clean state)
    echo "Resetting to commit $COMMIT_ID..."
    git reset --hard "$COMMIT_ID"
    git lfs pull
    sleep 3

    git submodule update --init --recursive

    # Clean any untracked files (optional, but good for CI)
    echo "Cleaning untracked files..."
    git clean -fd

    # lw_benchhub install
    echo "Install $ENV_NAME files..."
    pip install -e .  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

    # IsaacLab-Arena install
    pushd ./third_party/IsaacLab-Arena/
    echo "Install IsaacLab-Arena files..."
    pip install -e .  --extra-index-url https://mirrors.aliyun.com/pypi/simple/
    popd
else 
    echo "Skipping git operations. Keep current code in docker."
fi

echo "=== Running command: $COMMAND ==="

# Delegate the actual command run to run_ci_post.sh for unified post-logic
bash /workspace/$ENV_NAME/run_ci_post.sh "$COMMAND"
