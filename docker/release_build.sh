#!/bin/bash

# Check if IsaacLab is at the specified commit
set -e

# Function to restore .dockerignore backup if it exists
restore_dockerignore() {
    if [ -f ".dockerignore.backup" ]; then
        mv .dockerignore.backup .dockerignore
        echo "✅ Restored original .dockerignore from backup"
    fi
    DOCKFILE_CI=false
}

# Configuration: Set the expected commit hash here
IsaacLab_commit="0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20"
DOCKER_TAG=$1
lw_benchhub_commit=$2

echo "🔍 Checking if IsaacLab is at the specified commit: $IsaacLab_commit"

# Check if IsaacLab directory exists
if [ ! -d "third_party/IsaacLab" ]; then
    echo "❌ Error: third_party/IsaacLab directory does not exist"
    exit 1
fi


# Use pushd to enter IsaacLab directory
pushd third_party/IsaacLab

# Check if it's a Git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: IsaacLab is not a Git repository"
    popd  # Return to original directory before exiting
    exit 1
fi

# Get current commit hash
CURRENT_COMMIT=$(git rev-parse HEAD)
CURRENT_SHORT_COMMIT=$(git rev-parse --short HEAD)

echo "�� Current commit: $CURRENT_SHORT_COMMIT"

# Check if commit matches
if [ "$CURRENT_COMMIT" != "$IsaacLab_commit" ]; then
    echo "❌ Error: IsaacLab is not at the expected commit"
    echo "   Current: $CURRENT_COMMIT"
    echo "   Expected: $IsaacLab_commit"
    echo ""
    echo "Please execute the following commands to switch to the specified commit:"
    echo "cd third_party/IsaacLab"
    echo "git checkout $IsaacLab_commit"
    popd  # Return to original directory before exiting
    exit 1
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Error: IsaacLab has uncommitted changes:"
    git status --porcelain
    popd  # Return to original directory before exiting
    exit 1
fi

# Check if there are untracked files
untracked=$(git ls-files --others --exclude-standard)
if [ -n "$untracked" ]; then
    echo "❌ Error: IsaacLab has untracked files:"
    echo "$untracked"
    popd  # Return to original directory before exiting
    exit 1
fi

echo "✅ IsaacLab check passed!"

# Use popd to return to original directory
popd
echo "🎯 IsaacLab is ready for Docker build"
echo ""

echo "🔍 Checking if lw_benchhub is at the specified commit: $lw_benchhub_commit"

# Check if it's a Git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: lw_benchhub is not a Git repository"
    exit 1
fi

# Get current commit hash
CURRENT_COMMIT=$(git rev-parse HEAD)
CURRENT_SHORT_COMMIT=$(git rev-parse --short HEAD)

echo "�� Current commit: $CURRENT_SHORT_COMMIT"

# Check if commit matches
if [ "$CURRENT_COMMIT" != "$lw_benchhub_commit" ]; then
    echo "❌ Error: lw_benchhub is not at the expected commit"
    echo "   Current: $CURRENT_COMMIT"
    echo "   Expected: $lw_benchhub_commit"
    echo ""
    echo "Please execute the following command to switch to the specified commit:"
    echo "git checkout $lw_benchhub_commit"
    exit 1
fi

# Check if there are uncommitted changes (excluding untracked files)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "❌ Error: lw_benchhub has uncommitted changes:"
    git status --porcelain --untracked-files=no
    exit 1
fi

# Check if there are untracked files
untracked=$(git ls-files --others --exclude-standard)
if [ -n "$untracked" ]; then
    echo "⚠️  Warning: lw_benchhub has untracked files:"
    echo "$untracked"
    echo "   These files won't affect the build, but it's recommended to manage them"
    # Don't exit, continue execution
else
    echo "✅ lw_benchhub has no untracked files"
fi

echo "✅ lw_benchhub check passed!"
echo "�� lw_benchhub is ready for Docker build"
echo ""

#! /bin/bash
DIR="$(pwd)"
echo "Current workdir: ${DIR}"

# Store original DOCKER_TAG for later use
ORIGINAL_DOCKER_TAG="$DOCKER_TAG"
DOCKFILE_CI=false
# Choose dockerfile and dockerignore based on DOCKER_TAG
if [ "$DOCKER_TAG" = "ci" ]; then
    DOCKERFILE="docker/ci.dockerfile"
    DOCKERIGNORE=".dockerignore.ci"
    echo "Using CI dockerfile: $DOCKERFILE"
    echo "Using CI dockerignore: $DOCKERIGNORE"
    DOCKFILE_CI=true
    # Check if .dockerignore.ci exists
    if [ ! -f "$DOCKERIGNORE" ]; then
        echo "❌ Error: $DOCKERIGNORE file does not exist"
        exit 1
    fi
    
    # Backup original .dockerignore and use .dockerignore.ci
    if [ -f ".dockerignore" ]; then
        cp .dockerignore .dockerignore.backup
        echo "Backed up original .dockerignore to .dockerignore.backup"
    fi
    cp "$DOCKERIGNORE" .dockerignore
    echo "Using $DOCKERIGNORE as .dockerignore for build"
    
    # Verify the copy was successful
    if [ ! -f ".dockerignore" ]; then
        echo "❌ Error: Failed to copy $DOCKERIGNORE to .dockerignore"
        restore_dockerignore
        exit 1
    fi
else
    DOCKERFILE="docker/release.dockerfile"
    DOCKERIGNORE=".dockerignore"
    echo "Using release dockerfile: $DOCKERFILE"
    echo "Using release dockerignore: $DOCKERIGNORE"
fi

DOCKER_TAG="${DOCKER_TAG}_${lw_benchhub_commit:0:8}"
echo "Combined DOCKER_TAG: ${DOCKER_TAG}"

if [[ -z $DOCKER_TAG ]]
    then
        echo "[ERROR]: DOCKER_TAG is missing..."
        exit 1
fi

pushd "${DIR}"
echo "Docker building..."

docker build --network=host -t harbor.lightwheel.net/robot/lw_benchhub:${DOCKER_TAG} -f ${DOCKERFILE} .
if [ "$DOCKFILE_CI" = true ]; then
    restore_dockerignore
fi
docker tag harbor.lightwheel.net/robot/lw_benchhub:${DOCKER_TAG} lw-ali-harbor-registry.cn-shanghai.cr.aliyuncs.com/robot/lw_benchhub:${DOCKER_TAG}
popd

docker push harbor.lightwheel.net/robot/lw_benchhub:${DOCKER_TAG}
echo "Pushed to harbor.lightwheel.net/robot/lw_benchhub:${DOCKER_TAG}"
docker push lw-ali-harbor-registry.cn-shanghai.cr.aliyuncs.com/robot/lw_benchhub:${DOCKER_TAG}
echo "Pushed to lw-ali-harbor-registry.cn-shanghai.cr.aliyuncs.com/robot/lw_benchhub:${DOCKER_TAG}"
echo "docker download url for cloud: lw-ali-harbor-registry-vpc.cn-shanghai.cr.aliyuncs.com/robot/lw_benchhub:${DOCKER_TAG}"

echo "=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=* $(date +%Y-%m-%d\ %H:%M:%S) All process finished =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*"