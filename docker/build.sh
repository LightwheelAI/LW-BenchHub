#!/bin/bash
# Combined build script for lw_benchhub Docker images

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Base Image
BASE_IMAGE_NAME="lw_benchhub"
BASE_TAG="base"

# Application Image
APP_IMAGE_NAME="lw_benchhub"
APP_TAG="1.0"

# Build Arguments
PROXY_HOST=${PROXY_HOST:-"10.10.11.110"}
PROXY_PORT=${PROXY_PORT:-"7890"}

# --- Functions ---
build_base_image() {
    echo "--- Building Base Image: ${BASE_IMAGE_NAME}:${BASE_TAG} ---"
    
    # export DOCKER_BUILDKIT=1
        # --progress=auto \
    
    docker build \
        --file docker/base.dockerfile \
        --build-arg PROXY_HOST=${PROXY_HOST} \
        --build-arg PROXY_PORT=${PROXY_PORT} \
        -t ${BASE_IMAGE_NAME}:${BASE_TAG} \
        .
        
        # --load \
        
    echo "✅ Base image build complete: ${BASE_IMAGE_NAME}:${BASE_TAG}"
}

build_app_image() {
    echo "--- Building Application Image: ${APP_IMAGE_NAME}:${APP_TAG} ---"
    
    # export DOCKER_BUILDKIT=1
    
    docker build \
        --file docker/Dockerfile \
        --build-arg PROXY_HOST=${PROXY_HOST} \
        --build-arg PROXY_PORT=${PROXY_PORT} \
        -t ${APP_IMAGE_NAME}:${APP_TAG} \
        .
    docker rm -f lw_benchhub_warmup || true
    docker run -it --name lw_benchhub_warmup --gpus all ${APP_IMAGE_NAME}:${APP_TAG} -c -i "https_proxy=http://10.10.11.36:7897 http_proxy=http://10.10.11.36:7897 all_proxy=socks5://10.10.11.36:7897 python /workspace/lw_benchhub/lw_benchhub/scripts/teleop/warmup_isaac.py"
    docker commit lw_benchhub_warmup ${APP_IMAGE_NAME}:${APP_TAG}
    docker rm -f lw_benchhub_warmup || true
        
    echo "✅ Application image build complete: ${APP_IMAGE_NAME}:${APP_TAG}"
}

show_usage() {
    echo "Usage: $0 [base|app|all]"
    echo "  base: Build only the base image (${BASE_IMAGE_NAME}:${BASE_TAG})"
    echo "  app:  Build only the application image (${APP_IMAGE_NAME}:${APP_TAG})"
    echo "  all:  Build both the base and application images (default)"
    exit 1
}

# --- Main Logic ---
if [ "$#" -gt 1 ]; then
    show_usage
fi

ACTION=${1:-"all"}

case "$ACTION" in
    base)
        build_base_image
        ;;
    app)
        build_app_image
        ;;
    all)
        build_base_image
        build_app_image
        ;;
    *)
        show_usage
        ;;
esac

echo "--- Build Process Finished ---"
echo ""
echo "Or use the direct docker command to get an interactive shell:"
echo "docker run -it --rm --gpus all ${APP_IMAGE_NAME}:${APP_TAG}" 