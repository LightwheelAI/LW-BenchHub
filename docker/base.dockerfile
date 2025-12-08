FROM nvcr.io/nvidia/base/ubuntu:noble-20250619

# Set all environment variables at once
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    OMNI_SERVER=https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 \
    MIN_DRIVER_VERSION=570.169 \
    OMNI_KIT_ACCEPT_EULA=YES \
    LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    OMNI_KIT_ALLOW_ROOT=1 \
    CONDA_DIR=/opt/conda \
    ENV_NAME=lw_benchhub \
    PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH" \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Disable hardware-specific library optimizations
RUN touch /etc/ld.so.nohwcap

# Install all dependencies in one layer
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        # Basic system packages
        curl wget bzip2 unzip ca-certificates sudo git git-lfs \
        # Build tools
        build-essential cmake pkg-config \
        # Graphics and OpenGL
        libatomic1 libegl1 libgl1 libglu1-mesa libglx0 libgomp1 \
        libsm6 libxi6 libxrandr2 libxt6 libglib2.0-0 libnghttp2-14 \
        x11-apps mesa-utils libgl1-mesa-dri libglx-mesa0 \
        libx11-6 libxext6 libxrender1 libxfixes3 \
        # Media and utilities
        ffmpeg vim lsof ncurses-term openssh-server htop nvtop && \
    # Clean up
    apt-get -y autoremove && apt-get clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Install runtime configuration files
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d && \
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json && \
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_mesa.so.0"\n    }\n}\n' > /usr/share/glvnd/egl_vendor.d/50_mesa.json && \
    printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194"\n    }\n}\n' > /etc/vulkan/icd.d/nvidia_icd.json && \
    printf '{\n    "file_format_version" : "1.0.0",\n    "layer": {\n        "name": "VK_LAYER_NV_optimus",\n        "type": "INSTANCE",\n        "library_path": "libGLX_nvidia.so.0",\n        "api_version" : "1.3.194",\n        "implementation_version" : "1",\n        "description" : "NVIDIA Optimus layer",\n        "functions": {\n            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",\n            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"\n        },\n        "enable_environment": {\n            "__NV_PRIME_RENDER_OFFLOAD": "1"\n        },\n        "disable_environment": {\n            "DISABLE_LAYER_NV_OPTIMUS_1": ""\n        }\n    }\n}\n' > /etc/vulkan/implicit_layer.d/nvidia_layers.json

# Open ports for live streaming
EXPOSE 47998/udp 49100/tcp

# Download and install Miniconda, create environment, and install packages
WORKDIR /workspace
RUN mkdir -p /workspace/lw_benchhub/docker && \
    wget -O miniconda.sh https://artifactory.lightwheel.net/data/miniconda/miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    # Initialize conda and accept terms
    $CONDA_DIR/bin/conda init bash && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    $CONDA_DIR/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    $CONDA_DIR/bin/conda update -y -n base conda && \
    # Clean up miniconda installer
    rm miniconda.sh

# Copy environment file and create conda environment
COPY ./docker/environment.yml /workspace/lw_benchhub/docker/environment.yml
WORKDIR /workspace/lw_benchhub/docker
RUN $CONDA_DIR/bin/conda env create -f environment.yml && \
    $CONDA_DIR/bin/conda clean -afy && \
    echo "source activate $ENV_NAME" > ~/.bashrc

# Set shell for subsequent RUN commands
SHELL ["/bin/bash", "-c"]

# Install Python packages
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install numpy==1.26.4 --extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
        "isaacsim[all,extscache]==5.0.0" && \
    $CONDA_DIR/bin/conda clean -afy

# Copy and install IsaacLab
WORKDIR /workspace
RUN mkdir -p /workspace/lw_benchhub/third_party/IsaacLab-Arena/submodules
COPY ./third_party/IsaacLab-Arena/submodules/IsaacLab /workspace/lw_benchhub/third_party/IsaacLab-Arena/submodules/IsaacLab
COPY ./third_party/IsaacLab-Arena/submodules/Isaac-GR00T /workspace/lw_benchhub/third_party/IsaacLab-Arena/submodules/Isaac-GR00T

# set git config
RUN git config --global http.postBuffer 524288000
RUN git config --global https.postBuffer 524288000

WORKDIR /workspace/lw_benchhub/third_party/IsaacLab-Arena/submodules/IsaacLab
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate lw_benchhub && \
    git config --global http.proxy http://127.0.0.1:7897 && \
    # pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
    # Retry logic for Isaac Lab installation
    for i in {1..3}; do \
        echo "Attempting to install Isaac Lab dependencies (attempt $i/3)..."; \
        if ./isaaclab.sh --install; then \
            echo "✅ Installation successful on attempt $i"; \
            break; \
        else \
            echo "❌ Installation failed on attempt $i"; \
            if [ $i -eq 3 ]; then \
                echo "💥 All 3 attempts failed, exiting with error"; \
                exit 1; \
            fi; \
            echo "⏳ Waiting 5 seconds before retry..."; \
            sleep 5; \
        fi; \
    done && \
    echo "Clean up proxy settings" && \
    # save clear git proxy config
    git config --global --unset http.proxy 2>/dev/null || true && \
    git config --global --unset https.proxy 2>/dev/null || true && \
    # save clear pip config
    pip config unset global.extra-index-url 2>/dev/null || true && \
    echo "Clean up proxy settings finish"


RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate lw_benchhub && \
    git config --global http.proxy http://127.0.0.1:7897 && \
    # pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install lerobot==0.3.3  && \
    echo "Clean up proxy settings" && \
    # save clear git proxy config
    git config --global --unset http.proxy 2>/dev/null || true && \
    git config --global --unset https.proxy 2>/dev/null || true && \
    # save clear pip config
    pip config unset global.extra-index-url 2>/dev/null || true && \
    echo "Clean up proxy settings finish"
