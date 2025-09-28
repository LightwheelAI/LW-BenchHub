#from wanggeng 0925
FROM harbor.lightwheel.net/robot/lwlab:ci_feb16d11 

ENV OMNI_KIT_ACCEPT_EULA=YES
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV OMNI_KIT_ALLOW_ROOT=1

# Set environment variables (reduce interactive prompts & define environment name)
ENV CONDA_DIR=/opt/conda
ENV ENV_NAME=lwlab
ENV PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH"

# # Install dependencies and remove cache
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     cmake \
#     ffmpeg \
#     vim \
#     lsof \
#     git \
#     git-lfs \
#     bzip2 \
#     curl \
#     sudo \
#     ca-certificates \
#     libglib2.0-0 \
#     ncurses-term \
#     wget \
#     pkg-config \ 
#     x11-apps mesa-utils libgl1-mesa-glx libgl1-mesa-dri libegl1-mesa \
#     libx11-6 libxext6 libxrender1 libxrandr2 libxfixes3 libxi6 \
#     openssh-server htop nvtop && \
#     apt -y autoremove && apt clean autoclean && \
#     rm -rf /var/lib/apt/lists/*


# # Download and silently install Miniconda
# WORKDIR /workspace
# RUN mkdir -p /workspace/lwlab/docker
# COPY ./docker/miniconda.sh /workspace/lwlab/docker/miniconda.sh
# COPY ./docker/environment.yml /workspace/lwlab/docker/environment.yml


# WORKDIR /workspace/lwlab/docker
# RUN bash miniconda.sh -b -p $CONDA_DIR && \
#     # Initialize conda
#     conda init bash && \
#     # Accept Terms of Service for main channels
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
#     conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
#     # Update conda base packages
#     conda update -y -n base conda


# # Create new environment and install packages (using conda-lock style dependency management)
# RUN conda env create -f environment.yml && \
#     conda clean -afy && \
#     # Create shortcut for activating environment
#     echo "source activate $ENV_NAME" > ~/.bashrc

# # Activate environment and install core dependencies
# SHELL ["/bin/bash", "-c"]

# RUN source activate $ENV_NAME && \
#     pip install numpy==1.26.4 --extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
#     # Install NVIDIA Isaac Sim core components
#     pip install --no-cache-dir \
#         --extra-index-url https://pypi.nvidia.com \
#         --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
#         "isaacsim[all,extscache]==4.5.0" && \
#     # Clean cache
#     conda clean -afy

# # Copy IsaacLab
# WORKDIR /workspace
# RUN mkdir -p /workspace/lwlab/third_party
# COPY ./third_party/IsaacLab /workspace/lwlab/third_party/IsaacLab

# WORKDIR /workspace/lwlab/third_party/IsaacLab/
# RUN source activate $ENV_NAME && \
#     git config --global http.proxy http://127.0.0.1:8890 && \
#     pip config set global.extra-index-url https://mirrors.aliyun.com/pypi/simple/ && \
#     for i in {1..3}; do \
#     echo "Attempting to install Isaac Lab dependencies (attempt $i/3)..."; \
#     if ./isaaclab.sh --install; then \
#         echo "✅ Installation successful on attempt $i"; \
#         break; \
#     else \
#         echo "❌ Installation failed on attempt $i"; \
#         if [ $i -eq 3 ]; then \
#             echo "💥 All 3 attempts failed, exiting with error"; \
#             exit 1; \
#         fi; \
#         echo "⏳ Waiting 5 seconds before retry..."; \
#         sleep 5; \
#     fi; \
#     done

# RUN source activate $ENV_NAME && \
#     pip install meshcat==0.3.2 --extra-index-url https://mirrors.aliyun.com/pypi/simple/
# RUN source activate $ENV_NAME && \
#     pip install casadi==3.7.0 vuer[all]==0.0.60 pin-pink==3.1.0 --extra-index-url https://mirrors.aliyun.com/pypi/simple/

# ======================================================================
# RUN mkdir -p /root/.ssh && \
#     chmod 700 /root/.ssh
# COPY docker/.ssh/id_rsa /root/.ssh/id_rsa
# COPY docker/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
# COPY docker/.ssh/known_hosts /root/.ssh/known_hosts
# RUN chmod 600 /root/.ssh/id_rsa && \
#     chmod 644 /root/.ssh/id_rsa.pub && \
#     chmod 644 /root/.ssh/known_hosts

RUN mv /workspace/lwlab/third_party /workspace/third_party_origin_bak
RUN rm -rf /workspace/lwlab/* && mkdir -p /workspace/lwlab
COPY . /workspace/lwlab/
RUN rm -rf /workspace/lwlab/third_party 
RUN rm -rf /workspace/lwlab/docker

RUN mkdir -p /workspace/lwlab/third_party/ && mv /workspace/third_party_origin_bak/* /workspace/lwlab/third_party/

WORKDIR /workspace/lwlab
RUN source activate $ENV_NAME && \
    pip install -e .  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

WORKDIR /workspace/lwlab

ENTRYPOINT ["/bin/bash", "-c", "-i"]