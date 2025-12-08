FROM harbor.lightwheel.net/robot/lw_benchhub:base_isaaclab-fb270ab5_gr00t-3bce5530_lerobot_1108


ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV CONDA_DIR=/opt/conda
ENV ENV_NAME=lw_benchhub
ENV PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH"
# proxy from arg
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

SHELL ["/bin/bash", "-c"]
RUN $CONDA_DIR/bin/conda init bash

### local current dir is lw_benchhub/ (with submodules), and add ./third_party/robocasa_upload
# Copy all directories to /workspace/lw_benchhub
RUN rm -rf /workspace/lw_benchhub
COPY . /workspace/lw_benchhub/
RUN rm -rf /workspace/lw_benchhub/docker

# install robocasa_upload
WORKDIR /workspace/lw_benchhub/third_party/robocasa_upload
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .

# install lw_benchhub
WORKDIR /workspace/lw_benchhub
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .

# install IsaacLab-Arena
WORKDIR /workspace/lw_benchhub/third_party/IsaacLab-Arena
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .

# clear proxy
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
RUN unset HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

WORKDIR /workspace/lw_benchhub

ENTRYPOINT ["/bin/bash", "-c", "-i"]