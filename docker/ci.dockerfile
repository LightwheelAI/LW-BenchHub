FROM harbor.lightwheel.net/robot/lw_benchhub:base_isaaclab-fb270ab5_gr00t-3bce5530_lerobot_1108

# build arg
ARG SSH_PRIVATE_KEY=""
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV CONDA_DIR=/opt/conda
ENV ENV_NAME=lw_benchhub
ENV PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH"
# proxy from arg
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh
# COPY docker/.ssh/id_rsa /root/.ssh/id_rsa
# COPY docker/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
RUN if [ -n "$SSH_PRIVATE_KEY" ]; then \
        echo "$SSH_PRIVATE_KEY" > /root/.ssh/id_rsa && \
        chmod 600 /root/.ssh/id_rsa; \
    fi
COPY docker/.ssh/known_hosts /root/.ssh/known_hosts
RUN chmod 644 /root/.ssh/known_hosts
# RUN chmod 600 /root/.ssh/id_rsa && \
#     chmod 644 /root/.ssh/id_rsa.pub && \
#     chmod 644 /root/.ssh/known_hosts

# RUN mkdir -p /root/.ssh && \
#     ssh-keyscan -p 2022 git.lightwheel.ai >> /root/.ssh/known_hosts && \
#     chmod 644 /root/.ssh/known_hosts

RUN echo "Host git.lightwheel.ai" >> /root/.ssh/config && \
    echo "    Port 2022" >> /root/.ssh/config && \
    echo "    StrictHostKeyChecking no" >> /root/.ssh/config && \
    chmod 600 /root/.ssh/config

SHELL ["/bin/bash", "-c"]
RUN $CONDA_DIR/bin/conda init bash

RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install autopep8 flake8

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

RUN git config --global --add safe.directory /workspace/lw_benchhub
RUN git config --global --add safe.directory /workspace/lw_benchhub/third_party/IsaacLab-Arena
RUN git config --global --add safe.directory /workspace/lw_benchhub/third_party/IsaacLab-Arena/submodules/IsaacLab
RUN git config --global --add safe.directory /workspace/lw_benchhub/third_party/IsaacLab-Arena/submodules/Isaac-GR00T

# clear proxy
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
RUN unset HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

WORKDIR /workspace/lw_benchhub

ENTRYPOINT ["/bin/bash", "-c", "-i"]