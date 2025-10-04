FROM harbor.lightwheel.net/robot/lwlab:isaaclab_base5.0

ENV CONDA_DIR=/opt/conda
ENV ENV_NAME=lwlab
ENV PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH"

# RUN mkdir -p /root/.ssh && \
#     chmod 700 /root/.ssh
# COPY docker/.ssh/id_rsa /root/.ssh/id_rsa
# COPY docker/.ssh/id_rsa.pub /root/.ssh/id_rsa.pub
# COPY docker/.ssh/known_hosts /root/.ssh/known_hosts
# RUN chmod 600 /root/.ssh/id_rsa && \
#     chmod 644 /root/.ssh/id_rsa.pub && \
#     chmod 644 /root/.ssh/known_hosts

RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install autopep8 flake8 --extra-index-url https://mirrors.aliyun.com/pypi/simple/


COPY ./third_party/robocasa_upload /workspace/lwlab/third_party/robocasa_upload
WORKDIR /workspace/lwlab/third_party/robocasa_upload
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e . --extra-index-url https://mirrors.aliyun.com/pypi/simple/

# Copy all directories except third_party to /workspace/lwlab
COPY . /workspace/lwlab_bak/
RUN rm -rf /workspace/lwlab_bak/third_party 
RUN rm -rf /workspace/lwlab_bak/docker
RUN mv /workspace/lwlab_bak/* /workspace/lwlab/
RUN mv /workspace/lwlab_bak/.git* /workspace/lwlab/

WORKDIR /workspace/lwlab
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

RUN git config --global --add safe.directory /workspace/lwlab
RUN git config --global --add safe.directory /workspace/lwlab/third_party/IsaacLab

WORKDIR /workspace/lwlab

ENTRYPOINT ["/bin/bash", "-c", "-i"]