FROM harbor.lightwheel.net/robot/lwlab:isaaclab_base5.0

ENV CONDA_DIR=/opt/conda
ENV ENV_NAME=lwlab
ENV PATH="$CONDA_DIR/bin:$CONDA_DIR/envs/$ENV_NAME/bin:$PATH"

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

WORKDIR /workspace/lwlab
RUN source $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install -e .  --extra-index-url https://mirrors.aliyun.com/pypi/simple/

WORKDIR /workspace/lwlab

ENTRYPOINT ["/bin/bash", "-c", "-i"]