pip install upgrade pip
pip install uv
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
conda install pinocchio -c conda-forge -y

# install isaacsim
uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

git submodule update --init --recursive

# install isaaclab
cd third_party/IsaacLab-Arena/submodules/IsaacLab
bash isaaclab.sh --install

# install isaaclab-arena
cd ../..
uv pip install -e .

# install lw_benchhub
cd ../..
uv pip install -e .
# If you need to use LeRobot in LW-BenchHub, install it's requirements by
# uv pip install -e .[lerobot]

