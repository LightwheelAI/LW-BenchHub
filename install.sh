# isaaacsim
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
pip install --upgrade setuptools
pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com

# create third_party dir
mkdir -p third_party

# IsaacLab
cd third_party/isaac_arena/submodules/IsaacLab
bash isaaclab.sh --install
cd ../../../..

# robosuite
cd third_party
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout robocasa-dev
pip install -e .
cd ../..

# robocasa
cd third_party
git clone https://github.com/LightwheelAI/robocasa.git
cd robocasa
git checkout lwlab
pip install -e .
cd ../..

# lwlab
pip install -e .
