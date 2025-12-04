# piper 匹诺曹逆解

## conda环境搭建

代码只使用了pinocchio=3.6.0=py39hfd66edf_0版本的pinocchio测试

```shell
conda create -n tv python=3.9
conda activate tv
conda install pinocchio=3.6.0 -c conda-forge
pip install meshcat
pip install casadi
```

## 运行

运行piper单臂控制节点(也可以单独运行逆解，这一步是用来控制真实机械臂的)

```bash
roslaunch piper start_single_piper.launch
```

直接使用conda的python运行文件

```bash
source /{你的路径}/piper_ws/devel/setup.sh
conda activate tv
python /{你的路径}/piper_ws/src/piper/scripts/piper_pinocchio/piper_pinocchio.py
```
## Q&A

### 1

安装了`pinocchio`但是却出现

```bash
ImportError: cannot import name 'casadi' from 'pinocchio:
```

请先查看pinocchio版本

```bash
# 激活环境
conda activate vt
# 查询包
conda list | grep pinocchio
# 或
conda env export | grep pinocchio
```

该问题是由于casadi没有内置在pinocchio中导致

在其工程的changelog中记录到3.0.0之后的版本内置了casadi，但是3.7.0 drop了20.04的支持

<https://github.com/stack-of-tasks/pinocchio/blob/devel/CHANGELOG.md#300---2024-05-27>

使用以下指令查看可安装的pinocchio版本

```bash
conda search pinocchio -c conda-forge
```

查看是否有本文推荐的版本
