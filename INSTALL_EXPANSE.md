# EXPANSE HPC Cluster 安装指南

本指南专门针对在 EXPANSE HPC Cluster 上安装 Oracle-Guided-RL 项目。

## 前置要求

- EXPANSE 账户和访问权限
- 已安装 Miniconda 或 Anaconda
- 访问 `/expanse/lustre/projects/chi157/jji3` 目录的权限

## 安装步骤

### 1. 登录 EXPANSE 并导航到工作目录

```bash
# SSH 登录 EXPANSE
ssh your_username@login.expanse.sdsc.edu

# 导航到项目目录（不在 $HOME）
cd /expanse/lustre/projects/chi157/jji3
```

### 2. 克隆仓库

```bash
git clone https://github.com/LucienJi/Oracle-Guided-RL.git
cd Oracle-Guided-RL
```

### 3. 配置路径

运行路径配置脚本（重要！）：

```bash
bash setup_paths.sh
```

脚本会：
- 自动检测 EXPANSE 集群
- 提示你确认项目根目录（默认：`/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL`）
- 创建本地路径配置文件 `config/paths_local.yaml`（gitignored）
- 可选：添加到 `~/.bashrc`

### 4. 创建 Conda 环境

```bash
# 加载 conda（如果还没有）
module load gpu
module load cuda

# 创建环境
conda env create -f requirements/oracles.yaml

# 如果遇到路径问题，可以手动安装：
# conda create -n oracles python=3.10 -y
# conda activate oracles
# conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.4 -y
# conda install -c conda-forge mujoco>=3.3 glfw glew ffmpeg -y
# pip install -r requirements/base.txt
```

### 5. 安装 MuJoCo 和其他仿真环境

#### 5.1 MuJoCo

MuJoCo 应该已经通过 conda 安装。验证：

```bash
conda activate oracles
python -c "import mujoco; print('MuJoCo version:', mujoco.__version__)"
```

如果遇到问题，可以手动安装：

```bash
# 安装 MuJoCo Python bindings
pip install mujoco

# 如果需要系统库
# EXPANSE 上通常已经安装了必要的库
```

#### 5.2 安装第三方库

```bash
conda activate oracles

# 安装 CARL
cd third_party/CARL
pip install -e .
cd ../..

# 安装 Metaworld
cd third_party/Metaworld
pip install -e .
cd ../..

# 安装 HighwayEnv
cd third_party/HighwayEnv
pip install -e .
cd ../..

# 安装 MyoSuite（如果使用）
cd third_party/myosuite
pip install -e .
cd ../..
```

### 6. 设置环境变量

添加到 `~/.bashrc` 或 `~/.bash_profile`：

```bash
# Oracle-Guided-RL
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl  # 用于无头渲染（HPC 环境）

# 加载 conda
source ~/.bashrc  # 或 source ~/.bash_profile
```

### 7. 验证安装

```bash
conda activate oracles
cd /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL

# 验证关键依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import mujoco; print('MuJoCo OK')"
python -c "import gymnasium; print('Gymnasium OK')"
python -c "import dm_control; print('dm-control OK')"
python -c "import metaworld; print('Metaworld OK')"
```

## EXPANSE 特定注意事项

### 模块系统

EXPANSE 使用模块系统。加载必要的模块：

```bash
module load gpu
module load cuda/12.2.0  # 或你需要的版本
module load gcc/9.2.0    # 如果需要编译
```

### SLURM 作业提交

创建 SLURM 脚本示例：

```bash
#!/bin/bash
#SBATCH --job-name=oracle_train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# 加载模块
module load gpu
module load cuda

# 激活环境
source ~/.bashrc
conda activate oracles

# 设置路径
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl

# 运行训练
cd $ORACLES_PROJECT_ROOT
python scripts/train_*.py
```

### 数据存储

- 项目代码：`/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL`
- 检查点/数据：`/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL/checkpoints`
- 结果：`/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL/results`

### 无头渲染

在 HPC 环境中，使用 EGL 进行无头渲染：

```bash
export MUJOCO_GL=egl
```

确保在运行脚本和 SLURM 作业中都设置了这个变量。

## 常见问题

### CUDA 版本不匹配

EXPANSE 可能使用不同的 CUDA 版本。检查可用版本：

```bash
module avail cuda
```

然后修改 `requirements/oracles.yaml` 中的 `pytorch-cuda` 版本，或使用 pip 安装对应版本。

### 路径问题

如果配置文件找不到路径：

1. 确保运行了 `bash setup_paths.sh`
2. 检查 `config/paths_local.yaml` 是否存在且路径正确
3. 设置环境变量：`export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL`

### MuJoCo 渲染问题

如果遇到渲染问题：

```bash
export MUJOCO_GL=egl  # 无头渲染
# 或
export MUJOCO_GL=osmesa  # 软件渲染
```

## 下一步

安装完成后：
1. 检查配置文件路径是否正确
2. 准备数据/检查点（如果需要）
3. 提交 SLURM 作业开始训练

更多信息请查看主安装文档 [INSTALL.md](INSTALL.md)

