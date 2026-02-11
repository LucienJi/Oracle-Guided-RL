# EXPANSE 集群完整设置流程

本文档提供在 EXPANSE HPC Cluster 上设置 Oracle-Guided-RL 项目的完整步骤。

## 前置准备

### 1. 登录 EXPANSE

```bash
ssh your_username@login.expanse.sdsc.edu
```

### 2. 检查 Git 是否已安装

```bash
which git
git --version
```

如果没有安装，EXPANSE 通常有 git 模块：

```bash
module avail git
module load git  # 如果需要
```

或者使用系统自带的 git（通常已安装）。

### 3. 配置 Git（首次使用）

```bash
# 设置用户名和邮箱
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 验证配置
git config --list
```

### 4. 设置 SSH 密钥（推荐，用于 GitHub）

如果你还没有设置 SSH 密钥：

```bash
# 生成 SSH 密钥（如果还没有）
ssh-keygen -t ed25519 -C "your.email@example.com"
# 按 Enter 使用默认路径，可以设置密码或留空

# 查看公钥
cat ~/.ssh/id_ed25519.pub

# 复制公钥内容，添加到 GitHub: Settings -> SSH and GPG keys -> New SSH key
```

或者使用 HTTPS + Personal Access Token（更简单）：

```bash
# GitHub 会提示输入用户名和密码
# 密码使用 Personal Access Token (Settings -> Developer settings -> Personal access tokens)
```

## 完整设置流程

### 步骤 1: 导航到工作目录

```bash
# 导航到你的项目目录（不在 $HOME，使用 lustre）
cd /expanse/lustre/projects/chi157/jji3

# 确认你有写权限
ls -la
```

### 步骤 2: 克隆项目

```bash
# 使用 HTTPS（推荐，更简单）
git clone https://github.com/LucienJi/Oracle-Guided-RL.git

# 或使用 SSH（如果已配置）
# git clone git@github.com:LucienJi/Oracle-Guided-RL.git

cd Oracle-Guided-RL
```

### 步骤 3: 配置路径（重要！）

**选项 A: 使用 setup_paths.sh（推荐，不修改 ~/.bashrc）**

```bash
# 运行路径配置脚本
bash setup_paths.sh

# 脚本会：
# - 自动检测 EXPANSE 集群
# - 提示确认项目路径（默认：/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL）
# - 创建 config/paths_local.yaml
# - 不会修改 ~/.bashrc（符合 HPC 最佳实践）
```

**选项 B: 手动创建 paths_local.yaml（更简单）**

```bash
# 手动创建路径配置文件
cat > config/paths_local.yaml << EOF
# @package _global_
project_root: /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
EOF
```

**确认路径配置：**

```bash
# 检查创建的配置文件
cat config/paths_local.yaml

# 应该看到类似：
# project_root: /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
```

**详细说明**: 查看 [MANUAL_PATH_SETUP.md](MANUAL_PATH_SETUP.md)

### 步骤 4: 加载必要的模块

```bash
# 加载 GPU 和 CUDA 模块
module load gpu
module load cuda/12.2.0  # 或你需要的版本

# 检查可用模块
module avail cuda
module avail gpu

# 查看已加载的模块
module list
```

### 步骤 5: 创建 Conda 环境

```bash
# 确保 conda 已初始化
# 如果还没有，运行：conda init bash，然后 source ~/.bashrc

# 创建环境（这可能需要一些时间）
conda env create -f requirements/oracles.yaml

# 如果遇到问题，可以分步安装：
# conda create -n oracles python=3.10 -y
# conda activate oracles
# conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.4 -y
# conda install -c conda-forge mujoco>=3.3 glfw glew ffmpeg -y
# pip install -r requirements/base.txt
```

### 步骤 6: 激活环境并安装第三方库

```bash
# 激活环境
conda activate oracles

# 安装第三方库
cd third_party/CARL && pip install -e . && cd ../..
cd third_party/Metaworld && pip install -e . && cd ../..
cd third_party/HighwayEnv && pip install -e . && cd ../..
cd third_party/myosuite && pip install -e . && cd ../..

# 返回项目根目录
cd /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
```

### 步骤 7: 设置环境变量（可选，推荐在 SLURM 脚本中设置）

**⚠️ 注意：不推荐修改 ~/.bashrc（HPC 集群最佳实践）**

路径配置已经通过 `config/paths_local.yaml` 完成，环境变量是可选的。

**选项 1: 在 SLURM 脚本中设置（推荐）**

在 SLURM 脚本中直接设置环境变量，见下面的 SLURM 示例。

**选项 2: 在交互式会话中临时设置**

```bash
# 仅在当前会话中有效
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl
```

**选项 3: 创建本地环境文件（如果必须）**

```bash
# 创建 ~/.oracles_env（不修改 ~/.bashrc）
cat > ~/.oracles_env << EOF
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl
EOF

# 使用时加载
source ~/.oracles_env
```

### 步骤 8: 验证安装

```bash
# 激活环境（不依赖 ~/.bashrc）
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oracles

# 设置项目路径（临时，仅用于验证）
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
cd $ORACLES_PROJECT_ROOT

# 验证关键依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mujoco; print(f'MuJoCo: {mujoco.__version__}')"
python -c "import gymnasium; print('Gymnasium OK')"
python -c "import dm_control; print('dm-control OK')"
python -c "import metaworld; print('Metaworld OK')"
python -c "import highway_env; print('HighwayEnv OK')"
python -c "import carl; print('CARL OK')"

# 验证路径配置
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/paths.yaml')
print(f'Project root: {cfg.project_root}')
"
```

### 步骤 9: 测试运行（可选）

```bash
# 运行一个简单的测试
python -c "
import sys
sys.path.insert(0, '.')
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/paths.yaml')
print('✓ Path configuration loaded successfully')
print(f'  Project root: {cfg.project_root}')
"
```

## 创建 SLURM 作业脚本

创建 `submit_job.sh`：

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
module load cuda/12.2.0

# 设置项目路径（从 paths_local.yaml 读取，或手动设置）
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl

# 激活环境（不依赖 ~/.bashrc）
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oracles

# 进入项目目录
cd $ORACLES_PROJECT_ROOT

# 运行训练脚本（替换为你的实际脚本）
python scripts/train_*.py

echo "Job completed at $(date)"
```

**注意：**
- 不需要 `source ~/.bashrc`
- 使用 `conda info --base` 来激活 conda，更可靠
- 环境变量在 SLURM 脚本中设置，不影响系统配置

提交作业：

```bash
chmod +x submit_job.sh
sbatch submit_job.sh

# 查看作业状态
squeue -u $USER

# 查看输出
tail -f slurm-*.out
```

## 快速检查清单

在开始训练前，确认：

- [ ] Git 已配置
- [ ] 项目已克隆到 `/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL`
- [ ] 已运行 `bash setup_paths.sh` 配置路径
- [ ] `config/paths_local.yaml` 存在且路径正确
- [ ] Conda 环境 `oracles` 已创建
- [ ] 所有第三方库已安装
- [ ] 环境变量已设置（`ORACLES_PROJECT_ROOT`, `MUJOCO_GL`）
- [ ] 所有依赖验证通过
- [ ] 模块已加载（gpu, cuda）

## 常见问题

### Git 认证问题

如果遇到认证问题：

```bash
# 使用 Personal Access Token
# GitHub -> Settings -> Developer settings -> Personal access tokens -> Generate new token
# 复制 token，在 git clone 时作为密码使用
```

### Conda 环境创建失败

```bash
# 检查磁盘空间
df -h /expanse/lustre/projects/chi157/jji3

# 清理 conda 缓存
conda clean -a

# 分步安装（见步骤 5）
```

### 路径配置不工作

```bash
# 检查配置文件
cat config/paths_local.yaml

# 检查环境变量
echo $ORACLES_PROJECT_ROOT

# 手动设置
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
```

### CUDA 版本不匹配

```bash
# 检查可用 CUDA 版本
module avail cuda

# 修改 requirements/oracles.yaml 中的 pytorch-cuda 版本
# 或使用 pip 安装对应版本
```

## 下一步

设置完成后：

1. 准备数据/检查点（如果需要）
2. 修改配置文件（如果需要）
3. 创建 SLURM 作业脚本
4. 提交训练任务

## 有用的命令

```bash
# 查看作业
squeue -u $USER

# 取消作业
scancel <job_id>

# 查看作业详情
scontrol show job <job_id>

# 查看模块
module avail
module list

# 查看 GPU
nvidia-smi  # 在计算节点上

# 查看磁盘使用
du -sh /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
```

