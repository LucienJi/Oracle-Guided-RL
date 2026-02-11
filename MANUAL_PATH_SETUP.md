# 手动路径配置指南（不修改 ~/.bashrc）

如果你不想运行 `setup_paths.sh` 或不想修改 `~/.bashrc`，可以手动配置路径。

## 方法 1: 手动创建 paths_local.yaml（推荐）

这是最简单且推荐的方法，不需要修改任何系统文件。

```bash
# 在项目根目录下
cd /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL

# 创建路径配置文件
cat > config/paths_local.yaml << EOF
# @package _global_
# Local path configuration (gitignored)
project_root: /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
EOF
```

验证：

```bash
cat config/paths_local.yaml
```

## 方法 2: 使用环境变量（在 SLURM 脚本中）

不需要创建 `paths_local.yaml`，直接在 SLURM 脚本中设置环境变量：

```bash
#!/bin/bash
#SBATCH ...

# 设置环境变量
export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl

# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oracles

# 运行脚本
cd $ORACLES_PROJECT_ROOT
python scripts/train_*.py
```

## 方法 3: 使用示例文件

```bash
# 复制示例文件
cp config/paths_local.yaml.example config/paths_local.yaml

# 编辑文件
nano config/paths_local.yaml  # 或使用 vim/emacs

# 修改 project_root 为你的实际路径
```

## 路径优先级

Hydra 配置系统按以下优先级查找 `project_root`：

1. **`config/paths_local.yaml`** （最高优先级，推荐）
2. 环境变量 `ORACLES_PROJECT_ROOT`
3. 当前工作目录（fallback）

## 验证配置

```bash
# 激活环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oracles

# 测试路径配置
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('config/paths.yaml')
print(f'Project root: {cfg.project_root}')
"
```

应该输出你的项目路径。

## 在 SLURM 脚本中使用

无论使用哪种方法，在 SLURM 脚本中都不需要修改 `~/.bashrc`：

```bash
#!/bin/bash
#SBATCH --job-name=oracle_train
#SBATCH --partition=gpu-shared
#SBATCH --gres=gpu:1
# ... 其他 SLURM 参数

# 加载模块
module load gpu
module load cuda/12.2.0

# 设置环境变量（如果使用方法 2）
# export ORACLES_PROJECT_ROOT=/expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL
export MUJOCO_GL=egl

# 激活 conda（不依赖 ~/.bashrc）
source $(conda info --base)/etc/profile.d/conda.sh
conda activate oracles

# 进入项目目录
cd /expanse/lustre/projects/chi157/jji3/Oracle-Guided-RL

# 运行训练
python scripts/train_*.py
```

## 总结

- ✅ **推荐**: 创建 `config/paths_local.yaml`（方法 1）
- ✅ **可选**: 在 SLURM 脚本中设置环境变量（方法 2）
- ❌ **不推荐**: 修改 `~/.bashrc`（HPC 集群最佳实践）

