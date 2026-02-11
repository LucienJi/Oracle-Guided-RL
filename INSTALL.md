# Oracle-Guided-RL 环境安装指南

本指南将帮助你在新的服务器上安装和配置 Oracle-Guided-RL 项目的完整环境。

## 前置要求

- Linux 系统（推荐 Ubuntu 18.04+）
- CUDA 12.4 兼容的 GPU（用于 PyTorch）
- Conda 或 Miniconda 已安装

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/LucienJi/Oracle-Guided-RL.git
cd Oracle-Guided-RL
```

### 2. 创建 Conda 环境

项目使用 Conda 环境管理，推荐使用提供的 `oracles.yaml` 文件：

**重要**: 确保在项目根目录下运行此命令：

```bash
cd Oracle-Guided-RL
conda env create -f requirements/oracles.yaml
```

如果遇到路径问题，可以手动安装：

```bash
conda create -n oracles python=3.10 -y
conda activate oracles
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.4 mujoco>=3.3 glfw glew ffmpeg -y
pip install -r requirements/base.txt
```

这将创建一个名为 `oracles` 的 conda 环境，包含：
- Python 3.10
- PyTorch with CUDA 12.4
- MuJoCo >= 3.3
- 其他系统依赖（glfw, glew, ffmpeg）

### 3. 激活环境

```bash
conda activate oracles
```

### 4. 安装第三方库

项目包含几个第三方库需要安装：

#### 4.1 安装 CARL

```bash
cd third_party/CARL
pip install -e .
cd ../..
```

#### 4.2 安装 Metaworld

```bash
cd third_party/Metaworld
pip install -e .
cd ../..
```

#### 4.3 安装 HighwayEnv

```bash
cd third_party/HighwayEnv
pip install -e .
cd ../..
```

#### 4.4 安装 MyoSuite（如果使用）

```bash
cd third_party/myosuite
pip install -e .
cd ../..
```

### 5. 验证安装

运行以下命令验证关键依赖是否正确安装：

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import dm_control; print('dm-control installed')"
python -c "import metaworld; print('Metaworld installed')"
python -c "import highway_env; print('HighwayEnv installed')"
python -c "import carl; print('CARL installed')"
```

### 6. 设置环境变量（可选）

如果需要使用 EGL 渲染（无头服务器），设置：

```bash
export MUJOCO_GL=egl
```

或者添加到 `~/.bashrc`：

```bash
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
source ~/.bashrc
```

## 替代安装方法（使用 pip）

如果不想使用 Conda，也可以使用 pip：

### 1. 创建虚拟环境

```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装 PyTorch（根据你的 CUDA 版本）

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 或 CPU 版本
pip install torch torchvision
```

### 3. 安装 MuJoCo

```bash
pip install mujoco
```

### 4. 安装其他依赖

```bash
pip install -r requirements/base.txt
```

### 5. 安装第三方库（同上面的步骤 4）

## 常见问题

### CUDA 版本不匹配

如果遇到 CUDA 版本问题，可以修改 `requirements/oracles.yaml` 中的 `pytorch-cuda` 版本，或使用 pip 安装对应版本的 PyTorch。

### MuJoCo 安装问题

确保系统已安装必要的图形库：
```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglfw3 libglew2.1
```

### 第三方库导入错误

确保在项目根目录下运行代码，或添加项目路径到 PYTHONPATH：
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Oracle-Guided-RL"
```

## 测试安装

运行一个简单的测试脚本验证环境：

```bash
python -c "
import sys
sys.path.insert(0, '.')
from env.env_utils import make_env_and_eval_env_from_cfg
from omegaconf import OmegaConf
print('Environment setup successful!')
"
```

## 路径配置

项目使用统一的路径管理系统，避免硬编码绝对路径。

### 首次设置

运行路径配置脚本：

```bash
bash setup_paths.sh
```

这会创建 `config/paths_local.yaml`（gitignored），定义你的项目根目录。

### 在配置文件中使用路径

所有配置文件现在使用 `${paths.project_root}` 来引用项目根目录：

```yaml
oracle_0: '${paths.project_root}/checkpoints/...'
```

### 环境变量方式

也可以设置环境变量：

```bash
export ORACLES_PROJECT_ROOT=/path/to/Oracle-Guided-RL
```

### 批量更新现有配置

如果你有旧的配置文件包含绝对路径，可以批量更新：

```bash
python scripts/update_config_paths.py
```

这会将所有 `/share/data/ripl/jjt/projects/oracles` 替换为 `${paths.project_root}`。

## HPC 集群特定指南

### EXPANSE HPC Cluster

详细的 EXPANSE 安装指南请查看 [INSTALL_EXPANSE.md](INSTALL_EXPANSE.md)

## 下一步

安装完成后，你可以：
1. 查看配置文件：`config/base_configs/`
2. 运行训练脚本：`scripts/train_*.py`
3. 查看启动脚本：`launch_*.sh`（注意：这些脚本在 .gitignore 中，需要手动创建或从其他服务器复制）

## 注意事项

- 项目中的 `eval/`、`outputs/`、`checkpoints/` 等目录在 .gitignore 中，不会从 GitHub 克隆
- 如果需要数据或模型检查点，需要从其他服务器单独传输
- 确保有足够的磁盘空间（建议至少 50GB）
- **重要**：在新集群上首次使用时，务必运行 `bash setup_paths.sh` 配置路径

