# Oracle-Guided-RL

Oracle-Guided Reinforcement Learning 项目

## 快速开始

### 在另一台服务器上安装环境

#### 方法 1: 使用自动安装脚本（推荐）

```bash
git clone https://github.com/LucienJi/Oracle-Guided-RL.git
cd Oracle-Guided-RL
bash install.sh
bash setup_paths.sh  # 配置路径（重要！）
```

#### 方法 2: 手动安装

详细步骤请查看 [INSTALL.md](INSTALL.md)

**简要步骤：**
1. 克隆仓库
2. **运行路径配置**: `bash setup_paths.sh` ⚠️ **重要！**
3. 创建 conda 环境：`conda env create -f requirements/oracles.yaml`
4. 激活环境：`conda activate oracles`
5. 安装第三方库（CARL, Metaworld, HighwayEnv, MyoSuite）
6. 验证安装

## 路径配置（重要！）

项目使用统一的路径管理系统，避免硬编码绝对路径。

**首次使用时必须运行：**

```bash
bash setup_paths.sh
```

这会创建 `config/paths_local.yaml`（gitignored），定义你的项目根目录。所有配置文件会自动使用 `${paths.project_root}` 来引用路径。

**更多信息：** 查看 [INSTALL.md](INSTALL.md#路径配置)

## HPC 集群安装

### EXPANSE HPC Cluster

详细的 EXPANSE 安装指南请查看 [INSTALL_EXPANSE.md](INSTALL_EXPANSE.md)

## 项目结构

```
Oracle-Guided-RL/
├── algo/              # 算法实现
├── config/            # 配置文件
│   ├── paths.yaml     # 全局路径配置
│   └── paths_local.yaml  # 本地路径覆盖（gitignored，由 setup_paths.sh 创建）
├── data_buffer/       # 数据缓冲区
├── env/               # 环境封装
├── eval/              # 评估结果（gitignore）
├── model/             # 模型定义
├── requirements/      # 依赖文件
│   ├── base.txt       # Python 依赖
│   └── oracles.yaml   # Conda 环境配置
├── scripts/           # 训练脚本
├── third_party/       # 第三方库
├── install.sh         # 自动安装脚本
└── setup_paths.sh     # 路径配置脚本
```

## 使用说明

激活环境后，运行训练脚本：

```bash
conda activate oracles
python scripts/train_*.py
```

## 依赖

- Python 3.10
- PyTorch (CUDA 12.4)
- MuJoCo >= 3.3
- Gymnasium
- DeepMind Control Suite
- Metaworld
- HighwayEnv
- CARL

完整依赖列表见 `requirements/base.txt`

## 注意事项

- `eval/`, `outputs/`, `checkpoints/` 等目录在 .gitignore 中，不会从 GitHub 克隆
- 如需数据或模型，需要从其他服务器单独传输
- 确保服务器有 CUDA 12.4 兼容的 GPU
- **重要**：在新集群上首次使用时，务必运行 `bash setup_paths.sh` 配置路径

## 更多信息

- 详细安装说明: [INSTALL.md](INSTALL.md)
- EXPANSE HPC 安装: [INSTALL_EXPANSE.md](INSTALL_EXPANSE.md)
