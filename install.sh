#!/bin/bash
# Oracle-Guided-RL 快速安装脚本
# 使用方法: bash install.sh

set -euo pipefail

echo "=========================================="
echo "Oracle-Guided-RL 环境安装脚本"
echo "=========================================="

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 conda，请先安装 Miniconda 或 Anaconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

require_dir() {
    local path="$1"
    if [[ ! -d "$path" ]]; then
        echo "错误: 缺少依赖目录 $path"
        echo "当前仓库无法从干净 clone 独立复现这些 third_party 依赖。"
        echo "最小修复建议: 将这些依赖改为 git submodule 或提供固定 commit 的 bootstrap 脚本。"
        exit 1
    fi
}

echo ""
echo "步骤 1/5: 创建 Conda 环境..."
echo "----------------------------------------"
conda env create -f environment.yml

echo ""
echo "步骤 2/5: 检查并安装第三方库..."
echo "----------------------------------------"

require_dir "third_party/CARL"
require_dir "third_party/Metaworld"
require_dir "third_party/HighwayEnv"
require_dir "third_party/myosuite"

# 激活 conda 环境（在脚本中需要使用 conda run）
echo "安装 CARL..."
cd third_party/CARL
conda run -n oracles pip install -e .
cd "$SCRIPT_DIR"

echo "安装 Metaworld..."
cd third_party/Metaworld
conda run -n oracles pip install -e .
cd "$SCRIPT_DIR"

echo "安装 HighwayEnv..."
cd third_party/HighwayEnv
conda run -n oracles pip install -e .
cd "$SCRIPT_DIR"

echo "安装 MyoSuite..."
cd third_party/myosuite
conda run -n oracles pip install -e .
cd "$SCRIPT_DIR"

echo ""
echo "步骤 3/5: 验证安装..."
echo "----------------------------------------"

# 验证关键依赖
conda run -n oracles python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" || echo "✗ PyTorch 未正确安装"
conda run -n oracles python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" || echo "✗ CUDA 检查失败"
conda run -n oracles python -c "import gymnasium; print(f'✓ Gymnasium: {gymnasium.__version__}')" || echo "✗ Gymnasium 未正确安装"
conda run -n oracles python -c "import dm_control; print('✓ dm-control installed')" || echo "✗ dm-control 未正确安装"
conda run -n oracles python -c "import metaworld; print('✓ Metaworld installed')" || echo "✗ Metaworld 未正确安装"
conda run -n oracles python -c "import highway_env; print('✓ HighwayEnv installed')" || echo "✗ HighwayEnv 未正确安装"
conda run -n oracles python -c "import carl; print('✓ CARL installed')" || echo "✗ CARL 未正确安装"

echo ""
echo "步骤 4/5: 环境变量提示..."
echo "----------------------------------------"
echo "未修改 ~/.bashrc。"
echo "如需无头渲染，请在当前 shell 或作业脚本中显式设置："
echo "  export MUJOCO_GL=egl"

echo ""
echo "步骤 5/5: 安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 激活环境: conda activate oracles"
echo "  2. 进入项目目录: cd $SCRIPT_DIR"
echo "  3. 查看 README.md 获取唯一的 setup / quick start 路径"
echo ""
echo "如果遇到问题，请查看 INSTALL.md 获取详细说明。"
echo ""
