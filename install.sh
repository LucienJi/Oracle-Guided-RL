#!/bin/bash
# Oracle-Guided-RL 快速安装脚本
# 使用方法: bash install.sh

set -e  # 遇到错误立即退出

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

echo ""
echo "步骤 1/5: 创建 Conda 环境..."
echo "----------------------------------------"
conda env create -f requirements/oracles.yaml

echo ""
echo "步骤 2/5: 激活环境并安装第三方库..."
echo "----------------------------------------"

# 激活 conda 环境（在脚本中需要使用 conda run）
echo "安装 CARL..."
cd third_party/CARL
conda run -n oracles pip install -e . > /dev/null 2>&1 || {
    echo "警告: CARL 安装可能失败，请手动检查"
}
cd "$SCRIPT_DIR"

echo "安装 Metaworld..."
cd third_party/Metaworld
conda run -n oracles pip install -e . > /dev/null 2>&1 || {
    echo "警告: Metaworld 安装可能失败，请手动检查"
}
cd "$SCRIPT_DIR"

echo "安装 HighwayEnv..."
cd third_party/HighwayEnv
conda run -n oracles pip install -e . > /dev/null 2>&1 || {
    echo "警告: HighwayEnv 安装可能失败，请手动检查"
}
cd "$SCRIPT_DIR"

echo "安装 MyoSuite..."
cd third_party/myosuite
conda run -n oracles pip install -e . > /dev/null 2>&1 || {
    echo "警告: MyoSuite 安装可能失败，请手动检查"
}
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
echo "步骤 4/5: 设置环境变量..."
echo "----------------------------------------"

# 检查是否已设置 MUJOCO_GL
if ! grep -q "MUJOCO_GL" ~/.bashrc 2>/dev/null; then
    echo 'export MUJOCO_GL=egl' >> ~/.bashrc
    echo "✓ 已添加 MUJOCO_GL=egl 到 ~/.bashrc"
else
    echo "✓ MUJOCO_GL 已在 ~/.bashrc 中设置"
fi

echo ""
echo "步骤 5/5: 安装完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 激活环境: conda activate oracles"
echo "  2. 进入项目目录: cd $SCRIPT_DIR"
echo "  3. 运行训练脚本: python scripts/train_*.py"
echo ""
echo "如果遇到问题，请查看 INSTALL.md 获取详细说明。"
echo ""

