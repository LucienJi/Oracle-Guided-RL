#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_train_v2.sh"

export ACCOUNT="chi157"
export PARTITION="gpu-shared"
export TIME="1:00:00" 

tasks=(
    humanoid
    # acrobot_sparse
)

# 💡 定义你想一起跑的种子组
# 这里我们将 42, 43, 44 打包成一组
SEEDS_GROUP="42 43 44"

for task in "${tasks[@]}"; do
    method_suffix="simba_${task}"
    
    echo "📦 Submitting Packed Job for task: $task"
    # 调用 V2 launcher
    # 注意参数顺序：模块名 -> 段数(1) -> 种子列表 -> Python参数
    bash "$LAUNCH_SCRIPT" \
        "train_simba" "2" "$SEEDS_GROUP" \
        --config-name "dmc/simba_${task}" \
        method_name="${method_suffix}_expanse_debug"
    sleep 1
done

echo "🎉 Batch submission complete."
squeue -u "$(whoami)"