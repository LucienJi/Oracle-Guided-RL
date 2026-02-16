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

# 💡 6 seeds total: 2 jobs × 3 seeds each
SEEDS_GROUP1="42 43 44"
SEEDS_GROUP2="45 46 47"

for task in "${tasks[@]}"; do
    method_suffix="simba_${task}"
    
    echo "📦 Submitting Packed Job 1/2 (seeds $SEEDS_GROUP1) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "train_simba" "2" "$SEEDS_GROUP1" \
        --config-name "dmc/simba_${task}" \
        method_name="${method_suffix}_expanse_debug"
    sleep 1
    echo "📦 Submitting Packed Job 2/2 (seeds $SEEDS_GROUP2) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "train_simba" "2" "$SEEDS_GROUP2" \
        --config-name "dmc/simba_${task}" \
        method_name="${method_suffix}_expanse_debug"
    sleep 1
done

echo "🎉 Batch submission complete."
squeue -u "$(whoami)"