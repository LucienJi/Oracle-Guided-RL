#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 🔧 Expanse 常用设置 (全局覆盖)
# ==========================================
# 这些变量会被 launch_train.sh 读取
# 如果你想用 launcher 里的默认值，注释掉这些行即可
export ACCOUNT="chi157"           # 你的账户
export PARTITION="gpu-shared"     # 分区
export TIME="04:00:00"            # 修改这里统一控制所有任务的时间
export GPUS=1
export CPUS=4
export MEM="24G"

# ==========================================
# 📂 路径设置
# ==========================================
# 获取当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 假设 launch_train.sh 在同一目录下
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_train.sh"

# 检查 launcher 是否存在
if [[ ! -f "$LAUNCH_SCRIPT" ]]; then
    echo "❌ Error: Cannot find $LAUNCH_SCRIPT at $SCRIPT_DIR"
    exit 1
fi

echo "🚀 Starting batch experiment submission on Expanse..."
echo "📍 Using launcher: $LAUNCH_SCRIPT"
echo "⏱️  Time limit per job: $TIME"
echo

# 计数器
EXPERIMENT_COUNT=0
TOTAL_JOBS=0

# ==========================================
# 🛠️ 提交函数
# ==========================================
# Usage: submit_experiment <exp_name> <module> <segments> [python args...]
submit_experiment() {
    local exp_name="$1"
    local module="$2"
    local segments="$3"
    shift 3
    local python_args=("$@")
    
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🔬 Experiment #$EXPERIMENT_COUNT: $exp_name"
    echo "📦 Module: $module"
    echo "📊 Segments: $segments"
    echo "⚙️  Python args: ${python_args[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # 关键修改：直接使用 bash 调用，而不是 sbatch
    # 参数顺序必须匹配 launch_train.sh: $1=MODULE, $2=SEGMENTS, $@=ARGS
    if bash "$LAUNCH_SCRIPT" "$module" "$segments" "${python_args[@]}"; then
        echo "✅ Experiment $exp_name submitted successfully!"
        TOTAL_JOBS=$((TOTAL_JOBS + segments))
    else
        echo "❌ Experiment $exp_name submission failed!"
        return 1
    fi
    
    # 稍微暂停一下，避免在此处瞬间轰炸 Slurm 调度器（可选）
    sleep 1
    echo
}

# ==========================================
# 🧪 任务列表 (MetaWorld)
# ==========================================
tasks=(
    assembly
    # basketball
    # drawer-open
    # hammer
    # lever-pull
    # peg-insert-side
    peg-unplug-side
    # pick-place
    push-wall
    # stick-pull
)

# ==========================================
# 🔄 循环提交
# ==========================================
for task in "${tasks[@]}"; do
    # 自动将 task 名称转换为 CamelCase (例如 peg-insert-side -> PegInsertSide)
    method_suffix=$(echo "$task" | awk -F- '{for (i=1; i<=NF; i++) {printf toupper(substr($i,1,1)) substr($i,2)}}')
    
    # 你可以在这里修改 seeds 列表，例如：for seed in 42 43 44; do
    for seed in 42; do
        # 调用提交函数
        # 注意：这里调用的是 train_CurrimaxAdv，如果你要跑 simba，记得改成 train_simba
        submit_experiment "CurrimaxAdv_${task}${seed}" \
            "train_CurrimaxAdv" "3" \
            --config-name "metaworld/CurrimaxAdv_${task}" \
            method_name="CurrimaxAdv_${method_suffix}" \
            seed=$seed
    done
done

# ==========================================
# 📦 Box2D Experiments (注释状态)
# ==========================================
# box2d_configs=(
#     bipedal_learner
#     lander_heavy
#     racingcar_learner
#     weather_learner
# )

# for config in "${box2d_configs[@]}"; do
#     for seed in 42; do
#         submit_experiment "CurrimaxAdv_${config}_seed${seed}" \
#             "train_CurrimaxAdv_simba" "3" \
#             --config-name "box2d/CurrimaxAdv_${config}" \
#             seed=$seed
#     done
# done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Batch submission complete!"
echo "📈 Total experiments: $EXPERIMENT_COUNT"
echo "🔢 Total jobs created (segments): $TOTAL_JOBS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 显示队列状态
echo "📊 Current queue status for user $(whoami):"
squeue -u "$(whoami)" --format="%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R" || echo "Unable to get queue information"
