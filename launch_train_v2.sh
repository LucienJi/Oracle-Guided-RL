#!/usr/bin/env bash
set -euo pipefail

# ================= INPUTS =================
MODULE_NAME="$1"        # e.g., train_CurrimaxAdv
SEGMENTS="$2"           # 现在这个参数生效了！例如：3
SEEDS_LIST="$3"         # 传入一串种子，例如 "42 43 44"
shift 3
PYTHON_ARGS_BASE=("$@") # 其他参数

# ================= PATHS =================
BASE="/expanse/lustre/projects/chi157/jji3"
REPO_DIR="${BASE}/Oracle-Guided-RL"
LOG_DIR="${REPO_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

# ================= 资源计算 =================
IFS=' ' read -r -a SEED_ARRAY <<< "$SEEDS_LIST"
NUM_TASKS=${#SEED_ARRAY[@]}

# 动态计算 CPU 和 内存
# 如果 packing 3个任务，就拿满 10 个 CPU
if [ "$NUM_TASKS" -ge 3 ]; then
    TOTAL_CPUS=10
    TOTAL_MEM="80G"
else
    TOTAL_CPUS=$(( 4 * NUM_TASKS ))
    TOTAL_MEM="$(( 20 * NUM_TASKS ))G"
fi

# 基础设置
JOB_NAME="${MODULE_NAME}_pack"
ACCOUNT="${ACCOUNT:-chi157}"
PARTITION="${PARTITION:-gpu-shared}"
TIME="${TIME:-01:00:00}" # 建议设置为 4 小时或 10 小时，根据你的 Checkpoint 频率
CONSTRAINTS="${CONSTRAINTS:-lustre}"

echo "=========================================================="
echo "🚀 Hybrid Launch Strategy:"
echo "   ➡️  Parallel: Running ${NUM_TASKS} seeds together (${SEED_ARRAY[*]})"
echo "   ⬇️  Sequential: Chaining ${SEGMENTS} segments"
echo "   💻 Resources: 1 GPU, ${TOTAL_CPUS} CPUs, ${TOTAL_MEM} Mem"
echo "=========================================================="

# ================= 提交循环 (Segments Loop) =================
PREV_JOB=""

# Serialize Python arguments for safe passing through heredoc
# Use printf to properly quote each argument
PYTHON_ARGS_QUOTED=""
for arg in "${PYTHON_ARGS_BASE[@]}"; do
  PYTHON_ARGS_QUOTED="${PYTHON_ARGS_QUOTED}$(printf '%q ' "$arg")"
done
# Remove trailing space
PYTHON_ARGS_QUOTED="${PYTHON_ARGS_QUOTED% }"

for i in $(seq 1 "$SEGMENTS"); do
  echo "▶️  Submitting segment $i / $SEGMENTS..."
  
  # 构建 sbatch 命令
  SBATCH_CMD=(sbatch --parsable)
  if [[ -n "$PREV_JOB" ]]; then
    # 设置依赖：上一个 Job 结束后（无论成功失败）开始这一个
    # 这样确保你的代码能 load 之前的 checkpoint
    SBATCH_CMD+=(--dependency=afterany:"$PREV_JOB")
  fi

  # 生成并提交 Job 脚本
  SBATCH_OUTPUT=$("${SBATCH_CMD[@]}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_s${i}
#SBATCH --output=${LOG_DIR}/%x_seg${i}_%j.out
#SBATCH --error=${LOG_DIR}/%x_seg${i}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=${TOTAL_CPUS}
#SBATCH --mem=${TOTAL_MEM}
#SBATCH -t ${TIME}
#SBATCH -A ${ACCOUNT}
#SBATCH --constraint=${CONSTRAINTS}
#SBATCH --no-requeue

set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load sdsc gpu
cd ${REPO_DIR}
source ${BASE}/conda/miniconda3/etc/profile.d/conda.sh
conda activate oracles

export MUJOCO_GL="egl"
export EGL_PLATFORM="surfaceless"
unset DISPLAY
export TMPDIR="/scratch/\$USER/job_\$SLURM_JOB_ID"
mkdir -p "\$TMPDIR"

# 告诉 Hydra 或代码这是第几段（可选）
export SEGMENT_INDEX=${i}

echo "Starting Packed Segment ${i}..."

# --- 核心：并行启动所有 Seeds ---
pids=""
for seed in ${SEEDS_LIST}; do
  # 日志文件名带上 Segment 编号，方便调试
  TASK_LOG="${LOG_DIR}/${MODULE_NAME}_s${i}_seed\${seed}_\${SLURM_JOB_ID}.log"
  
  echo "   -> [Seg ${i}] Launching seed \$seed (Log: \$TASK_LOG)"
  
  # 后台运行 Python
  # 关键：你的 Python 代码必须具备 "如果 checkpoint 存在则自动 resume" 的功能
  # Reconstruct arguments array from quoted string
  eval "python -m scripts.${MODULE_NAME} ${PYTHON_ARGS_QUOTED} seed=\$seed" > "\$TASK_LOG" 2>&1 &
  
  pids="\$pids \$!"
done

echo "⏳ Waiting for pids: \$pids"
wait
echo "✅ Segment ${i} finished."

EOF
  )

  # 检查提交结果
  JOBID="${SBATCH_OUTPUT%%;*}"
  if ! [[ "$JOBID" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: Failed to parse JobID: $SBATCH_OUTPUT"
    exit 1
  fi
  
  echo "  ✅ JobID=$JOBID (Dependency: ${PREV_JOB:-None})"
  PREV_JOB=$JOBID

done

echo "🎉 All segments submitted!"