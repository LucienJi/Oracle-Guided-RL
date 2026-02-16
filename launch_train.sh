#!/usr/bin/env bash
set -euo pipefail

MODULE_NAME="$1"     # e.g., train_CurrimaxAdv
SEGMENTS="$2"        # 2 or 3
shift 2
PYTHON_ARGS=("$@")

# --- Paths on Expanse Lustre ---
BASE="/expanse/lustre/projects/chi157/jji3"
REPO_DIR="${BASE}/Oracle-Guided-RL"

# Logs/outputs
LOG_DIR="${REPO_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

# Build a shell-escaped python command line
PYTHON_CMD_LINE="$(printf '%q ' python -m "scripts.${MODULE_NAME}" "${PYTHON_ARGS[@]}")"

# ============ Defaults tailored for Expanse ============
JOB_NAME="${MODULE_NAME}_seq"
ACCOUNT="${ACCOUNT:-chi157}"

# For many small jobs, gpu-shared is usually what you want
PARTITION="${PARTITION:-gpu-shared}"

# Resource request
GPUS="${GPUS:-1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-24G}"              
TIME="${TIME:-00:30:00}"     #<----- we can change it   

# Lustre requirement (since code/ckpt live on /expanse/lustre/...)
CONSTRAINTS="${CONSTRAINTS:-lustre}"

# Modules typically needed on Expanse GPU nodes
MODULES=(sdsc gpu)

# Conda init (your Lustre-installed conda)
CONDA_INIT="${CONDA_INIT:-${BASE}/conda/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-oracles}"

# MuJoCo headless GPU defaults (safe even if you don't render)
MUJOCO_GL_DEFAULT="${MUJOCO_GL_DEFAULT:-egl}"
EGL_PLATFORM_DEFAULT="${EGL_PLATFORM_DEFAULT:-surfaceless}"

# ============ Submit sequential jobs ============
PREV_JOB=""

for i in $(seq 1 "$SEGMENTS"); do
  echo "▶️  Submitting segment $i / $SEGMENTS..."

  SBATCH_CMD=(sbatch --parsable)
  if [[ -n "$PREV_JOB" ]]; then
    SBATCH_CMD+=(--dependency=afterany:"$PREV_JOB")
  fi

  SBATCH_OUTPUT=$("${SBATCH_CMD[@]}" <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=${GPUS}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH -t ${TIME}
#SBATCH -A ${ACCOUNT}
#SBATCH --constraint=${CONSTRAINTS}
#SBATCH --no-requeue

set -euo pipefail

# Ensure module command exists in non-interactive batch shells
source /etc/profile.d/modules.sh

module purge
$(for m in "${MODULES[@]}"; do echo "module load $m"; done)

cd ${REPO_DIR}

# Conda
source ${CONDA_INIT}
conda activate ${CONDA_ENV}

# Prefer conda libs first (avoid module LD_LIBRARY_PATH collisions)
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"

# MuJoCo headless GPU (prevents GLFW/X11 issues on compute nodes)
export MUJOCO_GL="${MUJOCO_GL_DEFAULT}"
export EGL_PLATFORM="${EGL_PLATFORM_DEFAULT}"
unset DISPLAY

# Node-local scratch for tmp (fast, avoids Lustre tiny-file churn)
export TMPDIR="/scratch/\$USER/job_\$SLURM_JOB_ID"
mkdir -p "\$TMPDIR"

# Optional: tag segment index into Hydra/your code if you want
export SEGMENT_INDEX=${i}
export SEGMENT_COUNT=${SEGMENTS}

# Run training
${PYTHON_CMD_LINE}
EOF
  )

  JOBID="${SBATCH_OUTPUT%%;*}"
  if ! [[ "$JOBID" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: Failed to parse sbatch JobID from output: $SBATCH_OUTPUT"
    exit 1
  fi

  echo "  ✅ Submitted JobID=$JOBID"
  PREV_JOB=$JOBID
done

echo "🎉 All $SEGMENTS segments submitted!"
