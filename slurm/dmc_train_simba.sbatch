#!/bin/bash
# Representative Slurm script for the DMC SIMBA submission path.
# Edit the placeholder resource lines before submitting on your cluster.

#SBATCH --job-name=dmc_simba_submission
#SBATCH --partition=REPLACE_PARTITION
#SBATCH --account=REPLACE_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_INIT="/path/to/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="oracles"

cd "$REPO_ROOT"

# Update CONDA_INIT above for your cluster before running this script.
source "$CONDA_INIT"
conda activate "$CONDA_ENV"

export ORACLES_PROJECT_ROOT="$REPO_ROOT"

python -m scripts.train_simba \
  --config-name dmc/submission_smoke_cartpole \
  training.total_timesteps=1000 \
  training.eval_every=250 \
  training.save_freq=250 \
  training.use_wandb=false \
  training.save_video=false \
  training.resume=false \
  training.use_compile=false

# Monitor with:
#   squeue -u "$USER"
#   sacct -j <job_id> --format=JobID,JobName,State,Elapsed
