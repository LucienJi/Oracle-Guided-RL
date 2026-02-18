#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_train_v2.sh"

export ACCOUNT="chi157"
export PARTITION="gpu-shared"
export TIME="9:00:00"
export JOB_NAME_PREFIX="MAPS"

# Optional args: --total_timesteps=500000 --capacity=500000 --exclude=node1,node2
TOTAL_TIMESTEPS=500000
CAPACITY=500000
EXCLUDE_OPT=(--exclude=exp-9-60)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --total_timesteps=*) TOTAL_TIMESTEPS="${1#*=}" ;;
    --capacity=*)       CAPACITY="${1#*=}" ;;
    --exclude=*)        EXCLUDE_OPT=("$1") ;;
    *)                  echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# MetaWorld tasks from config/oracles/metaworld/
tasks=(
    # assembly
    # basketball
    # drawer-open
    hammer
    lever-pull
    peg-insert-side
    # peg-unplug-side
    # pick-place
    # push-wall
    # stick-pull
)

# Tasks that require env.sparse_reward=true
SPARSE_TASKS="hammer assembly drawer-open pick-place push-wall"

# 6 seeds total: 2 jobs × 3 seeds each
SEEDS_GROUP1="42 43 44"
SEEDS_GROUP2="45 46 47"
SEGMENTS="1"

for task in "${tasks[@]}"; do
    EXTRA_ARGS=("training.total_timesteps=${TOTAL_TIMESTEPS}" "training.capacity=${CAPACITY}")
    [[ " $SPARSE_TASKS " == *" $task "* ]] && EXTRA_ARGS+=(env.sparse_reward=true)

    echo "Submitting MAPS job 1/2 (seeds $SEEDS_GROUP1) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "baselines.train_maps" "$SEGMENTS" "$SEEDS_GROUP1" \
        --config-name "baselines_configs/maps/metaworld/${task}" \
        "${EXCLUDE_OPT[@]}" "${EXTRA_ARGS[@]}"
    sleep 1
    echo "Submitting MAPS job 2/2 (seeds $SEEDS_GROUP2) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "baselines.train_maps" "$SEGMENTS" "$SEEDS_GROUP2" \
        --config-name "baselines_configs/maps/metaworld/${task}" \
        "${EXCLUDE_OPT[@]}" "${EXTRA_ARGS[@]}"
    sleep 1
done

echo "Batch submission complete (MAPS)."
squeue -u "$(whoami)"
