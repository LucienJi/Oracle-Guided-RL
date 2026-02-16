#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_train_v2.sh"

export ACCOUNT="chi157"
export PARTITION="gpu-shared"
export TIME="1:00:00"
export JOB_NAME_PREFIX="MAPS"

# MetaWorld tasks from config/oracles/metaworld/
tasks=(
    assembly
    # basketball
    # drawer-open
    # hammer
    # lever-pull
    # peg-insert-side
    # peg-unplug-side
    # pick-place
    # push-wall
    # stick-pull
)

# Tasks that require env.sparse_reward=true
SPARSE_TASKS="hammer assembly drawer-open pick-place push-wall"

# 6 seeds total: 3 jobs × 2 seeds each
SEEDS_GROUP1="42 43"
SEEDS_GROUP2="44 45"
SEEDS_GROUP3="46 47"
SEGMENTS="1"

for task in "${tasks[@]}"; do
    EXTRA_ARGS=()
    [[ " $SPARSE_TASKS " == *" $task "* ]] && EXTRA_ARGS+=(env.sparse_reward=true)

    echo "Submitting MAPS job 1/3 (seeds $SEEDS_GROUP1) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "baselines.train_maps" "$SEGMENTS" "$SEEDS_GROUP1" \
        --config-name "baselines_configs/maps/metaworld/${task}" \
        "${EXTRA_ARGS[@]}"
    sleep 1
    echo "Submitting MAPS job 2/3 (seeds $SEEDS_GROUP2) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "baselines.train_maps" "$SEGMENTS" "$SEEDS_GROUP2" \
        --config-name "baselines_configs/maps/metaworld/${task}" \
        "${EXTRA_ARGS[@]}"
    sleep 1
    echo "Submitting MAPS job 3/3 (seeds $SEEDS_GROUP3) for task: $task"
    bash "$LAUNCH_SCRIPT" \
        "baselines.train_maps" "$SEGMENTS" "$SEEDS_GROUP3" \
        --config-name "baselines_configs/maps/metaworld/${task}" \
        "${EXTRA_ARGS[@]}"
    sleep 1
done

echo "Batch submission complete (MAPS)."
squeue -u "$(whoami)"
