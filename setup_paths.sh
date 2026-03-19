#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_PATH="$SCRIPT_DIR/config/paths_local.yaml"
PROJECT_ROOT="${ORACLES_PROJECT_ROOT:-$SCRIPT_DIR}"

printf '# @package paths_local\nproject_root: "%s"\n' "$PROJECT_ROOT" > "$OUTPUT_PATH"
echo "Wrote $OUTPUT_PATH"
echo "project_root: $PROJECT_ROOT"
