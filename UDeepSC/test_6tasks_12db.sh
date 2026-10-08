#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

tasks=(${TEST_TASKS:-imgc imgr textc textr vqa msa})

echo "Testing tasks: ${tasks[*]}"
for task in "${tasks[@]}"; do
  echo "============================================================"
  echo "Testing task: $task"
  TASK="$task" bash test_one_task_12db.sh
done
