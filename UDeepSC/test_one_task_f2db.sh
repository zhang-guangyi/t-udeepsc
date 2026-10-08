#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export GPU="${GPU:-3}"
export DEVICE="${DEVICE:-cuda}"
export OUTPUT_DIR="${OUTPUT_DIR:-ckpt_record_6tasks_f2dB}"
export MODEL="${MODEL:-UDeepSC_new_model}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export TEST_SNR="${TEST_SNR:-12}"
export TEST_SNR_LIST="${TEST_SNR_LIST:-}"
export EVAL_BATCHES="${EVAL_BATCHES:-0}"

task="${TASK:-${TA_PERFORM:-vqa}}"
case "$task" in
  imgc|imgr|textc|textr|vqa|msa) ;;
  *)
    echo "Unsupported TASK/TA_PERFORM: $task" >&2
    echo "Valid tasks: imgc imgr textc textr vqa msa" >&2
    exit 2
    ;;
esac

latest_checkpoint() {
  local search_dir="$1"
  find "$search_dir" -path '*/ckpt_*' -type f -name 'checkpoint-*.pth' \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-
}

resume="${RESUME:-}"
if [[ ! -f "$resume" ]]; then
  echo "Checkpoint not found: $resume" >&2
  echo "Set RESUME to your checkpoint file and rerun." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/test_logs" vqaeval_result
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$OUTPUT_DIR/test_logs/test_${task}_${run_id}.log"

cmd=(
  python3 udeepsc_main.py
  --model "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --device "$DEVICE"
  --ta_perform "$task"
  --test_snr "$TEST_SNR"
  --eval_batches "$EVAL_BATCHES"
  --resume "$resume"
  --eval
)

if [[ -n "$TEST_SNR_LIST" ]]; then
  cmd+=(--test_snr_list $TEST_SNR_LIST)
fi

echo "Task: $task"
echo "Checkpoint: $resume"
echo "Log file: $log_file"
echo "CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "$log_file"
