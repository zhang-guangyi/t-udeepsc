#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU="${GPU:-1}"
MODEL="${MODEL:-UDeepSC_new_model}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpt_record_f2dB_single}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INPUT_SIZE="${INPUT_SIZE:-32}"
LR="${LR:-6e-5}"
EPOCHS="${EPOCHS:-500}"
SAVE_FREQ="${SAVE_FREQ:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
TA_PERFORM="${TA_PERFORM:-imgc}"
TEST_TASKS="${TEST_TASKS:-}"
TRAIN_TASKS="${TRAIN_TASKS:-msa textr}"
TASKS_PER_STEP="${TASKS_PER_STEP:-0}"
TASK_WEIGHTS="${TASK_WEIGHTS:-}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-}"
TRAIN_SNR="${TRAIN_SNR:-12}"
TEST_SNR="${TEST_SNR:-12}"
TEST_SNR_LIST="${TEST_SNR_LIST:-}"
EVAL_FREQ="${EVAL_FREQ:-1}"
EVAL_BATCHES="${EVAL_BATCHES:-0}"
GRAD_CONFLICT_FREQ="${GRAD_CONFLICT_FREQ:-0}"
PRINT_FREQ="${PRINT_FREQ:-50}"


RESUME="${RESUME:-}"
INIT_CKPT="${INIT_CKPT:-}"
EVAL="${EVAL:-1}"   
   
cmd=(
  python3 udeepsc_main.py
  --model "$MODEL"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --input_size "$INPUT_SIZE"
  --lr "$LR"
  --epochs "$EPOCHS"
  --num_samples "$NUM_SAMPLES"
  --num_workers "$NUM_WORKERS"
  --opt_betas 0.95 0.99
  --save_freq "$SAVE_FREQ"
  --device "$DEVICE"
  --ta_perform "$TA_PERFORM"
  --train_tasks $TRAIN_TASKS
  --tasks_per_step "$TASKS_PER_STEP"
  --train_snr $TRAIN_SNR
  --test_snr "$TEST_SNR"
  --eval_freq "$EVAL_FREQ"
  --eval_batches "$EVAL_BATCHES"
  --grad_conflict_freq "$GRAD_CONFLICT_FREQ"
  --print_freq "$PRINT_FREQ"
)

if [[ -n "$TASK_WEIGHTS" ]]; then
  cmd+=(--task_weights $TASK_WEIGHTS)
fi

if [[ -n "$LOSS_WEIGHTS" ]]; then
  cmd+=(--loss_weights $LOSS_WEIGHTS)
fi

if [[ -n "$TEST_TASKS" ]]; then
  cmd+=(--test_tasks $TEST_TASKS)
fi

if [[ -n "$TEST_SNR_LIST" ]]; then
  cmd+=(--test_snr_list $TEST_SNR_LIST)
fi

if [[ -n "$RESUME" ]]; then
  cmd+=(--resume "$RESUME")
fi

if [[ -n "$INIT_CKPT" ]]; then
  cmd+=(--init_ckpt "$INIT_CKPT")
fi

if [[ "$EVAL" == "1" ]]; then
  cmd+=(--eval)
fi

log_dir="$OUTPUT_DIR/logs"
mkdir -p "$log_dir"
run_id="$(date +%Y%m%d_%H%M%S)"
log_file="$log_dir/run_${run_id}.log"

echo "Log file: $log_file"
echo "Metrics file: $OUTPUT_DIR/train_metrics.jsonl"
if [[ -n "$RESUME" ]]; then
  echo "Resume checkpoint: $RESUME"
else
  echo "Resume checkpoint: none"
fi
if [[ -n "$INIT_CKPT" ]]; then
  echo "Init checkpoint: $INIT_CKPT"
else
  echo "Init checkpoint: none"
fi
echo "CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
CUDA_VISIBLE_DEVICES="$GPU" python3 -u "${cmd[@]:1}" 2>&1 | tee "$log_file"
