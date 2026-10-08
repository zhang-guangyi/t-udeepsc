#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export GPU="${GPU:-0}"
export DEVICE="${DEVICE:-cuda}"
export OUTPUT_DIR="${OUTPUT_DIR:-ckpt_record_5tasks_f2dB}"
export MODEL="${MODEL:-UDeepSC_new_model}"
export BATCH_SIZE="${BATCH_SIZE:-24}"
export EPOCHS="${EPOCHS:-500}"
export LR="${LR:-5e-5}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export NUM_SAMPLES="${NUM_SAMPLES:-30000}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

export TRAIN_TASKS="${TRAIN_TASKS:-msa}"
export TASKS_PER_STEP="${TASKS_PER_STEP:-0}"
export TRAIN_SNR="${TRAIN_SNR:-0}" 
export TEST_SNR="${TEST_SNR:--2}"
export TEST_TASKS="${TEST_TASKS:-msa}"

export TA_PERFORM="${TA_PERFORM:-msa}"
export EVAL="${EVAL:-0}"
export EVAL_FREQ="${EVAL_FREQ:-1}"
export EVAL_BATCHES="${EVAL_BATCHES:-0}"
export GRAD_CONFLICT_FREQ="${GRAD_CONFLICT_FREQ:-0}"

# Keep the original empirical task scales unless explicitly overridden.
# export LOSS_WEIGHTS="${LOSS_WEIGHTS:-imgc:0.2 imgr:40 textc:0.6 textr:10 vqa:1 msa:8}"
# export LOSS_WEIGHTS="${LOSS_WEIGHTS:-imgc:0.2 imgr:30 textc:4 textr:0.5 vqa:2 msa:1}"
export LOSS_WEIGHTS="${LOSS_WEIGHTS:-imgc:0.2 imgr:30 textc:4 textr:5 vqa:1 msa:1}"
exec bash execute.sh
