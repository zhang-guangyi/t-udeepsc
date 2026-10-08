#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export GPU="${GPU:-0}"
export DEVICE="${DEVICE:-cuda}"
export OUTPUT_DIR="${OUTPUT_DIR:-ckpt_record_6tasks_12dB}"
export MODEL="${MODEL:-UDeepSC_new_model}"
export BATCH_SIZE="${BATCH_SIZE:-14}"
export EPOCHS="${EPOCHS:-500}"
export LR="${LR:-5e-5}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export NUM_SAMPLES="${NUM_SAMPLES:-10000}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

export TRAIN_TASKS="${TRAIN_TASKS:-imgc imgr textc textr vqa}"
# export TRAIN_TASKS="${TRAIN_TASKS:-textr}"
export TEST_TASKS="${TEST_TASKS:-textc}"
export TASKS_PER_STEP="${TASKS_PER_STEP:-0}"
export TRAIN_SNR="${TRAIN_SNR:-12}"
export TEST_SNR="${TEST_SNR:-12}"

export TA_PERFORM="${TA_PERFORM:-textr}"
export EVAL="${EVAL:-0}"
export EVAL_FREQ="${EVAL_FREQ:-1}"
export EVAL_BATCHES="${EVAL_BATCHES:-0}"
export GRAD_CONFLICT_FREQ="${GRAD_CONFLICT_FREQ:-0}"

# Keep the original empirical task scales unless explicitly overridden.
export LOSS_WEIGHTS="${LOSS_WEIGHTS:-imgc:0.2 imgr:4 textc:0.1 textr:0.1 vqa:0 msa:1}"
# export LOSS_WEIGHTS="${LOSS_WEIGHTS:-imgc:0.02 imgr:30 textc:0.006 textr:0.1 vqa:0.01 msa:8}"
exec bash execute.sh
