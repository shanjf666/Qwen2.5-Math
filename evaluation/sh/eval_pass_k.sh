#!/bin/bash
set -ex

# Usage: bash sh/eval_pass_k.sh <PROMPT_TYPE> <MODEL_NAME_OR_PATH> [N_SAMPLING] [TEMPERATURE]
# Example:
#   bash sh/eval_pass_k.sh qwen25-math-cot /path/to/your/model 16 0.7

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
N_SAMPLING=${3:-16}         # default: 16 samples per problem
TEMPERATURE=${4:-0.7}       # default: temperature 0.7
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1

# ============================================================
# Evaluate on: math, gsm8k, aime24, amc23
# With n_sampling for pass@k calculation
# ============================================================

# MATH dataset (5000 problems)
DATA_NAME="math"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

# GSM8K dataset
DATA_NAME="gsm8k"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

# AIME 2024
DATA_NAME="aime24"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

# AMC 2023
DATA_NAME="amc23"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite

echo "============================================================"
echo "Pass@k evaluation complete!"
echo "Results are saved to: ${OUTPUT_DIR}"
echo "============================================================"
