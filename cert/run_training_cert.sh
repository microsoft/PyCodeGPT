#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# setup for wandb
export WANDB_PROJECT="CERT"
export WANDB_API_KEY="Your wandb api key"

BASE_DATA_DIR="Your base data directory"
if [ ! -z "$AMLT_DATA_DIR" ]; then
    echo "Run experiment on AMLT."
    BASE_DATA_DIR=$AMLT_DATA_DIR
fi

# [Pandas, Numpy]
DOMAIN="Pandas"

# [normal, sketcher, generator]
TYPE="generator"

# ------------------Pandas------# ------Numpy--------------------------------
# You should replace the following variables according to your own settings.
# ------------------Pandas------# ------Numpy--------------------------------
DATA_DIR="${BASE_DATA_DIR}/CERT/${DOMAIN}/${TYPE}_bin"

N_GPUS="1"
NODE_SIZE="1"

if [ ! -z "$1" ]; then
    N_GPUS=$1
fi

if [ ! -z "$2" ]; then
    NODE_SIZE=$2
fi

BATCH_SIZE=1 # 24G:7 32G:8 16G:6
MAX_STEPS=100_000
BLOCK_SIZE=1024
GRAD_ACC_STEPS=2
WARMUP_STEPS=1_000
SAVE_STEPS=2_000

LR="5e-4"
WD="0.1"

# DO NOT take func score into consideration for resampling by setting a const weight 1.0
RS_WEIGHTS="1.0_0.5_1.0" #_0.5"
GAS="" #"512K_150K" #default is const

# --------------------------------------------------------------------------
# You should replace the following variables according to your own settings.
# --------------------------------------------------------------------------
OUTPUT_DIR="${BASE_DATA_DIR}/CERT/${DOMAIN}/experiments/${TYPE}_models"
CKPT_NAME=""

if [ ! -z "$AMLT_DATA_DIR" ]; then
    OUTPUT_DIR="$AMLT_DATA_DIR/CERT/${DOMAIN}/experiments/${TYPE}_models"
    BATCH_SIZE=10
    GRAD_ACC_STEPS=3

fi

STEP_SUMMARY="${WARMUP_STEPS}K_${MAX_STEPS}K_${SAVE_STEPS}K"
STEP_SUMMARY=${STEP_SUMMARY//_000/}

# Resampling with weight 0.4
# GRAD_ACC_STEPS + 1.0 per epoch
ID=""

if [ ! -z "$RS_WEIGHTS" ]; then
    ID="RS_${RS_WEIGHTS}"
fi

if [ ! -z "$GAS" ]; then
    ID="${ID}-GAS_${GAS}"
else
    GAS="const"
fi

if [ ! -z "$CKPT_NAME" ]; then
    ID="${ID}-RSUME"
fi

ACTUAL_GPUS=$((${N_GPUS}*${NODE_SIZE}))
RUN_NAME="${BATCH_SIZE}x${GRAD_ACC_STEPS}x${ACTUAL_GPUS}x${BLOCK_SIZE}-${LR}-${WD}-${STEP_SUMMARY}-${ID}"
RUN_OUTPUT_DIR="$OUTPUT_DIR/$RUN_NAME"

echo "Experiment Run Name: $RUN_NAME"
echo "Data Dir:" $DATA_DIR
echo "Actual GPUs:" $ACTUAL_GPUS
export DISTRIBUTED_GPU_SIZE=$ACTUAL_GPUS

echo "Output Dir:" $OUTPUT_DIR
echo "Init Actual Batch Size: ${BATCH_SIZE}x${GRAD_ACC_STEPS}x${N_GPUS}x${NODE_SIZE}"

Run_Command_Args=" --model_name_or_path $DATA_DIR/model"
Run_Command_Args="$Run_Command_Args --run_name $RUN_NAME"
Run_Command_Args="$Run_Command_Args --output_dir $RUN_OUTPUT_DIR"
Run_Command_Args="$Run_Command_Args --train_file $DATA_DIR/train"
Run_Command_Args="$Run_Command_Args --validation_file $DATA_DIR/valid"
Run_Command_Args="$Run_Command_Args --do_train"
Run_Command_Args="$Run_Command_Args --do_eval"

Run_Command_Args="$Run_Command_Args --block_size $BLOCK_SIZE"
Run_Command_Args="$Run_Command_Args --logging_steps 100"
Run_Command_Args="$Run_Command_Args --evaluation_strategy steps"
Run_Command_Args="$Run_Command_Args --eval_steps $SAVE_STEPS"
Run_Command_Args="$Run_Command_Args --save_steps $SAVE_STEPS"
Run_Command_Args="$Run_Command_Args --warmup_steps $WARMUP_STEPS"
Run_Command_Args="$Run_Command_Args --learning_rate $LR"
Run_Command_Args="$Run_Command_Args --adam_beta2 0.95"
Run_Command_Args="$Run_Command_Args --lr_scheduler_type cosine"
Run_Command_Args="$Run_Command_Args --resampling_weights $RS_WEIGHTS"

Run_Command_Args="$Run_Command_Args --max_steps $MAX_STEPS"
Run_Command_Args="$Run_Command_Args --per_device_train_batch_size $BATCH_SIZE"
Run_Command_Args="$Run_Command_Args --per_device_eval_batch_size  $BATCH_SIZE"
Run_Command_Args="$Run_Command_Args --gradient_accumulation_steps $GRAD_ACC_STEPS"
Run_Command_Args="$Run_Command_Args --weight_decay $WD"
Run_Command_Args="$Run_Command_Args --fp16"
Run_Command_Args="$Run_Command_Args --report_to wandb"

if [ "$OMPI_COMM_WORLD_RANK" == "0" ]; then
    mkdir "$OUTPUT_DIR/code/"
    cp -r "./" "$OUTPUT_DIR/code/"
    echo "Save experiment source code over."
fi

if [ ! -z "$GAS" ]; then
    Run_Command_Args="$Run_Command_Args --gradient_accumulation_strategy $GAS"
fi

if [ ! -z "$CKPT_NAME" ]; then
    CKPT_PATH=$"$OUTPUT_DIR/$CKPT_NAME"
    echo "Resume from checkpoint: $CKPT_PATH"
    Run_Command_Args="$Run_Command_Args --resume_from_checkpoint $CKPT_PATH --ignore_data_skip"
fi

echo "Run Command Args: $Run_Command_Args"

# deepspeed --num_gpus $N_GPUS run_gpt.py --deepspeed configs/ds_config.json $Run_Command_Args
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node $N_GPUS --use_env run_cert.py $Run_Command_Args
