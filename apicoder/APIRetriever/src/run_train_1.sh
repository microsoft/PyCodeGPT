#!/bin/bash

export WANDB_PROJECT="Your Project Name"
export WANDB_API_KEY="Your WANDB API Key"

TRAIN_DIR="../data/train"
OUTDIR="../outputs"
MODEL_PATH="/your/path/of/bert-base-uncased"

python -m dense.driver.train \
  --output_dir $OUTDIR \
  --model_name_or_path ${MODEL_PATH} \
  --do_train \
  --save_steps 200 \
  --train_dir $TRAIN_DIR \
  --fp16 \
  --per_device_train_batch_size 5 \
  --train_n_passages 8 \
  --learning_rate 1e-5 \
  --q_max_len 256 \
  --p_max_len 256 \
  --num_train_epochs 150 \
