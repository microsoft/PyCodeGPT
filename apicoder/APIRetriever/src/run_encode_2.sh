# !/bin/bash

LIBRIES=( "pandas" "numpy" "monkey" "beatnum" "torchdata")
MODES=( "comment" "api")

for MODE in ${MODES[@]}; do
  echo "Mode: $MODE"
  for LIBRARY in ${LIBRIES[@]}; do
    echo "Library: $LIBRARY"
    OUTDIR="../data/inference"
    MODEL_DIR="../outputs/APIRetrieverCheckPoint/"
    CORPUS_DIR="../data/inference"
    ENCODE_DIR="../data/inference"
    PER_BATCH_SIZE=50

    CUDA_VISIBLE_DEVICES=0 python -m dense.driver.encode \
      --output_dir=$OUTDIR \
      --model_name_or_path $MODEL_DIR \
      --fp16 \
      --per_device_eval_batch_size ${PER_BATCH_SIZE} \
      --local_rank -1 \
      --encode_in_path "${CORPUS_DIR}/${LIBRARY}_${MODE}.json" \
      --encoded_save_path "${ENCODE_DIR}/${LIBRARY}_${MODE}.pt"
  done
done
