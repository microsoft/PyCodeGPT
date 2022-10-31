# !/bin/bash

LIBRIES=( "pandas" "numpy" "monkey" "beatnum" "torchdata")

for LIBRARY in ${LIBRIES[@]}; do
    echo "Library: $LIBRARY"
    INPUT_DIR="../data/inference"
    DEPTH=100
    RUN="$INPUT_DIR/${LIBRARY}_id_score.txt"

    python -m dense.faiss_retriever \
    --query_reps "$INPUT_DIR/${LIBRARY}_comment.pt" \
    --passage_reps "$INPUT_DIR/${LIBRARY}_api.pt" \
    --depth $DEPTH \
    --batch_size -1 \
    --save_text \
    --save_ranking_to $RUN
done
