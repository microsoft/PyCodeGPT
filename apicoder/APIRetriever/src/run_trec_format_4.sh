# !/bin/bash

LIBRIES=( "pandas" "numpy" "monkey" "beatnum" "torchdata")

for LIBRARY in ${LIBRIES[@]}; do
    echo "Library: $LIBRARY"
    INPUT_DIR="../data/inference"
    RUN="$INPUT_DIR/${LIBRARY}_id_score.txt"
    TREC_RUN="$INPUT_DIR/${LIBRARY}_id_score.trec"
    
    python -m dense.utils.format.convert_result_to_trec --input $RUN --output $TREC_RUN
done
