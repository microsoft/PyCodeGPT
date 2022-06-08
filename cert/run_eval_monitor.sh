#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# [Pandas, Numpy]
DOMAIN="Numpy"

# [PYCODEGPT, PYCODEGPT_XL, CERT]
MODEL_VERSION="PYCODEGPT"

BASE_DIR="In local machine, please change this to your base data directory."

if [ ! -z "$AMLT_DATA_DIR" ]; then
    echo "Run evaluation experiment on AMLT."
    BASE_TOP_DIR=$AMLT_DATA_DIR
    BASE_DIR="In amlt machine, please change this to your base data directory."
fi

# --------------------------------------------------------------------------
# You should replace the following variables according to your own settings.
# --------------------------------------------------------------------------
if [ ${DOMAIN} == "Pandas" ]; then
    if [ ${MODEL_VERSION} == "PYCODEGPT" ]; then
        TEMP="1.0"
        NUM_SAMPLES="200"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME="${BASE_DIR}/pycodegpt-110M"
        Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert.py ${Run_Args} 
    elif [ ${MODEL_VERSION} == "PYCODEGPT_XL" ]; then
        TEMP="0.9"
        NUM_SAMPLES="200"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME="${BASE_DIR}/${DOMAIN}/experiments/normal_models/5x3x4x1024-5e-4-0.1-1K_100K_2K-RS_1.0_0.5_1.0/checkpoint-90000"
        Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert.py ${Run_Args}
    elif [ ${MODEL_VERSION} == "CERT" ]; then
        TEMP="1.0"
        TEMP2="1.0"
        NUM_SAMPLES="200"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME_SKETCHER="${BASE_DIR}/${DOMAIN}/experiments/sketcher_models/10x3x8x1024-5e-4-0.1-1K_100K_2K-RS_1.0_0.5_1.0/checkpoint-90000"
        CKPT_NAME_GENERATOR="${BASE_DIR}/${DOMAIN}/experiments/generator_models/10x3x8x1024-5e-4-0.1-1K_100K_2K-RS_1.0_0.5_1.0/checkpoint-20000"
        Run_Args="-model $CKPT_NAME_SKETCHER -model2 $CKPT_NAME_GENERATOR -t $TEMP -t2 $TEMP2 -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert_unified.py ${Run_Args}  
    fi
elif [ ${DOMAIN} == "Numpy" ]; then
    if [ ${MODEL_VERSION} == "PYCODEGPT" ]; then
        TEMP="1.0"
        NUM_SAMPLES="200"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME="${BASE_DIR}/pycodegpt-110M"
        Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert.py ${Run_Args} 
    elif [ ${MODEL_VERSION} == "PYCODEGPT_XL" ]; then
        TEMP="0.9"
        NUM_SAMPLES="200"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME="${BASE_DIR}/${DOMAIN}/experiments/normal_models/7x3x4x1024-5e-4-0.1-1K_100K_2K-RS_1.0_0.5_1.0/checkpoint-90000"
        Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert.py ${Run_Args}  
    elif [ ${MODEL_VERSION} == "CERT" ]; then
        TEMP="1.0"
        TEMP2="0.2"
        NUM_SAMPLES="200"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME_SKETCHER="${BASE_DIR}/${DOMAIN}/experiments/sketcher_models/10x3x8x1024-5e-4-0.1-1K_100K_2K-RS_1.0_0.5_1.0/checkpoint-98000"
        CKPT_NAME_GENERATOR="${BASE_DIR}/${DOMAIN}/experiments/generator_models/10x3x8x1024-5e-4-0.1-1K_100K_2K-RS_1.0_0.5_1.0/checkpoint-90000"
        Run_Args="-model $CKPT_NAME_SKETCHER -model2 $CKPT_NAME_GENERATOR -t $TEMP -t2 $TEMP2 -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert_unified.py ${Run_Args}  
    fi
fi

echo "All Done!"
