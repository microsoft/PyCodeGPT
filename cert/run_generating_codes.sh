#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# [Pandas, Numpy]
DOMAIN="Pandas"

# [PYCODEGPT, CERT]
MODEL_VERSION="PYCODEGPT"

BASE_DIR="In local machine, please change this to your base data directory."

# --------------------------------------------------------------------------
# You should replace the following variables according to your own settings.
# --------------------------------------------------------------------------
if [ ${DOMAIN} == "Pandas" ]; then
    if [ ${MODEL_VERSION} == "PYCODEGPT" ]; then
        TEMP="1.0"
        NUM_SAMPLES="1"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME="${BASE_DIR}/pycodegpt-110M"
        Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert.py ${Run_Args} 
    elif [ ${MODEL_VERSION} == "CERT" ]; then
        TEMP="1.0"
        TEMP2="1.0"
        NUM_SAMPLES="1"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME_SKETCHER="${BASE_DIR}/sketcher-pandas"
        CKPT_NAME_GENERATOR="${BASE_DIR}/generator-pandas"
        Run_Args="-model $CKPT_NAME_SKETCHER -model2 $CKPT_NAME_GENERATOR -t $TEMP -t2 $TEMP2 -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert_unified.py ${Run_Args}  
    fi
elif [ ${DOMAIN} == "Numpy" ]; then
    if [ ${MODEL_VERSION} == "PYCODEGPT" ]; then
        TEMP="1.0"
        NUM_SAMPLES="1"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME="${BASE_DIR}/pycodegpt-110M"
        Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert.py ${Run_Args} 
    elif [ ${MODEL_VERSION} == "CERT" ]; then
        TEMP="1.0"
        TEMP2="0.2"
        NUM_SAMPLES="1"
        MAX_TOKNES="100"
        TOP_P="0.9"
        CKPT_NAME_SKETCHER="${BASE_DIR}/sketcher-numpy"
        CKPT_NAME_GENERATOR="${BASE_DIR}/generator-numpy"
        Run_Args="-model $CKPT_NAME_SKETCHER -model2 $CKPT_NAME_GENERATOR -t $TEMP -t2 $TEMP2 -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION"
        echo "Run Args: $Run_Args"
        python eval_cert_unified.py ${Run_Args}  
    fi
fi

echo "All Done!"
