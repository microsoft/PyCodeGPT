#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

ID=$(date +"%m%d")
BASE_DATA_DIR="Your base data directory"

# [Pandas, Numpy]
DOMAIN="Pandas"
# [normal, sketcher, generator]
TYPE="generator"
# [train, valid]
SPLIT="valid"
# [True, False]
IS_DEBUG="False"

# --------------------------------------------------------------------------
# You should replace the following variables according to your own settings.
# --------------------------------------------------------------------------
DATA_DIR="${BASE_DATA_DIR}/datasets/CERT/${DOMAIN}/data"
MODEL_DIR="${BASE_DATA_DIR}/models/pycodegpt-110M"
OUTPUT_DIR="${BASE_DATA_DIR}/datasets/CERT/${DOMAIN}/${TYPE}_bin"

if [ ! -z "$AMLT_DATA_DIR" ]; then
    echo "Run experiment on AMLT."
    BASE_DATA_DIR=$AMLT_DATA_DIR
    DATA_DIR="${BASE_DATA_DIR}/CERT/${DOMAIN}/data"
    MODEL_DIR="${BASE_DATA_DIR}/CERT/pycodegpt-110M"
    OUTPUT_DIR="${BASE_DATA_DIR}/CERT/${DOMAIN}/${TYPE}_bin"
fi

if [ $IS_DEBUG == "True" ]; then
    N_CPUS="1"
else
    N_CPUS="20"
fi


if [ ! -z "$1" ]; then
    N_CPUS="$1"
fi

if [ ! -z "$2" ]; then
    echo "Using distributed nodes: $2"
    export DistributedNodes=$2
fi

if [ ! -z "$AMLT_DATA_DIR" ]; then
    echo "Run experiment on AMLT."
fi

Args="-i $DATA_DIR -o $OUTPUT_DIR -model $MODEL_DIR -t $N_CPUS -d $DOMAIN -type $TYPE -isdebug $IS_DEBUG"
echo "Run encode_domain for ${SPLIT} data: $Args"
python encode_domain.py $Args -split ${SPLIT}