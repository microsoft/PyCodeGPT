#!/bin/bash

# [True, False]
HUMAN_IN_THE_LOOP="False"
# ["_no", "_make_sense"]
MAKE_SENSE="_no"
# [machine, top3_perfect, top4_perfect, top5_perfect, human_labelled]
USER_NAME="machine"
# [0, 1, 2, 3, 5, "n"]
API_NUMBER=0
# [Pandas, Numpy, Monkey, BeatNum, TorchData]
DOMAIN="TorchData"
# [CodeGen, API_Coder] [codet5, CodeGPT, CodeClippy, CodeParrot]
MODEL_VERSION="CodeGen"
TEMP=$1

BASE_DIR="your/base/dir"

if [ ${MODEL_VERSION} == "CodeGen" ]; then
    NUM_SAMPLES="1"
    MAX_TOKNES="100"
    TOP_P="0.9"
    CKPT_NAME="${BASE_DIR}/codegen-350M-mono"
    Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION --api_number $API_NUMBER --human_in_the_loop $HUMAN_IN_THE_LOOP --user_name $USER_NAME --make_sense $MAKE_SENSE"
    echo "Run Args: $Run_Args"
    python eval_private.py ${Run_Args}
elif [ ${MODEL_VERSION} == "API_Coder" ]; then
    NUM_SAMPLES="200"
    MAX_TOKNES="100"
    TOP_P="0.9"
    CKPT_NAME="${BASE_DIR}/CodeGenAPI-350M-mono"
    Run_Args="-model $CKPT_NAME -t $TEMP -p $TOP_P -l $MAX_TOKNES -n $NUM_SAMPLES -d $DOMAIN -mv $MODEL_VERSION --api_number $API_NUMBER --human_in_the_loop $HUMAN_IN_THE_LOOP --user_name $USER_NAME --make_sense $MAKE_SENSE"
    echo "Run Args: $Run_Args"
    python eval_private.py ${Run_Args} 
elif [ ${MODEL_VERSION} == "codet5" ]; then
    python eval_baseline.py -m "$BASE_DIR/codet5-base" -temp $TEMP -type codet5 -lib $DOMAIN --api_number $API_NUMBER --user_name $USER_NAME
elif [ ${MODEL_VERSION} == "CodeGPT" ]; then
    python eval_baseline.py -m "$BASE_DIR/CodeGPT-small-py-adaptedGPT2" -temp $TEMP -type gpt2 -lib $DOMAIN --api_number $API_NUMBER --user_name $USER_NAME
elif [ ${MODEL_VERSION} == "CodeClippy" ]; then
    python eval_baseline.py -m "$BASE_DIR/gpt-neo-125M-code-clippy" -temp $TEMP -type gpt-neo -lib $DOMAIN --api_number $API_NUMBER --user_name $USER_NAME
elif [ ${MODEL_VERSION} == "CodeParrot" ]; then
    python eval_baseline.py -m "$BASE_DIR/codeparrot-small" -temp $TEMP -type gpt2 -lib $DOMAIN --api_number $API_NUMBER --user_name $USER_NAME
fi

echo "All Done!"