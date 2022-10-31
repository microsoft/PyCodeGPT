#!/bin/bash

BASE_DIR="your/base/dir"

TEMP=$1

# Temperature needs to be replaced with: $TEMP
# Remember to change the human/data path; Remember to add torchdata in requirements.txt; Start with CERT/
POST_PATH="XXX/codeparrot-small/official_TorchData_machine_gpt2_apinum_5_temp_$TEMP.samples.jsonl"

EVALUATION_FILE="$BASE_DIR/$POST_PATH"
echo "Evaluation File Path: $EVALUATION_FILE"
evaluate_functional_correctness $EVALUATION_FILE

echo "All Done!"