#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

BASE_DIR="Your base data directory"

# ----------------------------------------------------------------------------------------------------
# The variable below should be changed according to the output file path of `run_eval_monitor.sh`.
# ----------------------------------------------------------------------------------------------------
POST_PATH="CERT/pandas-numpy-eval/data/Example_Pandas_PYCODEGPT_samples.jsonl"
EVALUATION_FILE="$BASE_DIR/$POST_PATH"
echo "Evaluation File Path: $EVALUATION_FILE"

evaluate_functional_correctness $EVALUATION_FILE

echo "File: $"
echo "All Done!"
