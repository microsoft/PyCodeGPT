#!/bin/bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

BASE_DIR="Your base data directory"

if [ ! -z "$AMLT_DATA_DIR" ]; then
    echo "Run evaluation experiment on AMLT."
    BASE_DIR=$AMLT_DATA_DIR
fi

# ----------------------------------------------------------------------------------------------------
# The variable below should be changed according to the output file path of `run_eval_monitor.sh`.
# ----------------------------------------------------------------------------------------------------
POST_PATH="CERT/pycodegpt-110M/Numpy_API_eval.PYCODEGPT.t1.0.p0.9.l100.n200.samples.jsonl"
EVALUATION_FILE="$BASE_DIR/$POST_PATH"
echo "Evaluation File Path: $EVALUATION_FILE"

evaluate_functional_correctness $EVALUATION_FILE

echo "File: $"
echo "All Done!"
