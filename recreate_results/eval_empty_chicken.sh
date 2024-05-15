#!/bin/bash

# Run spike train inference with the control "zero spike" model.
# There isn't any spikes to infer, however, the associated metrics still need
# computing.

# Inputs
data_dir="./data/chicken_2021_08_17"
recording_name="Chicken_17_08_21_Phase_00"
# Outputs
metrics_path="./out/recreate_paper/chicken/zero_model_metrics.parquet"

python ./pybin/infer_zero_model.py \
--data-dir $data_dir \
--rec-name $recording_name \
$metrics_path

# Check the exit status 
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Python script exited with error: $exit_status"
    exit $exit_status
fi

echo "Zero-spike model metrics path: $metrics_path"
