#!/bin/bash

# Run spike train inference with the control "zero spike" model.
# There isn't any spikes to infer, however, the associated metrics still need
# computing.

# Inputs
data_dir="./data/frog_2022_04_29"
recording_name="Xla_2022-04-29_Ph00_19"
# Outputs
metrics_path="./out/recreate_paper/frog/zero_model_metrics.parquet"

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
