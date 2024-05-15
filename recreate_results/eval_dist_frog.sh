#!/bin/bash

# Run spike train inference for the spike distance model with frog data.

# Inputs
data_dir="./data/frog_2022_04_29"
recording_name="Xla_2022-04-29_Ph00_19"
experiment_dir="./out/recreate_paper/frog/dist/0"
# Outputs
spiketrain_path="${experiment_dir}/spiketain.parquet"
metrics_path="${experiment_dir}/metrics.parquet"

# Calculate metrics
python ./pybin/infer_dist.py \
    "$experiment_dir" \
    "$spiketrain_path" \
    "$metrics_path" \
	--data-dir "$data_dir" \
	--rec-name "$recording_name" \
    --strides 80 \
    --use-test-ds

# Check the exit status
exit_status=$?
if [ $exit_status -ne 0 ]; then
    echo "Python script exited with error: $exit_status"
    exit $exit_status
fi

echo "Calculation finished."
echo "Metrics: $metrics_path"
echo "Spike train: $metrics_path"
