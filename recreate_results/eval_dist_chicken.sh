#!/bin/bash

# Run spike train inference for the spike distance model with chicken data.

# Inputs
data_dir="./data/chicken_2021_08_17"
recording_name="Chicken_17_08_21_Phase_00"
experiment_dir="./out/recreate_paper/chicken/dist/0"
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
