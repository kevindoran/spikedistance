#!/bin/bash

# Run spike train inference for the Poisson model distance model.
# Data: frog

# Inputs
data_dir="./data/frog_2022_04_29"
recording_name="Xla_2022-04-29_Ph00_19"
experiment_dir="./out/recreate_paper/frog/poisson/0"
# Outputs
metrics_path="${experiment_dir}/metrics.parquet"
spiketrain_path="${experiment_dir}/spiketrain.parquet"
sampling_metrics_path="${experiment_dir}/metrics_sampling.parquet"
sampling_spiketrain_path="${experiment_dir}/spiketrain_sampling.parquet"
mode_metrics_path="${experiment_dir}/metrics_mode.parquet"
mode_spiketrain_path="${experiment_dir}/spiketrain_mode.parquet"

# Round output
python ./pybin/infer_poisson.py \
	$experiment_dir  \
	$spiketrain_path \
	$metrics_path  \
	--data-dir "$data_dir" \
	--rec-name "$recording_name" \
	--use-test-ds \
	--infer-method "round"

# Check the exit status
exit_status=$?
if [ $exit_status -ne 0 ]; then
	echo "Poisson inference (via rounding) failed: $exit_status"
    exit $exit_status
fi


# Inference via sampling (from Poisson distribution)
python ./pybin/infer_poisson.py \
	$experiment_dir  \
	$sampling_spiketrain_path \
	$sampling_metrics_path  \
	--data-dir "$data_dir" \
	--rec-name "$recording_name" \
	--use-test-ds \
	--infer-method "sample"

# Check the exit status
exit_status=$?
if [ $exit_status -ne 0 ]; then
	echo "Poisson inference (sampling) failed: $exit_status"
    exit $exit_status
fi


# Floor/mode output
python ./pybin/infer_poisson.py \
	$experiment_dir  \
	$mode_spiketrain_path \
	$mode_metrics_path  \
	--data-dir "$data_dir" \
	--rec-name "$recording_name" \
	--use-test-ds \
	--infer-method "floor"

# Check the exit status
exit_status=$?
if [ $exit_status -ne 0 ]; then
	echo "Poisson inference (via mode) failed: $exit_status"
    exit $exit_status
fi

echo "Calculation finished."
echo "Metrics (via rounding): $metrics_path"
echo "Spike train (via rounding): $spiketrain_path"
echo "Metrics (via sampling): $sampling_metrics_path"
echo "Spike train (via sampling): $sampling_spiketrain_path"
echo "Metrics (via mode): $mode_metrics_path"
echo "Spike train (via mode): $mode_spiketrain_path"
