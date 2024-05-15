#!/bin/bash

# This script will train all models and evaluate them on the test dataset. 
# All models and evaluation results will be stored under ./out/recreate_results/.
# Running everything can days multiple days, so it may be desirable to run only
# a subset.


# Extract the data.
tar -xzf ./data/frog_2022_04_29.tar.gz --directory ./data/
tar -xzf ./data/chicken_2021_08_17.tar.gz --directory ./data/

# Train all models. Each can take >10 hours.
python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_dist_chicken.yml
python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_poisson_chicken.yml
python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_dist_frog.yml
python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_poisson_frog.yml

# Run inference for all models (including the zero-spike control).
./recreate_results/eval_dist_chicken.sh
./recreate_results/eval_poisson_chicken.sh
./recreate_results/eval_dist_frog.sh
./recreate_results/eval_poisson_frog.sh
./recreate_results/eval_empty_chicken.sh
./recreate_results/eval_empty_frog.sh

