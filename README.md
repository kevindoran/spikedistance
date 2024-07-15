Code for the paper "Spike distance function as a learning objective for spike prediction". ([project page](https://kdoran.com/spikedistance/)) ([ICML 2024 paper](https://icml.cc/virtual/2024/poster/33205))

# Setup
Install the local package `retinapy`:

    pip install ./retinapy

## Dependencies

Dependencies include: 

    - Python (>=3.10)
    - Cuda (>=11.8) 

Pip dependencies are specified in `./retinapy/setup.py`. 
A frozen set of Pip requirements is listed in `requirements_snapshot.txt`.

# Recreate results
All below commands are run from the project root directory.

## 1. Prepare data

Extract the data with:

    tar -xzf ./data/chicken_2021_08_17.tar.gz --directory ./data/
    tar -xzf ./data/frog_2022_04_29.tar.gz --directory ./data/

The provided data includes the subset of the cells used in the experiments. The
data for all cells is larger. The full chicken data is provided with the
publication by 
[Seifert et al (2023)](https://www-nature-com.sussex.idm.oclc.org/articles/s41467-023-41032-z#data-availability).
The full frog data is available on request. The code used to filter out cells is
included in `pybin/filter_cells.py`.


## 2 Training
Train the models on both datasets.

Distance array model:

    python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_dist_chicken.yml
    python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_dist_frog.yml

Poisson models:

    python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_poisson_chicken.yml
    python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_poisson_frog.yml


## 3. Evaluation
Evaluate the models on both datasets.

Distance array model:

    ./recreate_results/eval_dist_chicken.sh
    ./recreate_results/eval_dist_frog.sh

Poisson models:

    ./recreate_results/eval_poisson_chicken.sh
    ./recreate_results/eval_poisson_frog.sh

Zero-spike model (control):

    ./recreate_results/eval_empty_chicken.sh
    ./recreate_results/eval_empty_frog.sh

## Run everything

All the above commands are combined in the script:

    ./recreate_results/run_all.sh


# Code summary
The code to recreate the experiments sits within a larger project.

For this paper, the relevant PyTorch modules are in `retinapy/models.py`:

    - the distance array module uses `CnnUNet3` 
    - the Poisson model uses `PoissonNet2` 

The training entry point is in `spikeprediction.py`; a `TrainableGroup` creates 
a `Trainable` which contains a model: 

    - `DistfieldUnet2` creates a `SingleDistTrainable` with a `CnnUNet3`.
    - `PoissonNet2` creates a `PoissonTrainable` with a `PoissonNet2`.

Calculation of and inference from spike distance arrays is handled in 
`retinapy/spikedistance.py`. 

Batch inference for the three model types (Poisson, distance and zero-spike control)
is carried out in `pybin/infer_poisson.py`, `pybin/infer_dist.py` and `pybin/infer_zero_model.py`.
The inference produces Polars dataframes, saved in Parquet files.


# Docker
Functionality beyond recreating the results may need additional packages. 
The Dockerfile `./Dockerfile` creates an image that was used in development and 
contains many more dependencies.

Run `./scripts/builddocker` to build an image. 

Run `./scripts/runcmd "<cmd>"` to create a container and run a bash command.

For example:

    ./scripts/runcmd "python ./retinapy/src/retinapy/spikeprediction.py --config ./recreate_results/train_dist_chicken.yml"

