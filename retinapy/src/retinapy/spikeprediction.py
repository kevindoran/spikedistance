"""
Quite a few experiments are kicked off by this file. There is a bit of messy
infrastruture used to kick off the training of a large number of different model 
configurations with one command. 

A TrainableGroup is used to create Trainable and compatible datasets.
Most TrainableGroups are expected to be able to prepair trainables and datasets
for all of the Configurations, although, in later experiments, most 
TrainableGroups only deal with a subset. The TrainableGroups have names and
are registered in _create_tgroups(). The names are matched against options 
passed on the cmdline to determine which (probably multiple) training runs to 
kick off.
"""

import argparse
from collections import defaultdict
import concurrent
import dataclasses
import enum
import functools
import logging
import math
import multiprocessing as mp
import pathlib
import re
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Set,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
)
import deprecated
import einops
import numpy as np
import scipy
import torch
import yaml
import plotly
import plotly.graph_objects
import polars as pl
import retinapy
import retinapy._logging
import retinapy.cmdline
import retinapy.dataset
import retinapy.mea as mea
import retinapy.metrics
import retinapy.models
import retinapy.nn
import retinapy.spikedistance as sdf
import retinapy.train
import retinapy.vis
import torchinfo


DEFAULT_OUT_BASE_DIR = "./out/"
LOG_FILENAME = "train.log"
ARGS_FILENAME = "args.yaml"
TRAINABLE_CONFIG_FILENAME = "trainable_config.yaml"
TENSORBOARD_DIR = "tensorboard"

IN_CHANNELS = 4 + 1
# Quite often there are lengths in the range 300.
# The pad acts as the maximum, so it's a good candidate for a norm factor.
# Example: setting normalization to 400 would cause 400 time steps to be fit
# into the [0,1] region.
LOSS_CALC_PAD_MS = 200
DIST_CLAMP_MS = 200
SPLIT_RATIO = (7, 2, 1)
MIN_SPIKES_DEFAULT = 0
assert sum(SPLIT_RATIO) == 10, (
    "The minimum spike setting depends on the unit of the split ratio"
    "summing to 10."
)

# Anything above 19 Hz is just a channel following the 20 Hz trigger.
MAX_SPIKE_RATE = 19  # Hz
SUPPORTED_SENSOR_SAMPLE_RATE = 17852.767845719834  # Hz

_logger = logging.getLogger(__name__)

# A [train, val, test], where each of train, val and test are a list of
# recording parts.
ContiguousChunks: TypeAlias = List[mea.SpikeRecording]
RecordingTrainValTest: TypeAlias = Tuple[
    ContiguousChunks, ContiguousChunks, ContiguousChunks
]


def arg_parsers():
    """Parse commandline and config file arguments.

    The approach carried out here is inspired by the pytorch-image-models
    project:
        https://github.com/rwightman/pytorch-image-models

    Arguments are populated in the following order:
        1. Default values
        2. Config file
        3. Command line
    """
    config_parser = retinapy.cmdline.create_yaml_parser()
    parser = argparse.ArgumentParser(description="Spike detection training")
    # fmt: off

    # Model/config arguments
    # Using -k as a filter, just like pytest.
    parser.add_argument("-k", type=str, default=None, metavar="EXPRESSION", help="Filter configs and models to train or test.")

    # Optimization parameters
    opt_group = parser.add_argument_group("Optimizer parameters")
    opt_group.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    opt_group.add_argument("--weight-decay", type=float, default=1e-6, help="weight decay (default: 2e-5)")

    # Data
    data_group = parser.add_argument_group("Data parameters")
    data_group.add_argument("--data-dir", type=str, default=None, metavar="FILE", help="Path to stimulus pattern file.")
    data_group.add_argument("--recording-names", nargs='+', type=str, default=None, help="Names of recordings within the recording file.")
    data_group.add_argument("--cluster-ids", nargs='+', type=int, default=None, help="Cluster ID to train on.")
    data_group.add_argument("--min-spikes", type=int, default=MIN_SPIKES_DEFAULT, help="Minimum spikes needed per-split division in train, val or test datasets for a cluster to be included.") 

    parser.add_argument("--steps-til-eval", type=int, default=None, help="Steps until validation.")
    parser.add_argument("--steps-til-log", type=int, default=100, help="How many batches to wait before logging a status update.")
    parser.add_argument("--evals-til-eval-train-ds", type=int, default=10, help="After how many validation runs with the validation data should validation be run with the training data.")
    parser.add_argument("--initial-checkpoint", type=str, default=None, help="Initialize model from the checkpoint at this path.")
    #parser.add_argument("--resume", type=str, default=None, help="Resume full model and optimizer state from checkpoint path.")
    parser.add_argument("--output", type=str, default=None, metavar="DIR", help="Path to output folder (default: current dir).")
    parser.add_argument("--labels", type=str, default=None, help="List of experiment labels. Used for naming files and/or subfolders.")
    parser.add_argument("--epochs", type=int, default=8, metavar="N", help="number of epochs to train (default: 300)")
    parser.add_argument("--early-stopping", type=int, default=None, metavar="N", help="number of epochs with no improvement after which to stop")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="batch size for eval")
    parser.add_argument("--num-workers", type=int, default=24, help="Number of workers for data loading.")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, help="Pin memory?")
    parser.add_argument("--log-activations", action=argparse.BooleanOptionalAction, help="Enable tag logging.")

    transformer_model_group = parser.add_argument_group("Transformer model options")
    transformer_model_group.add_argument("--zdim", type=int, default=2, help="VAE latent dimension")
    transformer_model_group.add_argument("--vae-beta", type=float, default=0.01, help="VAE beta parameter.")
    transformer_model_group.add_argument("--stride", type=int, default=17, help="Dataset stride.")
    transformer_model_group.add_argument("--num-heads", type=int, default=8, help="Number of transformer heads.")
    transformer_model_group.add_argument("--head-dim", type=int, default=32, help="Dimension of transformer heads.")
    transformer_model_group.add_argument("--num-tlayers", type=int, default=5, help="Number of transformer layers.")
    cnn_model_group = parser.add_argument_group("CNN model options")
    cnn_model_group.add_argument("--num-down", type=int, default=None, help="Number of downsample layers.")
    cnn_model_group.add_argument("--num-mid", type=int, default=None, help="Number of stride-1 layers between down and up.")
    cnn_model_group.add_argument("--num-up", type=int, default=None, help="Number of upsample layers.")
    # fmt: on
    return parser, config_parser


def opt_from_yaml(yaml_path: str | pathlib.Path):
    cmdline_parser, _ = arg_parsers()
    retinapy.cmdline.populate_from_yaml(cmdline_parser, yaml_path)
    res = cmdline_parser.parse_args([])
    return res


class Configuration:
    def __init__(self, downsample, input_len, output_len):
        self.downsample = downsample
        self.input_len = input_len
        self.output_len = output_len

    def __str__(self):
        return f"{self.downsample}ds_{self.input_len}in_{self.output_len}out"

    @property
    def output_ms(self):
        return self.output_len * self.downsample / (mea.ELECTRODE_FREQ / 1000)


def configuration_representer(dumper, data):
    return dumper.represent_mapping(
        "!Configuration",
        {
            "downsample": data.downsample,
            "input_len": data.input_len,
            "output_len": data.output_len,
        },
    )


def configuration_constructor(loader, node):
    value = loader.construct_mapping(node, deep=True)
    return Configuration(**value)


yaml.add_representer(Configuration, configuration_representer)
yaml.add_constructor("!Configuration", configuration_constructor)

"""
The following table is useful as a reference to see the effects of each
downsampling factor. The downsampling rates in the table were chosen to fall
closely to some "simple" frequencies/periods, like 1 kHz, etc.

Downsample factors
==================
Downsample : Freq (Hz)   : Timestep period (ms)
1          : 17852.768   :
2          :  8926.389   :
4          :  4463.192   :  0.224
9          :  1983.641   :  0.504
18         :   991.820   :  1.001
36         :   495.910   :  2.016
71         :   251.447   :  3.977
89         :   200.593   :  4.985
143        :   124.845   :  8.001
179        :    99.736   : 10.026
"""
# Roughly, we are aiming for the following:
downsample_factors = [9, 18, 89, 179]
input_lengths_ms = [1000, 1600]
output_lenghts_ms = [1, 10, 50, 100]

"""
The following are the model input-output configurations we will use. The
three variables are:

    - downsampling factor
    - input length
    - output length

Given that each model will be trained for each configuration, it is not
feasible to have a large number of configurations. So some care has been
given to pick an initial set that will hopefully be interesting and give us
some insight into the problem. After this, we should be able to narrow in
on a smaller set of configurations.

There are a few considerations driving the choices for these variables.

Downsample factor
-----------------
We are not sure at what point downsampling causes loss of important
information. Ideally, we would like to work with a low sample rate. Trying with
a few different rates will help us to determine the trade-offs at each
downsampling factor. It's likely that further experiments can work with a
single sampling rate.

Input length
------------
I am told that previous experiments have found that retinal ganglion cells can
depend on the past 1200 ms of input. At least one of the papers I am trying to
benchmark against used much shorter inputs (400ms). I am setting a high input
rate of 1600 to give a decent buffer so that we can support/test an argument
that no more than X ms seem to be used, where X is currently hypothesized to be
around 1200. I am setting a low input length of 1000 ms to test if 1000 is
insufficient. The choice of 1000 is somewhat arbitrary. If 1000 is
insufficient, we can justify working with a shorter 1000 ms input, which is a
win from an engineering point of view. If 1000 is insufficient, then this is
nice evidence to support the hypothesis that >1000 ms are used. If we had
chosen 400 ms, this result would not be interesting, as I think it is widely
accepted that the ganglion cells depend on more than the last 400 ms. So 1000
was chosen to try be a win-win: either an engineering win, or a win from the
point of view of having interesting evidence.

Output length
-------------
Varying the output duration is a way to test how temporally precise a model
is. The output duration represents the duration over which spikes will be
summed to calculate the "spike count" for the output interval. In addition,
varying output duration allows us to test our evaluation metrics. For example,
what is the relationship between accuracy, false positives, false negatives,
and any correlation measures. The first set of experiments are using
[1, 10, 50, 100] ms output durations. The 1 ms output duration is expected to
be too difficult to model, while the 100 ms output is expected to be easy.
So, these two extremes will act as a sanity check, helping to identify any
strange behaviour. The 10 ms output was chosen as I have seen it in other
literature, so if will be useful for comparison. After these 3 durations,
I wasn't sure what to choose. Given that it's the first set of experiments,
I'm not expecting amazing models, so periods greater than 10 ms might be
more useful for comparison than the more difficult shorter periods. I hope
we can get to the point where periods between 5 ms and 10 ms are interesting.
"""
all_configs = tuple(
    Configuration(*c)
    for c in [
        # 0.504 ms bins.
        #   1000.18 ms input.
        #        1.008 ms output
        (9, 1984, 2),
        #       10.82 ms output
        (9, 1984, 20),
        #       21.60 ms output
        (9, 1984, 100),
        #       50.41 ms output
        (9, 1984, 198),
        #       99.82 ms output
        #   1600.09 ms input.
        #        1.008 ms output
        (9, 3174, 2),
        #       10.82 ms output
        (9, 3174, 20),
        #       21.60 ms output
        (9, 3174, 100),
        #       50.41 ms output
        (9, 3174, 198),
        #       99.82 ms output
        # 1.001 ms bins.
        #   1000.18 ms input.
        #       5.04 ms output
        (18, 992, 5),
        #      10.08 ms output
        (18, 992, 10),
        #      20.16 ms output
        (18, 992, 20),
        #      40.32 ms output
        (18, 992, 40),
        #      80.64 ms output
        (18, 992, 80),
        #     161.29 ms output
        (18, 992, 160),
        #   1599.08 ms input.
        #        1.008 ms output
        (18, 1586, 1),
        #       10.08 ms output
        (18, 1586, 10),
        #       50.41 ms output
        (18, 1586, 50),
        #      100.82 ms output
        (18, 1586, 100),
        # Downsample by 89, giving 4.985 ms bins. At this rate, we can't
        # output 1 ms bins, so there are only 6 configurations for this
        # downsample factor.
        # 4.985 ms bins
        #    997.04 ms input
        #    Alternative is the closer 1.002 ms with 201 bins, but going with
        #    201 bins to try and keep the input/output bins even numbers.
        #       9.970 ms output
        (89, 200, 2),
        #       49.85 ms output
        (89, 200, 10),
        #       99.70 ms output
        (89, 200, 20),
        #   1595.27 ms input
        #   Alternative is the 1600.27 ms, with 321 bins, but going with 320
        #   bins to try and keep the input/output bins even numbers.
        #       9.970 ms output
        (89, 320, 2),
        #       49.85 ms output
        (89, 320, 10),
        #       99.70 ms output
        (89, 320, 20),
        # Downsample by 179, giving 10.026 ms bins. Same as with 89, we can't
        # output 1 ms bins, so there are only 6 configurations for this
        # downsample factor.
        # 10.026 ms bins
        #   1002.65 ms input
        #       10.037 ms output
        (179, 100, 1),
        #       20.053 ms output
        (179, 100, 2),
        #       50.132 ms output
        (179, 100, 5),
        #       10.037 ms output
        #   1604.22 ms input
        (179, 160, 1),
        #       20.053 ms output
        (179, 160, 2),
        #       50.132 ms output
        (179, 160, 5),
    ]
)

def ms_to_num_bins(time_ms, downsample_factor):
    res = time_ms * (mea.ELECTRODE_FREQ / 1000) / downsample_factor
    return res


def num_bins_to_ms(num_bins, downsample_factor):
    res = num_bins * downsample_factor / (mea.ELECTRODE_FREQ / 1000)
    return res


def get_configurations():
    res = []
    for downsample in downsample_factors:
        for in_len in input_lengths_ms:
            in_bins = ms_to_num_bins(in_len, downsample)
            for out_len in output_lenghts_ms:
                out_bins = ms_to_num_bins(out_len, downsample)
                if in_bins < 1 or out_bins < 1:
                    # Not enough resolution at this downsample factor.
                    continue
                in_bins_int = round(in_bins)
                out_bins_int = round(out_bins)
                res.append(Configuration(downsample, in_bins_int, out_bins_int))
    return res


class DistDataManager(retinapy.train.DataManager):
    """
    Interface for (train, val, test) BasicDistDatasets dataset.
    """

    def __init__(
        self,
        splits: Sequence[RecordingTrainValTest],
        input_len: int,
        output_len: int,
        downsample: int,
        dist_prefix_len: int,
        train_stride: int,
        use_augmentation: bool = False,
    ):
        """
        Args:
            splits: Iterable of 3 train, val, test splits.
        """
        # Promote the single split-tuple to a list of them.
        tr, v, ts = zip(*splits)
        self.train_recs = tr
        self.val_recs = v
        self.test_recs = ts
        self._input_len = input_len
        self._output_len = output_len
        self._downsample = downsample
        self._dist_prefix_len = dist_prefix_len
        self._train_stride = train_stride
        self._use_augmentation = use_augmentation

    def _to_ds(
        self,
        rec_parts: ContiguousChunks,
        stride: int,
        shuffle_stride: bool,
        use_augmentation: bool,
    ):
        res = retinapy.dataset.BasicDistDataset(
            rec_parts,
            input_len=self._input_len,
            output_len=self._output_len,
            pad=round(ms_to_num_bins(LOSS_CALC_PAD_MS, self._downsample)),
            dist_prefix_len=self._dist_prefix_len,
            dist_clamp=round(ms_to_num_bins(DIST_CLAMP_MS, self._downsample)),
            stride=stride,
            shuffle_stride=shuffle_stride,
            use_augmentation=use_augmentation,
        )
        return res

    @staticmethod
    def _single(recs: Sequence[ContiguousChunks]):
        # This manager currently only supports 1 recording.
        if len(recs) != 1:
            raise ValueError(
                "This manager only supports 1 recording." f"Got ({len(recs)})."
            )
        rec = recs[0]
        return rec

    def to_train_ds(self, recs: Sequence[ContiguousChunks]):
        rec_parts = self._single(recs)
        return self._to_ds(
            rec_parts, self._train_stride, True, self._use_augmentation
        )

    def to_val_ds(self, recs: Sequence[ContiguousChunks]):
        rec_parts = self._single(recs)
        return self._to_ds(rec_parts, 1, False, False)

    def to_test_ds(self, recs: Sequence[ContiguousChunks]):
        rec_parts = self._single(recs)
        return self._to_ds(rec_parts, 1, False, False)

    def train_ds(self) -> torch.utils.data.Dataset:
        return self.to_train_ds(self.train_recs)

    def val_ds(self) -> torch.utils.data.Dataset:
        return self.to_val_ds(self.val_recs)

    def test_ds(self) -> torch.utils.data.Dataset:
        return self.to_test_ds(self.test_recs)

    def single_cid_val_ds(self, cid: int) -> torch.utils.data.Dataset:
        assert (
            len(self.val_recs) == 1
        ), "This manager only supports 1 recording."
        rec_parts = [chunk.clusters({cid}) for chunk in self.val_recs[0]]
        return self._to_ds(rec_parts, 1, False, False)


class PoissonTrainable(retinapy.train.Trainable):
    def __init__(self, train_ds, val_ds, test_ds, model, model_label):
        super(PoissonTrainable, self).__init__(model, model_label)
        self._train_ds = train_ds
        self._val_ds = val_ds
        self._test_ds = test_ds
        self.loss_fn = torch.nn.PoissonNLLLoss(log_input=False)

    @property
    def in_device(self):
        res = next(self.model.parameters()).device
        return res

    @property
    def train_ds(self):
        return self._train_ds

    @property
    def val_ds(self):
        return self._val_ds

    @property
    def test_ds(self):
        return self._test_ds

    def forward(self, sample):
        X, y = sample
        X = X.float().to(self.in_device)
        y = y.float().to(self.in_device)
        model_output = self.model(X)
        loss = self.loss_fn(model_output, target=y)
        return model_output, loss

    def quick_forward(self, sample):
        X, _ = sample
        X = X.float().to(self.in_device)
        model_output = self.model(X)
        return model_output

    def evaluate_train(self, dl_fn):
        return self._evaluate(dl_fn(self.train_ds))

    def evaluate_val(self, dl_fn):
        return self._evaluate(dl_fn(self.val_ds))

    @property
    def sample_rate(self):
        sample_rates_equal = (
            self.train_ds.sample_rate
            == self.test_ds.sample_rate
            == self.val_ds.sample_rate
        )
        assert sample_rates_equal
        # We can use any of the datasets.
        return self.train_ds.sample_rate

    @property
    def sample_period_ms(self):
        return 1000 / self.sample_rate

    @property
    def num_input_bins(self):
        input_len_equals = (
            self.train_ds.input_len
            == self.test_ds.input_len
            == self.val_ds.input_len
        )
        assert input_len_equals
        # We can use any of the datasets.
        return self.train_ds.input_len

    @property
    def num_output_bins(self):
        output_len_equals = (
            self.train_ds.output_len
            == self.test_ds.output_len
            == self.val_ds.output_len
        )
        assert output_len_equals
        # We can use any of the datasets.
        return self.train_ds.output_len

    def _evaluate(self, dl):
        predictions = []
        targets = []
        # Note: the loss per batch is averaged, so we are averaging this again
        # as we loop through each batch.
        loss_meter = retinapy._logging.Meter("loss")
        for (X, y) in dl:
            X = X.float().cuda()
            y = y.float().cuda()
            model_output = self.model(X)
            loss_meter.update(
                self.loss_fn(model_output, target=y).item(), y.shape[0]
            )
            predictions.append(model_output.cpu())
            targets.append(y.cpu())
        # Don't forget to check if the model output is log(the Poisson λ parameter)
        # or not log!
        predictions = torch.round(torch.cat(predictions))
        targets = torch.cat(targets)
        acc = (predictions == targets).float().mean().item()
        pearson_corr = scipy.stats.pearsonr(predictions, targets)[0]
        results = {
            "metrics": [
                retinapy._logging.loss_metric(loss_meter.avg),
                retinapy._logging.Metric("accuracy", acc),
                retinapy._logging.Metric("pearson_corr", pearson_corr),
            ]
        }
        return results

    def model_summary(self, batch_size: int):
        dl = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size)
        X, y = next(iter(dl))
        X = X.float().to(self.in_device)
        # TODO: use torcheval
        res = torchinfo.summary(
            self.model,
            input_data=X,
            col_names=["input_size", "output_size", "mult_adds", "num_params"],
            device=self.in_device,
            depth=4,
        )
        return res


class DistTrainable_(retinapy.train.Trainable):
    """
    A base trainable for the distance models.

    There are quite a few things that are common to all such models, such as:
        - the conversion from distance array to model output (and vice versa)
        - the requirement to wrap properties like sample rate that would
          otherwise not be contained in one obvious place.
    """

    def __init__(
        self,
        ds_manager,
        model,
        model_label,
    ):
        super().__init__(model, model_label)
        self.ds_manager = ds_manager

    # @property
    @functools.cached_property
    def train_ds(self):
        return self.ds_manager.train_ds()

    @functools.cached_property
    def val_ds(self):
        return self.ds_manager.val_ds()

    @functools.cached_property
    def test_ds(self):
        return self.ds_manager.test_ds()

    @property
    def in_device(self):
        res = next(self.model.parameters()).device
        return res

    def ms_to_bins(self, ms: float) -> int:
        num_bins = max(1, round(ms * (self.sample_rate / 1000)))
        return num_bins

    @functools.cached_property
    def max_bin_dist(self):
        return self.ms_to_bins(DIST_CLAMP_MS)

    @functools.cached_property
    def sample_period_ms(self):
        return 1000 / self.sample_rate

    @functools.cached_property
    def sample_rate(self):
        sample_rates_equal = (
            self.train_ds.sample_rate
            == self.test_ds.sample_rate
            == self.val_ds.sample_rate
        )
        assert sample_rates_equal
        # We can use any of the datasets.
        return self.train_ds.sample_rate

    @staticmethod
    def dist_to_nn_output(dist):
        return torch.log(dist)

    @staticmethod
    def nn_output_to_dist(nn_output):
        return torch.exp(nn_output)


class DistVAETrainable(DistTrainable_):
    def __init__(
        self,
        train_ds: torch.utils.data.Dataset,
        val_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        rec_cluster_ids: mea.RecClusterIds,
        model: torch.nn.Module,
        model_label: str,
        eval_lengths_ms: Iterable[int],
        # clusters_to_eval: Iterable[int],
        vae_beta: float,
    ):
        """Trainable for a multi-cluster distance model.

        This is a minimal extension of the DistFieldTrainable.

        Args:
            rec_cluster_ids: a map for all supported rec-cluster pairs. This
                map should not contain any clusters for which the trainable
                is not expected to have in its embedding.

        """
        super().__init__(train_ds, val_ds, test_ds, model, model_label)
        self.rec_cluster_ids = rec_cluster_ids
        self.vae_beta = vae_beta

        self.dist_loss_fn = retinapy.models.dist_loss
        # Network output should ideally have mean,sd = (0, 1). Network output
        # 20*exp([-3, 3])  = [1.0, 402], which is a pretty good range, with
        # 20 being the mid point. Is this too low?
        self.dist_norm = 20
        # Network output should ideally have mean,sd = (0, 1). Network output
        # 20*exp([-3, 3])  = [1.0, 402], which is a pretty good range, with
        # 20 being the mid point. Is this too low?
        self.max_eval_count = int(2e5)

    def loss(self, m_dist, z_mu, z_lorvar, target):
        batch_size = m_dist.shape[0]
        # Scale to get roughly in the ballpark of 1.
        dist_loss = self.dist_loss_fn(m_dist, target)
        kl_loss = -0.5 * torch.sum(1 + z_lorvar - z_mu.pow(2) - z_lorvar.exp())
        dist_loss = dist_loss / batch_size
        # β = 1/1000
        β = self.vae_beta
        kl_loss = β * kl_loss / batch_size
        total = dist_loss + kl_loss
        return total, dist_loss, kl_loss

    def model_summary(self, batch_size: int):
        dl = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size)
        sample = next(iter(dl))
        masked_snippet = (
            sample["snippet"][:, 0 : mea.NUM_STIMULUS_LEDS + 1]
            .float()
            .to(self.in_device)
        )
        cluster_id = sample["cluster_id"].to(self.in_device)
        res = torchinfo.summary(
            self.model,
            input_data=[masked_snippet, cluster_id],
            col_names=["input_size", "output_size", "mult_adds", "num_params"],
            device=self.in_device,
            depth=4,
        )
        return res

    def encode(self, cluster_gids: torch.Tensor):
        """
        Return the latent representation of the given recording-cluster pairs.

        Args:
            rec_idxs: a batched tensor of recording indexes.
            cluster_idxs: a batch tensor of cluster indexes.
        """
        # We wrap the to-from device calls here, just for a trial.
        # Maybe we should make a separate non-sampling function, but it's fine
        # for now.
        in_device = cluster_gids.device
        cluster_gids = cluster_gids.long().to(self.in_device)
        _, z, _ = self.model.encode_vae(cluster_gids)
        return z.to(in_device)

    def all_encodings(
        self,
    ) -> Tuple[Iterable[str], Iterable[int], Iterable[int], torch.Tensor]:
        """
        Returns: tuple of lists, recording_names, cluster_ids,
            global_cluster_ids, encodings.
        """
        c_gids = list(self.rec_cluster_ids.values())
        rec_names, c_ids = zip(*list(self.rec_cluster_ids.keys()))
        encodings = self.encode(
            torch.tensor(c_gids, device=self.in_device).long()
        )
        return rec_names, c_ids, c_gids, encodings

    def _forward(self, sample):
        masked_snippet = sample["snippet"].float().to(self.in_device)
        dist = sample["dist"].float().to(self.in_device)
        cluster_id = sample["cluster_id"].int().to(self.in_device)
        m_dist, z_mu, z_logvar = self.model(masked_snippet, cluster_id)
        # Dist model
        y = self.dist_to_nn_output(dist)
        loss, dist_loss, kl_loss = self.loss(m_dist, z_mu, z_logvar, target=y)
        return m_dist, loss, dist_loss, kl_loss

    def forward(self, sample):
        m_dist, loss, _, _ = self._forward(sample)
        return m_dist, loss

    def infer(self, sample):
        """Minimal forward pass for inference."""
        masked_snippet = sample["snippet"].float().to(self.in_device)
        cluster_id = sample["cluster_id"].int().to(self.in_device)
        m_dist, _, _ = self.model(masked_snippet, cluster_id)
        return m_dist

    def evaluate_single(self, cluster_dl, results):
        loss_meter = retinapy._logging.Meter("loss")

        m_out = []
        lhs_spikes = []
        targets = []
        stimulus = []
        target_dist = []
        cluster_id = None
        # We separate the data -> model and inference so that we don't have
        # nested sub-processes.
        for i, sample in enumerate(cluster_dl):
            cluster_id = sample["cluster_id"].item()
            model_output, loss = self.forward(sample)
            loss_meter.update(loss.item())
            # Collect inputs for inference.
            stimulus.append(sample["stimulus"])
            target_spikes = sample["target_spikes"].float().cuda()
            targets.append(target_spikes)
            target_dist.append(self.nn_output_to_dist(sample["dist"]))
            m_out.append(model_output)
            lhs_spikes.append(lhs_spikes)

        infer_inputs = [
            torch.cat(stimulus),
            torch.cat(target_dist),
            torch.cat(m_out).clone(),
            torch.cat(targets),
            torch.cat(lhs_spikes),
        ]
        max_eval_bins = self.ms_to_bins(max(self.eval_lengths_ms))

        def eval_single(stimulus, dist, model_out, target_spikes, lhs_spike):
            out_spikes = sdf.predict(
                dist,
                lhs_spike,
                max_dist=self.max_bin_dist,
                dist_prefix_len=0,  # TODO: this is sub-optimal.
            )[:max_eval_bins]
            fig = retinapy.vis.dist_model_in_out(
                stimulus[0:-1].cpu().numpy(),
                stimulus[-1].cpu().numpy(),
                target_dist=dist.cpu().numpy(),
                model_out=model_out.cpu().numpy(),
                out_spikes=out_spikes.cpu().numpy(),
                bin_duration_ms=self.sample_period_ms,
            )
            fig_tensor = torch.tensor(
                retinapy._logging.plotly_fig_to_array(fig)
            )
            pred_eval_len = [
                torch.sum(out_spikes[0 : self.ms_to_bins(eval_len)])
                for eval_len in self.eval_lengths_ms
            ]
            target_eval_len = [
                torch.sum(target_spikes[0 : self.ms_to_bins(eval_len)])
                for eval_len in self.eval_lengths_ms
            ]
            return pred_eval_len, target_eval_len, fig_tensor

        with mp.Pool(processes=cluster_dl.num_workers) as pool:
            res = pool.imap(eval_single, infer_inputs)
            pool.close()
            pool.join()

        pred_count = []
        target_count = []
        frames = []
        for p, t, fig in res:
            pred_count.append(p)
            target_count.append(t)
            frames.append(fig)
        frames = torch.stack(frames)

        results["metrics"].append(
            retinapy._logging.Metric(
                f"loss-{cluster_id}", loss_meter.avg(), increasing=False
            )
        )
        results[f"model_io-{cluster_id}"] = retinapy._logging.PlotlyVideo(
            frames
        )

    def evaluate_train(self, dl_fn):
        self._evaluate(dl_fn(self.train_ds))

    def evaluate_val(self, dl_fn):
        self._evaluate(dl_fn(self.val_ds))

    def _evaluate(self, dl):
        predictions = defaultdict(list)
        targets = defaultdict(list)
        loss_meter = retinapy._logging.Meter("loss")
        kl_loss_meter = retinapy._logging.Meter("kl-loss")
        input_output_figs = []
        for i, sample in enumerate(dl):
            # Don't run out of memory, or take too long.
            num_so_far = dl.batch_size * i
            if num_so_far > self.max_eval_count:
                break
            target_spikes = sample["target_spikes"].float().cuda()
            model_output, loss, _, kl_loss = self._forward(sample)
            loss_meter.update(loss.item())
            kl_loss_meter.update(kl_loss.item())
            # Count accuracies
            for eval_len in self.eval_lengths_ms:
                eval_bins = self.ms_to_bins(eval_len)
                pred = self.quick_infer(model_output, num_bins=eval_bins)
                y = torch.sum(target_spikes[:, 0:eval_bins], dim=1)
                predictions[eval_len].append(pred)
                targets[eval_len].append(y)
            # Plot some example input-outputs
            if len(input_output_figs) < self.num_plots:
                # Plot the first batch element.
                idx = 0
                # Don't bother if there is no spike.
                contains_spike = torch.sum(target_spikes[0]) > 0
                if contains_spike:
                    c_gid = sample["cluster_id"][idx].item()
                    rec_name, c_id = self.rec_cluster_ids.inverse[c_gid]
                    title = f"rec: {rec_name} c_id: {c_id}"
                    input_output_figs.append(
                        self.input_output_fig(
                            sample["snippet"][idx],
                            sample["dist"][idx],
                            model_output[idx],
                            title,
                        )
                    )

        metrics = [
            retinapy._logging.Metric("loss", loss_meter.avg, increasing=False),
            retinapy._logging.Metric(
                "kl-loss", kl_loss_meter.avg, increasing=False
            ),
        ]
        for eval_len in self.eval_lengths_ms:
            p = torch.cat(predictions[eval_len])
            t = torch.cat(targets[eval_len])
            acc = (p == t).float().mean().item()
            pearson_corr = scipy.stats.pearsonr(
                p.cpu().numpy(), t.cpu().numpy()
            )[0]
            metrics.append(
                retinapy._logging.Metric(f"accuracy-{eval_len}_ms", acc)
            )
            metrics.append(
                retinapy._logging.Metric(
                    f"pearson_corr-{eval_len}_ms", pearson_corr
                )
            )
        results = {
            "metrics": metrics,
            "input-output-figs": retinapy._logging.PlotlyFigureList(
                input_output_figs
            ),
        }
        # Add the latent space visualization.
        rec_names, c_ids, _, zs = self.all_encodings()
        if self.model.z_dim == 2:
            latent_fig = retinapy.vis.latent2d_fig(
                rec_names, c_ids, zs[0].cpu().numpy(), zs[1].cpu().numpy()
            )
            results["latent-fig"] = retinapy._logging.PlotlyFigureList(
                [latent_fig]
            )
        # And try out Tensorboard's embedding feature.
        results["z-embeddings"] = retinapy._logging.Embeddings(
            embeddings=zs.cpu().numpy(),
            labels=[
                f"{r_name}-{int(c_id)}"
                for r_name, c_id in zip(rec_names, c_ids)
            ],
        )
        return results


class SingleDistTrainable(DistTrainable_):
    """Trainable for a distance model, for a single virtual cell."""

    DEFAULT_REFACTORY_MS = 2

    eval_modes = ["basic", "basic-quick", "detail", "detail-2"]
    loss_types = ["L1", "L2"]

    def __init__(
        self,
        ds_manager: DistDataManager,
        model,
        model_label,
        eval_mode="basic",
        # eval_mode="detail-2",
        loss_type="L2",
        output_mean: Optional[float] = None,
        output_sd: Optional[float] = None,
    ):
        """
        Args:
            ds_manager: A DatasetManager object. Currently, we haven't
                refactored any of the parent classes, so here is where we will
                create the datasets.
        """
        super().__init__(
            ds_manager,
            model,
            model_label,
        )
        if eval_mode not in self.eval_modes:
            raise ValueError(
                f"eval_mode must be one of {self.eval_modes}, got {eval_mode}"
            )
        self.eval_mode = eval_mode
        if loss_type not in self.loss_types:
            raise ValueError(
                f"loss_type must be one of {self.loss_types}, got {loss_type}"
            )
        self.dist_loss_fn = (
            retinapy.models.dist_loss_l1
            if loss_type == "L1"
            else retinapy.models.dist_loss
        )
        self.ds_manager = ds_manager
        self.refactory_len = self.ms_to_bins(self.DEFAULT_REFACTORY_MS)
        # Note: this probably should be done separately outside of init.
        self.init_norm(output_mean, output_sd)

    def init_norm(
        self,
        output_mean: Optional[float] = None,
        output_sd: Optional[float] = None,
    ):
        def calc_output_mean():
            train_recs = self.ds_manager.train_recs
            assert (
                len(train_recs) == 1
            ), "This model only supports 1 virtual cell."
            rec_parts = train_recs[0]
            assert (
                rec_parts[0].spikes.shape[1] == 1
            ), "This model only supports 1 virtual cell."
            # There can be multiple "parts" of a recording that are used
            # together as the training set. Calculate the distances for each
            # and concat the results.
            dist = np.concatenate(
                [
                    sdf.distance_arr(
                        rec_parts[i].spikes[:, 0],
                        default_distance=self.max_bin_dist,
                    )
                    for i in range(len(rec_parts))
                ]
            )
            nn_out = self.dist_to_nn_output(torch.tensor(dist))
            m, sd = torch.mean(nn_out).item(), torch.std(nn_out).item()
            return m, sd

        if output_mean is None or output_sd is None:
            output_mean, output_sd = calc_output_mean()
        # Two models supported:
        if hasattr(self.model, "set_output_mean_sd"):
            _logger.info(
                "Setting out (mean, sd) to: "
                f"({output_mean:.3f}, {output_sd:.3f})"
            )
            self.model.set_output_mean_sd(output_mean, output_sd)
        else:
            # Backward compatible.
            assert hasattr(self.model, "set_output_mean")
            _logger.info(f"Setting out mean to: ({output_mean:.3f})")
            self.model.set_output_mean(output_mean)
        self.model.set_input_mean_sd(
            # Bernoulli distribution with p=0.5 has var = 0.25, sd = 0.5
            torch.full((5,), 0.5),
            torch.tensor([0.5, 0.5, 0.5, 0.5, 1]),
        )

    def out_mean_sd(self):
        res = (self.model.output_mean.item(), self.model.output_scale.item())
        return res

    def loss(
        self, m_dist, t_dist
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            - m_dist: model output, log space
            - t_dist: target distance, linear, from sample
        """
        t_dist = self.dist_to_nn_output(t_dist)
        batch_size = m_dist.shape[0]
        batch_sum = self.dist_loss_fn(m_dist, t_dist)
        batch_ave = batch_sum / batch_size
        return batch_ave, {}

    def loss_lin(self, m_dist, t_dist):
        m_dist = self.nn_output_to_dist(m_dist)

        def to_loss_input(a):
            res = torch.log(a)
            return res

        t_dist = self.dist_to_nn_output(t_dist)
        m_dist = to_loss_input(m_dist)
        t_dist = to_loss_input(t_dist)
        batch_size = m_dist.shape[0]
        batch_sum = self.dist_loss_fn(m_dist, t_dist)
        batch_ave = batch_sum / batch_size
        return batch_ave

    def _forward(
        self, sample
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass, with loss calculation.

        Returns a tuple:
            - model output
            - total loss
            - named sub-losses
        """
        masked_snippet = sample["snippet"].float().cuda()
        tdist = sample["dist"].float().cuda()
        m_out = self.model(masked_snippet)
        total_loss, named_losses = self.loss(m_out, t_dist=tdist)
        return m_out, total_loss, named_losses

    def forward(self, sample):
        m_out, total_loss, _ = self._forward(sample)
        return m_out, total_loss

    def forward_no_loss(self, sample):
        """Slightly faster, if loss calc is not insignificant."""
        masked_snippet = sample["snippet"].float().cuda()
        model_output = self.model(masked_snippet)
        return model_output

    def model_summary(self, batch_size: int):
        dl = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size)
        sample = next(iter(dl))
        masked_snippet = sample["snippet"].float().cuda()
        res = torchinfo.summary(
            self.model,
            input_data=masked_snippet,
            col_names=["input_size", "output_size", "mult_adds", "num_params"],
            device=self.in_device,
            depth=4,
        )
        return res

    def evaluate_train(self, dl_fn):
        strided_ds = self.ds_manager.train_ds()
        loss_metrics = self.calc_loss(dl_fn(strided_ds))
        results = {"metrics": loss_metrics}
        return results

    def evaluate_val(self, dl_fn):
        if self.eval_mode == "basic":
            res = self.evaluate_val_basic(dl_fn)
        elif self.eval_mode == "basic-quick":
            res = self.evaluate_val_quick(dl_fn)
        elif self.eval_mode == "detail":
            res = self.evaluate_val_detail(dl_fn)
        elif self.eval_mode == "detail-2":
            res = self.evaluate_val_detail(dl_fn, full_metrics=True)
        else:
            raise ValueError(f"Unknown eval mode: {self.eval_mode}")
        return res

    def evaluate_val_basic(self, dl_fn):
        """
        Calculate loss only, using a short stride for higher precision.
        """
        strided_ds = self.ds_manager.val_ds()
        # A low stride.
        strided_ds.stride = 3
        loss_metrics = self.calc_loss(dl_fn(strided_ds))
        results = {"metrics": loss_metrics}
        return results

    def evaluate_val_quick(self, dl_fn):
        """
        Calculate loss only, using a long stride for quick results.
        """
        strided_ds = self.ds_manager.val_ds()
        # A long stride.
        strided_ds.stride = 80
        loss_metrics = self.calc_loss(dl_fn(strided_ds))
        results = {"metrics": loss_metrics}
        return results

    def evaluate_val_detail(self, dl_fn, full_metrics=False):
        """
        Calculate many metrics and produce figures.

        A long stride is used for quicker results, so this method is not
        suitable for model selection. Actually, two strides are used, one
        for the loss calc, and one for the other metrics and figures.
        There are two strides as the metrics' calculation requires a known
        stride in miliseconds and may be set longer than the stride for loss.
        """
        dl = dl_fn(self.val_ds)
        # A long stride for loss calc.
        strided_ds = self.ds_manager.val_ds()
        strided_ds.stride = 80
        loss_metrics = self.calc_loss(dl)
        # A reasonably long stride, set in terms of miliseconds for metrics.
        stride_ms = 80
        strided_ds.stride = self.ms_to_bins(stride_ms)
        # Copy the dl options from the dl_fn()'s output dl.
        metrics = loss_metrics
        if full_metrics:
            dist_spikes_tuple = _calc_output_arrays(
                self,
                strided_ds,
                dl.batch_size,
                self.max_bin_dist,
                self.refactory_len,
                dl.num_workers,
                dl.pin_memory,
            )
            metrics += detailed_metrics(
                *dist_spikes_tuple,
                bin_ms=self.sample_period_ms,
            )
        # A stride that is small enough to produce a smooth video.
        strided_ds.stride = 10
        # frames = [t, x,  t,  t,  t,  x,   t,   t,   x,   x]
        frames = [0, 43, 47, 48, 50, 99, 103, 106, 107, 108]
        fig_frames = self.create_io_frames(
            strided_ds,
            frames=frames,
            batch_size=dl.batch_size,
            # Making the video seems to be a bit more memory intensive per
            # worker. Or maybe there is a leak?
            num_workers=dl.num_workers - 3,
        )
        # video = retinapy._logging.PlotlyVideo(fig_frames)
        results = {
            "metrics": metrics,
            # "model-io": video,
            "model-io-frames": retinapy._logging.PlotlyFigureList(fig_frames),
        }
        return results

    def calc_loss(self, dl) -> List[retinapy._logging.Metric]:
        """
        Returns one or more loss metrics.

        The first loss metric must be the total loss.
        """
        loss_meter = retinapy._logging.Meter("loss")
        other_meters = {}
        it = iter(dl)
        _, loss, named_outputs = self._forward(next(it))
        loss_meter.update(loss.item())
        for name, output in named_outputs.items():
            other_meters[name] = retinapy._logging.Meter(name)
            other_meters[name].update(output.item())
        for sample in it:
            _, loss, named_outputs = self._forward(sample)
            loss_meter.update(loss.item())
            for name, output in named_outputs.items():
                other_meters[name].update(output.item())
        metrics = [retinapy._logging.loss_metric(loss_meter.avg)]
        for name, meter in other_meters.items():
            metrics.append(retinapy._logging.Metric(name, meter.avg))
        return metrics

    @torch.no_grad()
    def create_io_frames(
        self, ds, frames: Sequence[int], batch_size, num_workers
    ):
        """
        Args:
            batch_size: the batch size to use for the dataloader.
            num_workers: the number of workers to use for the dataloader.
        """

        class FrameDs(torch.utils.data.Dataset):
            def __init__(self, ds, frames):
                self.ds = ds
                self.frames = frames

            def __getitem__(self, idx):
                return self.ds[self.frames[idx]]

            def __len__(self):
                return len(self.frames)

        dl = torch.utils.data.DataLoader(
            FrameDs(ds, frames),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        m_out = []
        lhs_spikes = []
        target_dist = []
        target_spikes = []
        in_spikes = []

        for sample in dl:
            in_spikes.append(sample["snippet"][:, -1].float())
            model_output = self.forward_no_loss(sample)
            m_out.append(model_output)
            target_dist.append(self.dist_to_nn_output(sample["dist"].float()))
            target_spikes.append(sample["target_spikes"].float())
            lhs_spikes.append(
                sdf.lhs_spike(
                    sample["snippet"][:, -1].float(), self.max_bin_dist
                )
            )
        del dl

        infer_inputs = [
            torch.cat(in_spikes).cpu(),
            torch.cat(target_spikes).cpu(),
            torch.cat(target_dist).cpu(),
            torch.cat(m_out).cpu().clone(),
            torch.cat(lhs_spikes).cpu(),
        ]

        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            res_async = []
            for i in range(len(frames)):
                res_async.append(
                    pool.apply_async(
                        io_frame_fn,
                        args=(
                            self,
                            ds.dist_prefix_len,
                            infer_inputs[0][i],
                            infer_inputs[1][i],
                            infer_inputs[2][i],
                            infer_inputs[3][i],
                            infer_inputs[4][i].item(),
                            f"frame: {i:04d}",
                        ),
                    )
                )
            figs = [r.get() for r in res_async]
            pool.close()
            pool.join()
        return figs


@torch.no_grad()
def _calc_output_arrays(
    trainable,
    ds,
    batch_size,
    max_bin_dist,
    refactory_len,
    num_workers,
    pin_memory,
):
    """
    Concats model output and inferred spikes from multiple forward() calls.

    Each distance array and spike prediction array will be clipped to
    a length equal to the dataset's stride, so that concatenation maintains
    the integrity of the time axis.

    Currently, the distance array is clipped as [0:ds.stride], but it may
    be preferable to clip like:

        [ds.dist_prefix_len:ds.dist_prefix_len + ds.stride]

    """
    has_dist_to_out_fn = hasattr(trainable, "dist_to_nn_output") and callable(
        trainable.dist_to_nn_output
    )
    has_forward_no_loss_fn = hasattr(trainable, "forward_no_loss") and callable(
        trainable.forward_no_loss
    )
    if not has_dist_to_out_fn:
        raise ValueError(
            "The trainable must have a dist_to_nn_output() method."
        )
    if not has_forward_no_loss_fn:
        raise ValueError("The trainable must have a forward_no_loss() method.")
    only_one_cid = ds.recording.num_clusters() == 1
    if not only_one_cid:
        raise ValueError(
            "Only 1 cluster is supported. The recording had "
            f"({ds.recording.num_clusters()})."
        )
    dist_actual = []
    dist_pred = []
    lhs_spikes = []
    spikes_actual = []
    stride = ds.stride
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    for i, sample in enumerate(dl):
        actual_dist = trainable.dist_to_nn_output(sample["dist"].float())
        dist_actual.append(actual_dist)
        spikes_actual.append(sample["target_spikes"].float())
        model_output = trainable.forward_no_loss(sample).cpu()
        dist_pred.append(model_output)
        lhs_spikes.append(
            sdf.lhs_spike(sample["snippet"][:, -1].float(), max_bin_dist)
        )
    del dl
    # Convert to 2D array [N, len]
    dist_actual = torch.cat(dist_actual, dim=0)
    dist_pred = torch.cat(dist_pred, dim=0)
    lhs_spikes = torch.cat(lhs_spikes, dim=0)
    spikes_actual = torch.cat(spikes_actual, dim=0)

    def infer_fn(i):
        s_pred = sdf.predict(
            dist_pred[i],
            int(lhs_spikes[i].item()),
            max_bin_dist,
            dist_prefix_len=ds.dist_prefix_len,
            refactory=refactory_len,
        )[:stride]
        return s_pred

    # I couldn't get the multiprocessing to work, so just use plain
    # python.
    spike_pred_len = len(dist_pred[0]) - ds.dist_prefix_len
    if spike_pred_len > 0:
        spikes_pred = list(map(infer_fn, range(len(dist_pred))))
    else:
        spikes_pred = [torch.zeros(size=(stride,))]

    # Multiprocessing version
    # =======================
    # infer_fn = functools.partial(
    #     _forward_for_eval_infer_fn,
    #     self.max_bin_dist,
    #     stride,
    # )
    # with mp.Pool(processes=num_workers) as pool:
    #     spikes_pred = list(pool.map(
    #         infer_fn,
    #         [(dist_pred[i].clone(), lhs_spikes[i]) for i in range(len(dist_pred))],
    #     ))
    #     pool.close()
    #     pool.join()

    # Convert [N, len] to [N, stride], then flatten.
    dist_actual = dist_actual[:, 0:stride].flatten()
    dist_pred = dist_pred[:, 0:stride].flatten()
    spikes_actual = spikes_actual[:, 0:stride].flatten()
    spikes_pred = torch.cat(spikes_pred, dim=0)
    return dist_actual, dist_pred, spikes_actual, spikes_pred


class MultiDistTrainable(DistTrainable_):
    """
    Multi-cell version of SingleDistTrainable.

    If it can generalize easily, then SingleDistTrainable can be removed.
    """

    DEFAULT_REFACTORY_MS = 2

    eval_modes = ["basic-quick", "basic", "detail"]

    def __init__(
        self,
        ds_manager: DistDataManager,
        model,
        model_label,
        eval_mode="detail",
        output_mean: Optional[float] = None,
        output_sd: Optional[float] = None,
        clusters_to_eval: Optional[Iterable[int]] = None,
    ):
        super().__init__(
            ds_manager,
            model,
            model_label,
        )
        self.clusters_to_eval = clusters_to_eval if clusters_to_eval else []
        if eval_mode not in self.eval_modes:
            raise ValueError(
                f"eval_mode must be one of {self.eval_modes}, got {eval_mode}"
            )
        self.eval_mode = eval_mode
        self.ds_manager = ds_manager
        self.refactory_len = self.ms_to_bins(self.DEFAULT_REFACTORY_MS)
        # Note: this probably should be done separately outside of init.
        self.init_norm(output_mean, output_sd)

    def init_norm(
        self,
        output_mean: Optional[float] = None,
        output_sd: Optional[float] = None,
    ):
        def estimate_output_mean():
            train_recs = self.ds_manager.train_recs
            # Only use the first recording (hence estimate)
            def to_dist(spikes):
                res = np.log(sdf.distance_arr(spikes, self.max_bin_dist))
                return res

            dists = np.apply_along_axis(to_dist, 0, train_recs[0][0].spikes)
            m, sd = np.mean(dists), np.std(dists)
            return m, sd

        if output_mean is None or output_sd is None:
            mean_est, sd_est = estimate_output_mean()
            output_mean = output_mean if output_mean is not None else mean_est
            output_sd = output_sd if output_sd is not None else sd_est
        _logger.info(
            f"Setting out (mean, sd) to: ({output_mean:.3f}, {output_sd:.3f})"
        )
        self.model.set_output_mean_sd(output_mean, output_sd)
        self.model.set_input_mean_sd(
            # Bernoulli distribution with p=0.5 has var = 0.25, sd = 0.5
            torch.full((5,), 0.5),
            torch.tensor([0.5, 0.5, 0.5, 0.5, 1]),
        )

    def loss(
        self, m_dist, t_dist
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            - m_dist: model output, log space
            - t_dist: target distance, linear, from sample
        """
        t_dist = self.dist_to_nn_output(t_dist)
        batch_size = m_dist.shape[0]
        batch_sum = retinapy.models.dist_loss_l1(m_dist, t_dist)
        batch_ave = batch_sum / batch_size
        return batch_ave, {}

    def _forward(
        self, sample
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        masked_snippet = sample["snippet"].float().cuda()
        cid = sample["cluster_id"].float().cuda()
        tdist = sample["dist"].float().cuda()
        m_out = self.model(masked_snippet, cid)
        total_loss, named_losses = self.loss(m_out, t_dist=tdist)
        return m_out, total_loss, named_losses

    def forward_no_loss(self, sample):
        """Slightly faster, if loss calc is not insignificant."""
        masked_snippet = sample["snippet"].float().cuda()
        cid = sample["cluster_id"].float().cuda()
        model_output = self.model(masked_snippet, cid)
        return model_output

    def forward(self, sample):
        m_out, total_loss, _ = self._forward(sample)
        return m_out, total_loss

    def model_summary(self, batch_size: int):
        dl = torch.utils.data.DataLoader(self.train_ds, batch_size=batch_size)
        sample = next(iter(dl))
        masked_snippet = sample["snippet"].float().cuda()
        cid = sample["cluster_id"].float().cuda()
        res = torchinfo.summary(
            self.model,
            input_data=(masked_snippet, cid),
            col_names=["input_size", "output_size", "mult_adds", "num_params"],
            device=self.in_device,
            depth=4,
        )
        return res

    def evaluate_train(self, dl_fn):
        strided_ds = self.ds_manager.train_ds()
        loss_metrics = self.calc_loss(dl_fn(strided_ds))
        results = {"metrics": loss_metrics}
        return results

    def evaluate_val(self, dl_fn):
        if self.eval_mode == "basic":
            res = self.evaluate_val_basic(dl_fn)
        elif self.eval_mode == "basic-quick":
            res = self.evaluate_val_quick(dl_fn)
        elif self.eval_mode == "detail":
            res = self.evaluate_val_detail(dl_fn)
        else:
            raise ValueError(f"Unknown eval mode: {self.eval_mode}")
        return res

    def evaluate_val_basic(self, dl_fn):
        """
        Calculate loss only, using a short stride for higher precision.
        """
        strided_ds = self.ds_manager.val_ds()
        # A low stride.
        strided_ds.stride = 3
        loss_metrics = self.calc_loss(dl_fn(strided_ds))
        results = {"metrics": loss_metrics}
        return results

    def evaluate_val_quick(self, dl_fn):
        """
        Calculate loss only, using a long stride for quick results.
        """
        strided_ds = self.ds_manager.val_ds()
        # A long stride.
        strided_ds.stride = 1024
        loss_metrics = self.calc_loss(dl_fn(strided_ds))
        results = {"metrics": loss_metrics}
        return results

    def evaluate_val_detail(self, dl_fn):
        """
        Calculate many metrics and produce figures.

        A long stride is used for quicker results, so this method is not
        suitable for model selection. Actually, two strides are used, one
        for the loss calc, and one for the other metrics and figures.
        There are two strides as the metrics' calculation requires a known
        stride in milliseconds and may be set longer than the stride for loss.
        """
        # 1. Loss metrics.
        # A long stride for loss calc.
        strided_ds = self.ds_manager.val_ds()
        strided_ds.stride = 2000
        dl = dl_fn(strided_ds)
        results = {"metrics": self.calc_loss(dl)}
        # 1. IO frame images.
        for cid in self.clusters_to_eval:
            cid_ds = self.ds_manager.single_cid_val_ds(cid)
            # 1. IO frames
            cid_ds.stride = 50
            num_frames = 7
            fig_frames = self.create_io_frames(
                cid_ds,
                frames=num_frames,
                batch_size=dl.batch_size,
                # Making the video seems to be a bit more memory intensive per
                # worker. Or maybe there is a leak?
                num_workers=dl.num_workers - 3,
            )
            results[f"c{cid}/frames"] = retinapy._logging.PlotlyFigureList(
                fig_frames
            )
            # 2. Spike train stats.
            # A reasonably long stride, set in terms of miliseconds for metrics.
            stride_ms = 80
            cid_ds.stride = self.ms_to_bins(stride_ms)
            # Copy the dl options from the dl_fn()'s output dl.
            dist_spikes_tuple = _calc_output_arrays(
                self,
                cid_ds,
                dl.batch_size,
                self.max_bin_dist,
                self.refactory_len,
                dl.num_workers,
                dl.pin_memory,
            )
            results["metrics"] += detailed_metrics(
                *dist_spikes_tuple,
                bin_ms=self.sample_period_ms,
                label_prefix=f"c{cid}/",
            )
        return results

    def calc_loss(self, dl) -> List[retinapy._logging.Metric]:
        """
        Returns one or more loss metrics.

        The first loss metric must be the total loss.
        """
        loss_meter = retinapy._logging.Meter("loss")
        other_meters = {}
        it = iter(dl)
        _, loss, named_outputs = self._forward(next(it))
        loss_meter.update(loss.item())
        for name, output in named_outputs.items():
            other_meters[name] = retinapy._logging.Meter(name)
            other_meters[name].update(output.item())
        for sample in it:
            _, loss, named_outputs = self._forward(sample)
            loss_meter.update(loss.item())
            for name, output in named_outputs.items():
                other_meters[name].update(output.item())
        metrics = [retinapy._logging.loss_metric(loss.item())]
        for name, meter in other_meters.items():
            metrics.append(retinapy._logging.Metric(name, meter.avg))
        return metrics

    @torch.no_grad()
    def create_io_frames(
        self, ds, frames: Union[int, Sequence[int]], batch_size, num_workers
    ):
        """
        Args:
            frames: either specific frames to be used, or the number of frames
                desired, chosen at this function's discretion.
            batch_size: the batch size to use for the dataloader.
            num_workers: the number of workers to use for the dataloader.
        """

        class FrameDs(torch.utils.data.Dataset):
            def __init__(self, ds, frames):
                self.ds = ds
                self.frames = frames

            def __getitem__(self, idx):
                return self.ds[self.frames[idx]]

            def __len__(self):
                return len(self.frames)

        def skip_blank_ds(ds, num_frames):
            res = []
            for sample in ds:
                if len(res) >= num_frames:
                    break
                elif sample["target_spikes"].sum() == 0:
                    continue
                else:
                    res.append(sample)
            return res

        if type(frames) is int:
            ds_wrapper = skip_blank_ds(ds, frames)
        else:
            ds_wrapper = FrameDs(ds, frames)
        num_frames = len(ds_wrapper)

        dl = torch.utils.data.DataLoader(
            ds_wrapper,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        m_out = []
        lhs_spikes = []
        target_dist = []
        target_spikes = []
        in_spikes = []

        for sample in dl:
            in_spikes.append(sample["snippet"][:, -1].float())
            model_output = self.forward_no_loss(sample)
            m_out.append(model_output)
            target_dist.append(self.dist_to_nn_output(sample["dist"].float()))
            target_spikes.append(sample["target_spikes"].float())
            lhs_spikes.append(
                sdf.lhs_spike(
                    sample["snippet"][:, -1].float(), self.max_bin_dist
                )
            )
        del dl

        infer_inputs = [
            torch.cat(in_spikes).cpu(),
            torch.cat(target_spikes).cpu(),
            torch.cat(target_dist).cpu(),
            torch.cat(m_out).cpu().clone(),
            torch.cat(lhs_spikes).cpu(),
        ]

        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            res_async = []
            for i in range(num_frames):
                res_async.append(
                    pool.apply_async(
                        io_frame_fn,
                        args=(
                            self,
                            ds.dist_prefix_len,
                            infer_inputs[0][i],
                            infer_inputs[1][i],
                            infer_inputs[2][i],
                            infer_inputs[3][i],
                            infer_inputs[4][i].item(),
                            f"frame: {i:04d}",
                        ),
                    )
                )
            figs = [r.get() for r in res_async]
            pool.close()
            pool.join()
        return figs


# Python requires functions to be module scope in order to pickle, which is
# a requirement for multiprocessing.
def io_frame_fn(
    obj: SingleDistTrainable,
    dist_prefix_len: int,
    in_spikes,
    target_spikes,
    target_dist,
    model_out,
    lhs_spike: int,
    title: str,
) -> plotly.graph_objects.Figure:
    """
    Function to be run by pool.
    """
    pred_len = len(target_dist) - dist_prefix_len
    if pred_len > 0:
        pred_spikes = sdf.predict(
            # Don't forget, the prediction function takes the normal linear
            # distance.
            obj.nn_output_to_dist(model_out),
            lhs_spike,
            max_dist=obj.max_bin_dist,
            dist_prefix_len=dist_prefix_len,
            refactory=obj.refactory_len,
        )
    else:
        # Zero spike prediction lenght may occur if we are only interested in
        # predicting past distance field, which is something that may be done for
        # pretraining.
        pred_spikes = torch.tensor([])

    # fig = retinapy.vis.dist_model_io_slim(
    #     stimulus=model_in[0:-1].cpu().numpy(),
    #     in_spikes=model_in[-1].cpu().numpy(),
    #     target_dist=target_dist.cpu().numpy(),
    #     model_out=model_out.cpu().numpy(),
    #     out_spikes=pred_spikes.cpu().numpy(),
    #     bin_duration_ms=obj.sample_period_ms,
    # )
    fig = retinapy.vis.dist_model_out(
        in_spikes=in_spikes.cpu().numpy(),
        target_spikes=target_spikes.cpu().numpy(),
        target_dist=target_dist.cpu().numpy(),
        model_out=model_out.cpu().numpy(),
        pred_spikes=pred_spikes.cpu().numpy(),
        x_start_ms=-200,
        dist_prefix_len=dist_prefix_len,
        bin_duration_ms=obj.sample_period_ms,
        title=title,
    )

    return fig


def detailed_metrics(
    dist_actual,
    dist_pred,
    spikes_actual,
    spikes_pred,
    bin_ms,
    label_prefix=None,
) -> List[retinapy._logging.Metric]:
    """
    Calculates metrics related to the model's output.

    Returns a dictionary that can be merged with other results.

    The results include:
        - Pearson correlation for various degrees of smoothing.
        - Schreiber correlation for various degrees of smoothing.
        - The output distance array is split into 3 chunks
          (beginning, middle, end) then calculating Pearson correlation
          (no smoothing, as we are dealing with the distance). This metric
          is aimed at trying to see if the model is better or worse at
          predicting the distance at different parts of the output.
        - A video comparing actual and predicted distance.

    Args:
        ds: the dataset to operate over. The stride of the dataset must
            be set before calling this method.
        batch_size: passed on to forward() call. Used to create a
            dataloader.
        num_workers: passed on to forward() call. Used to create
            a dataloader.
        pin_memory: passed on to forward() call. Used to create
            a dataloader.
    """
    label_prefix = label_prefix if label_prefix else ""
    N_CHUNK = 3
    df = pl.DataFrame(
        data=[
            (
                dist_actual.tolist(),
                dist_pred.tolist(),
                spikes_actual.tolist(),
                spikes_pred.tolist(),
            )
        ],
        schema=["d_actual", "d_pred", "s_actual", "s_pred"],
    )

    def van_rossum(row, τ_ms):
        return retinapy.metrics.van_rossum(
            row["s_actual"], row["s_pred"], bin_ms, τ_ms
        )

    def pcorr(row, num_bins):
        return retinapy.metrics.binned_pcorr(
            row["s_actual"], row["s_pred"], num_bins
        )

    def schreiber(row, σ_ms):
        return retinapy.metrics.schreiber(
            row["s_actual"], row["s_pred"], bin_ms, σ_ms
        )

    def chuck_pcorr(row, N, i):
        chunk_actual = np.array_split(row["d_actual"], N)[i]
        chunk_pred = np.array_split(row["d_pred"], N)[i]
        res = retinapy.metrics.pcorr(chunk_actual, chunk_pred)
        return res

    stats_df = df.with_columns(
        [
            # Distance correlation. Should this be done in stats fn?
            pl.struct(["d_actual", "d_pred"])
            .apply(lambda x: retinapy.metrics.pcorr(x["d_actual"], x["d_pred"]))
            .alias("distf_pcorr"),
            # For seeing if there is change over time.
            # Chunked distance correlation.
            *(
                pl.struct(["d_actual", "d_pred"])
                .apply(functools.partial(chuck_pcorr, N=N_CHUNK, i=i))
                .alias(f"chunk_distf_pcorr-chunk{i}of{N_CHUNK}")
                for i in range(N_CHUNK)
            ),
            # MSE
            pl.struct(["d_actual", "d_pred"])
            .apply(lambda x: retinapy.metrics.mse(x["d_actual"], x["d_pred"]))
            .alias("distf_mse"),
            pl.struct(["s_actual", "s_pred"])
            .apply(lambda x: retinapy.metrics.pcorr(x["s_actual"], x["s_pred"]))
            .alias(f"pcorr-{round(bin_ms)}_ms"),
            *(
                pl.struct(["s_actual", "s_pred"])
                .apply(functools.partial(pcorr, num_bins=b))
                .alias(f"pcorr-σ{b}_ms")
                for b in [round(s / bin_ms) for s in (5, 10, 20, 40, 80)]
            ),
            *(
                pl.struct(["s_actual", "s_pred"])
                .apply(functools.partial(schreiber, σ_ms=s))
                .alias(f"schreiber-σ{s}_ms")
                for s in (1, 2, 5, 10)
            ),
            *(
                pl.struct(["s_actual", "s_pred"])
                .apply(functools.partial(van_rossum, τ_ms=s))
                .alias(f"vrossum-τ{s}_ms")
                for s in (25, 50, 100, 200)
            ),
        ]
    ).select(pl.all().exclude(["s_actual", "s_pred", "d_actual", "d_pred"]))
    stats = stats_df.to_dicts()
    assert len(stats) == 1, "Only 1 row!"
    metrics = [
        retinapy._logging.Metric(f"{label_prefix}{k}", v)
        for k, v in stats[0].items()
    ]
    return metrics


# def _forward_for_eval_infer_fn(
#     max_bin_dist,
#     stride,
#     args,
# ):
#     dist_pred, lhs_spike = args
#     s_pred = sdf.predict(
#         dist_pred,
#         lhs_spike,
#         max_bin_dist,
#         stride,
#         refactory=2,
#     )
#     return s_pred


class PixelCnnTrainable(DistTrainable_):
    """
    Causal recurrence, with distance objective.

    In retrospect, I don't think this is going to be a good direction without
    a lot of changes to the vanilla approach. While training, access to the
    distance array of the previous step makes a lot of decisions easy: just
    follow the down or up direction that has already been set. The decision
    to have a spike is made by the first snippet that departs from the
    already set course. This is concentrating the decision and not allowing
    the network to express "definitely a spake in this broad area" which
    is the whole point of the distance array in the first place.

    The PixelCnn approach may work with hierarchical zoom approach, as
    there is pressure to adhere to the estimate of a broader area.

    Autoregressive approach suffer as the spike history is vital for
    prediction. The history is also quite accurate. So models will develop
    a dependency on this precise input. The model's output is not accurate,
    yet the autoregressive approach does not allow the model to know it is
    now working with predictions rather than ground truth. To ameliorate this
    issue, we can start distorting the spike history, ideally to just the
    same level as the inaccuracy we expect in our prediction. However, then
    we don't actually allow ourselves to communicate strong ground-truths
    to the model.
    """

    NUM_INFER = 8

    def __init__(
        self,
        train_ds,
        val_ds,
        test_ds,
        model,
        model_label,
        input_len,
        eval_lengths_ms=None,
    ):
        super().__init__(
            train_ds, val_ds, test_ds, model, model_label, eval_lengths_ms
        )
        self.input_len = input_len

    def loss(self, m_dist, target):
        batch_size = m_dist.shape[0]
        batch_sum = retinapy.models.dist_loss(m_dist, target)
        batch_ave = batch_sum / batch_size
        return batch_ave

    def forward(self, sample):
        snippet = sample["snippet"].float().to(self.in_device)
        snippet[:, -1, :] = self.dist_to_nn_output(snippet[:, -1, :])
        m_out = self.model(snippet)
        loss = self.loss(m_out, target=snippet[:, -1, :])
        return m_out, loss

    def infer_dist(self, sample):
        b, c, l = sample["snippet"].shape
        assert c == 5, "Should have 4 stimuli and 1 distance array."
        # res.shape = (b, l)
        # Needed? Don't want to change it from the caller.
        snippet = sample["snippet"].clone().float().to(self.in_device)
        snippet[:, -1, :] = self.dist_to_nn_output(snippet[:, -1, :])
        # Mask area to be generated, just to make sure no leakage.
        snippet[:, -1, self.input_len :] = 0
        for p in range(self.input_len, l):
            m_out = self.model(snippet)
            snippet[:, -1, p] = m_out[:, p]
        out_distf = snippet[:, -1, self.input_len :]
        return out_distf

    def evaluate(self, val_dl):
        loss_meter = retinapy._logging.Meter("loss")
        for i, sample in enumerate(val_dl):
            _, loss = self.forward(sample)
            loss_meter.update(loss.item())

        # Full predictions
        num_figs_todo = self.NUM_INFER
        plotly_figs = []
        for i, sample in enumerate(val_dl):
            if num_figs_todo <= 0:
                break
            distf = self.infer_dist(sample)
            batch_len, l = distf.shape
            for e in range(min(batch_len, num_figs_todo)):
                snippet = sample["snippet"][e]
                stimulus = snippet[0:4]
                dist_actual = self.dist_to_nn_output(snippet[-1])
                fig = retinapy.vis.pixelcnn_model_in_out(
                    stimulus.cpu().numpy(),
                    dist_actual.cpu().numpy(),
                    distf[e].cpu().numpy(),
                    DIST_CLAMP_MS,
                )
                plotly_figs.append(fig)
                num_figs_todo -= 1

        metrics = [
            retinapy._logging.Metric("loss", loss_meter.avg, increasing=False)
        ]
        results = {
            "metrics": metrics,
            "input-output-figs": retinapy._logging.PlotlyFigureList(
                plotly_figs
            ),
        }
        return results


def create_multi_cluster_df_datasets(
    splits: Iterable[RecordingTrainValTest],
    input_len: int,
    output_len: int,
    downsample: int,
    stride: int,
):
    """
    Args:
        splits: A list of splits, one train-val-test split per recording.
            Each train-val-test split is a length 3 list, where the elements
            are themselves lists containing the contiguous chunks that are
            taken from various parts of a recording.
    """
    snippet_len = input_len + output_len
    def to_ds(segments, augmentation: bool):
        parts = [
            retinapy.dataset.DistDataset(
                s,
                snippet_len=snippet_len,
                mask_begin=input_len,
                mask_end=snippet_len,
                pad=round(ms_to_num_bins(LOSS_CALC_PAD_MS, downsample)),
                dist_clamp=round(ms_to_num_bins(DIST_CLAMP_MS, downsample)),
                stride=stride,
                enable_augmentation=augmentation,
                allow_cheating=False,
            )
            for s in segments
        ]
        ds = retinapy.dataset.ConcatDistDataset(parts)
        return ds
    # Collect all segments across all recordings and group into train, val and 
    # test.
    train_segments, val_segments, test_segments = (
            [sum(segs, []) for segs in zip(*splits)]
    )
    res = tuple(
        to_ds(split, aug) for split, aug in zip([train_segments, val_segments, test_segments], [True, True, False])
    )
    return res


def recording_splits(
    recordings: Iterable[mea.CompressedSpikeRecording],
    downsample: int,
    num_workers: int,
    min_spikes: tuple[int, int, int] = (0, 0, 0),
) -> List[RecordingTrainValTest]:
    """Create N train/val/test splits from a list of N recordings.

    This function might end up returning a generator if it has to be chained
    with further processing or filtering. The list return type is not
    appropriate in that case, as it will likely consume excess memory and
    prevent multiprocessing to operate on a whole chain of calls; prefer a
    generator of futures, or a queue. Currently, we are getting away fine with
    just a list return type. There are still quite a few calls to this function
    that wrap the return type in a list, as it used to return a generator.
    Keeping them like that as it's not unlikely to change again.

    Args:
        recordings: List of recordings from which to create splits.
        downsample: Factor to downsample by.
        num_workers: Max number of thread workers to use.
        min_spikes: 3 spike count minimums, one for each split. Any cluster
            with insufficient spikes in any of the three splits will be removed.

    Returns:
        List of N train/val/test splits.
    """
    # sum(min_spikes) is a lower bound number of spikes a cluster will need
    # in order to produce a (train, val, test) split with at least
    # (min_spikes[0], min_spikes[1], min_spikes[2]) spikes respectively.
    min_count = sum(min_spikes)
    recordings = (
        r.filter_clusters(max_rate=MAX_SPIKE_RATE, min_count=min_count)
        for r in recordings
    )

    def run(comp_rec):
        rec = mea.decompress_recording(comp_rec, downsample)
        train_val_test_splits = mea.mirror_split2(rec, split_ratio=SPLIT_RATIO)
        train_val_test_splits = mea.remove_few_spike_clusters(
            train_val_test_splits, min_spikes
        )
        return train_val_test_splits

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        # res = list(executor.map(run, recordings))
        fns = [
            functools.partial(mea.decompress_recording, downsample=downsample),
            functools.partial(mea.mirror_split2, split_ratio=SPLIT_RATIO),
            functools.partial(
                mea.remove_few_spike_clusters, min_counts=min_spikes
            ),
        ]
        data = recordings
        for fn in fns:
            data = executor.map(fn, data)
        res = list(data)
    return res


def get_ds_split(
    data_dir,
    rec_name,
    downsample: int,
    num_workers: int,
    clusters: Optional[Set[int]] = None,
):
    """
    Get the (train, val, test) split for a single recording.

    Indexing the splits is a bit opaque, so this common case is encapsulated
    here to avoid mistakes.
    """
    rec0 = mea.single_3brain_recording(
        rec_name, data_dir, include_clusters=clusters
    )
    rec_ds_split = recording_splits(
        [rec0], downsample=downsample, num_workers=num_workers
    )[0]
    return rec_ds_split


def single_cluster_split(rec_split, cluster_id):
    """
    Extract the (train, val, test) for a single cell.

    To reduce mistakes, this common indexing pattern is encapsulated here.
    """
    train, val, test = [
        [part.clusters({cluster_id}) for part in split] for split in rec_split
    ]
    return (train, val, test)


@deprecated.deprecated(reason="Use a dataset manager instead.")
def create_distfield_datasets(
    train_val_test_splits: RecordingTrainValTest,
    input_len: int,
    output_len: int,
    downsample: int,
    stride: int = 1,
    use_augmentation: bool = True,
):
    snippet_len = input_len + output_len
    train_val_test_datasets = [
        retinapy.dataset.DistDataset(
            r,
            snippet_len=snippet_len,
            mask_begin=input_len,
            mask_end=snippet_len,
            pad=round(ms_to_num_bins(LOSS_CALC_PAD_MS, downsample)),
            dist_clamp=round(ms_to_num_bins(DIST_CLAMP_MS, downsample)),
            stride=stride,
            enable_augmentation=_use_aug,
            allow_cheating=False,
        )
        for (r, _use_aug) in zip(
            train_val_test_splits, [use_augmentation, False, False]
        )
    ]
    return train_val_test_datasets


def create_count_datasets(
    train_val_test_splits: RecordingTrainValTest,
    input_len: int,
    output_len: int,
    stride: int = 1,
):
    """
    Creates the spike count datasets for the given recording data.

    Elements (X, y) are produced, with semantics
    (stimulus-spike history, spike count).


    X shape
    -------
    X's shape depends on whether `cid_as_dim` is True or False: if False, X has
    shape (5, input_len); if True X has shape (num_clusters, 5, input_len). y
    is an integer scalar. The 5 channels comprise of 4 stimulus channels and
    1 channel for spike history.

    y shape
    -------
    y is a single integer. It is the number of spikes in `output_len` number
    of bins.

    Three datasets are returned: train, validation and test.

    These datasets take the form:
        X,y = (stimulus-spike history, num spikes)

    The length of the input history, the output binning duration and the
    downsample rate can be configured.

    Args:
        train_val_test_splits: A triplet of recordings.
        input_len: the number of bins to use for X

    """
    train_val_test_datasets = [
        retinapy.dataset.SpikeCountDataset(
            r,
            input_len=input_len,
            output_len=output_len,
            stride=stride,
        )
        for r in train_val_test_splits
    ]
    return train_val_test_datasets


class TrainableGroup:
    """Not sure if the methods will be made static or not. Currently, there
    is one use of this methods being called on an instance; however, it's
    an edge case that should probably be handled in another way."""
    def trainable_label(self, config):
        raise NotImplementedError

    def create_trainable(
        self,
        splits: Iterable[RecordingTrainValTest],
        config: Optional[Configuration] = None,
        opt=None,
    ):
        raise NotImplementedError


class MultiClusterDistFieldTGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return (
            f"MultiClusterDistField-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @staticmethod
    def model_output_len(config) -> int:
        """Determine the model output length for the required eval length.

        Currently, it's double the eval length. Large model length provides
        more information to the model while training, and provides the option
        """
        min_model_output_ms = 50
        min_model_output_bins = ms_to_num_bins(
            min_model_output_ms, config.downsample
        )
        eval_len_bins = ms_to_num_bins(config.output_len, config.downsample)
        model_bins = round(max(min_model_output_bins, eval_len_bins * 2))
        return model_bins

    @classmethod
    def create_trainable(
        cls,
        splits: Iterable[RecordingTrainValTest],
        config,
        opt,
        rec_cluster_ids: mea.RecClusterIds,
    ):
        output_len = cls.model_output_len(config)
        train_ds, val_ds, test_ds = create_multi_cluster_df_datasets(
            splits,
            config.input_len,
            output_len,
            config.downsample,
            stride=opt.stride,
        )
        # There is separation between the target inference duration, say 10ms,
        # and the output length of the model, say 20ms. The model output should
        # be at least as large as the target inference duration, and it will
        # probably benefit in being larger.
        model = retinapy.models.CatMultiClusterModel(
            config.input_len + output_len,
            output_len,
            cls.num_downsample_layers(config.input_len, config.output_len),
            num_clusters=max(rec_cluster_ids.values()) + 1,
            z_dim=opt.zdim,
        )
        res = DistVAETrainable(
            train_ds,
            val_ds,
            test_ds,
            rec_cluster_ids,
            model,
            DistFieldCnnTGroup.trainable_label(config),
            eval_lengths_ms=[10, 20, 50],
            vae_beta=opt.vae_beta,
        )
        return res

    @staticmethod
    def num_downsample_layers(in_len, out_len):
        res_f = math.log(in_len / out_len, 2)
        num_downsample = int(res_f)
        if res_f - num_downsample > 0.4:
            logging.warning(
                "Model input/output lengths are not well matched. "
                f"Downsample desired: ({res_f:.4f}), downsample being used: "
                f"({num_downsample})."
            )
        return num_downsample


class TransformerTGroup(TrainableGroup):
    """
    How many bins will each supported model output?
    """

    input_to_output_len = {1984: 200, 992: 150, 3174: 400, 1586: 200}
    # stim_ds = {1984: 6, 992: 5, 3174: 7, 1586: 6}[config.input_len]
    input_to_stim_ds = {1984: 7, 992: 7, 3174: 8, 1586: 7}
    # spike_patch_len = {1984: 16, 992: 8, 3174: 32, 1586: 16}[
    input_to_spike_patch_len = {1984: 16, 992: 64, 3174: 32, 1586: 16}

    @staticmethod
    def trainable_label(config):
        return f"Transformer-{config.downsample}" f"ds_{config.input_len}in"

    @classmethod
    def create_trainable(
        cls,
        splits: Iterable[RecordingTrainValTest],
        config,
        opt,
        rec_cluster_ids: mea.RecClusterIds,
    ):
        """
        Args:
            rec_cluster_ids: map from globally unique recording name and
                cluster id pair to numeric ids used by the dataset and model.
        """
        # There is separation between the target inference duration, say 10ms,
        # and the output length of the model, say 20ms. The model output should
        # be at least as large as the target inference duration, and it will
        # probably benefit in being larger.
        model_out_len = cls.input_to_output_len[config.input_len]
        stim_ds = cls.input_to_stim_ds[config.input_len]
        spike_patch_len = cls.input_to_spike_patch_len[config.input_len]
        train_ds, val_ds, test_ds = create_multi_cluster_df_datasets(
            splits,
            config.input_len,
            model_out_len,
            config.downsample,
            stride=opt.stride,
        )
        model = retinapy.models.TransformerModel(
            config.input_len,
            model_out_len,
            stim_downsample=stim_ds,
            num_clusters=max(rec_cluster_ids.values()) + 1,
            z_dim=opt.zdim,
            num_heads=opt.num_heads,
            head_dim=opt.head_dim,
            num_tlayers=opt.num_tlayers,
            spike_patch_len=spike_patch_len,
        )
        res = DistVAETrainable(
            train_ds,
            val_ds,
            test_ds,
            rec_cluster_ids,
            model,
            TransformerTGroup.trainable_label(config),
            eval_lengths_ms=[10, 20, 50],
            vae_beta=opt.vae_beta,
        )
        return res


class ClusteringTGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return (
            f"ClusterTransformer-{config.downsample}" f"ds_{config.input_len}in"
        )

    @staticmethod
    def create_trainable(
        splits: Iterable[RecordingTrainValTest],
        config,
        opt,
        rec_cluster_ids: mea.RecClusterIds,
    ):
        # There is separation between the target inference duration, say 10ms,
        # and the output length of the model, say 20ms. The model output should
        # be at least as large as the target inference duration, and it will
        # probably benefit in being larger.
        model_out_len = {1984: 200, 992: 150, 3174: 400, 1586: 200}[
            config.input_len
        ]
        stim_ds = {1984: 6, 992: 5, 3174: 7, 1586: 6}[config.input_len]
        train_ds, val_ds, test_ds = create_multi_cluster_df_datasets(
            splits,
            config.input_len,
            model_out_len,
            config.downsample,
            stride=opt.stride,
        )
        model = retinapy.models.ClusteringTransformer(
            config.input_len,
            model_out_len,
            stim_downsample=stim_ds,
            num_clusters=max(rec_cluster_ids.values()) + 1,
            z_dim=opt.zdim,
            num_heads=opt.num_heads,
            head_dim=opt.head_dim,
            num_tlayers=opt.num_tlayers,
        )
        res = DistVAETrainable(
            train_ds,
            val_ds,
            test_ds,
            rec_cluster_ids,
            model,
            DistFieldCnnTGroup.trainable_label(config),
            eval_lengths_ms=[10, 20, 50],
            vae_beta=opt.vae_beta,
        )
        return res


class DistFieldCnnTGroup(TrainableGroup):

    DIST_PREFIX_LEN = 26

    @staticmethod
    def model_output_len(input_len):
        output_lens = {1984: 200, 992: 100, 3174: 400, 1586: 200}
        return output_lens[input_len]

    @staticmethod
    def trainable_label(config):
        return f"DistFieldCnn-{config.downsample}" f"ds_{config.input_len}in"

    @classmethod
    def create_trainable(cls, split: RecordingTrainValTest, config, opt):
        is_single_cluster = all(
            sp.num_clusters() == 1 for s in split for sp in s
        )
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )
        # There is separation between the target inference duration, say 10ms,
        # and the output length of the model, say 20ms. The model output should
        # be at least as large as the target inference duration, and it will
        # probably benefit in being larger.
        model_out_len = cls.model_output_len(config.input_len)
        ds_manager = DistDataManager(
            [split],
            config.input_len,
            model_out_len,
            cls.DIST_PREFIX_LEN,
            config.downsample,
            opt.stride,
        )
        model = retinapy.models.DistanceFieldCnnModel(
            config.input_len + model_out_len,
            model_out_len,
        )
        res = SingleDistTrainable(
            ds_manager,
            model,
            DistFieldCnnTGroup.trainable_label(config),
        )
        return res


class PoissonCnnPyramid(TrainableGroup):
    def __init__(self, num_mid_layers):
        self.num_mid_layers = num_mid_layers

    def trainable_label(self, config):
        return (
            f"{self.tgroup_label()}-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    def tgroup_label(self):
        return f"PoissonCnnPyramid{self.num_mid_layers}M"

    def create_trainable(
        self, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(
            sp.num_clusters() == 1 for s in split for sp in s
        )
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )

        in_len = config.input_len
        out_len = config.output_len

        m = retinapy.models.CnnPyramid(
            in_len,
            out_len,
            self.num_mid_layers,
            objective=retinapy.models.CnnPyramid.Objective.poisson,
        )
        train_ds, val_ds, test_ds = create_count_datasets(
            split, config.input_len, config.output_len, opt.stride
        )
        label = self.trainable_label(config)
        trainable = PoissonTrainable(train_ds, val_ds, test_ds, m, label)
        return trainable


class PoissonNet(TrainableGroup):

    DEFAULT_NUM_DOWN = 7
    DEFAULT_NUM_MID = 2

    @classmethod
    def trainable_label(cls, config):
        return (
            f"{cls.tgroup_label()}-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @classmethod
    def tgroup_label(cls):
        return f"PoissonNet"

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(
            sp.num_clusters() == 1 for s in split for sp in s
        )
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )

        in_len = config.input_len
        if not in_len == 992:
            raise ValueError("Currenly only supports 992 input.")

        num_down = (
            opt.num_down if opt.num_down is not None else cls.DEFAULT_NUM_DOWN
        )
        num_mid = (
            opt.num_mid if opt.num_mid is not None else cls.DEFAULT_NUM_MID
        )
        m = retinapy.models.PoissonNet(in_len, num_down, num_mid)

        train_ds, val_ds, test_ds = create_count_datasets(
            split, config.input_len, config.output_len, opt.stride
        )
        label = cls.trainable_label(config)
        trainable = PoissonTrainable(train_ds, val_ds, test_ds, m, label)
        return trainable


class PoissonNet2(TrainableGroup):

    DEFAULT_NUM_DOWN = 7
    DEFAULT_NUM_MID = 5

    @classmethod
    def trainable_label(cls, config):
        return (
            f"{cls.tgroup_label()}-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @classmethod
    def tgroup_label(cls):
        return f"PoissonNet2"

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(
            sp.num_clusters() == 1 for s in split for sp in s
        )
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )

        in_len = config.input_len
        if not in_len == 992:
            raise ValueError("Currenly only supports 992 input.")

        num_down = (
            opt.num_down if opt.num_down is not None else cls.DEFAULT_NUM_DOWN
        )
        num_mid = (
            opt.num_mid if opt.num_mid is not None else cls.DEFAULT_NUM_MID
        )
        m = retinapy.models.PoissonNet2(in_len, num_down, num_mid)

        train_ds, val_ds, test_ds = create_count_datasets(
            split, config.input_len, config.output_len, opt.stride
        )
        label = cls.trainable_label(config)
        trainable = PoissonTrainable(train_ds, val_ds, test_ds, m, label)
        return trainable


class DistfieldCnnPyramid(TrainableGroup):
    DIST_PREFIX_LEN = 26

    @staticmethod
    def model_output_len(input_len):
        output_lens = {1984: 200, 992: 150, 3174: 400, 1586: 200}
        return output_lens[input_len]

    def __init__(self, num_mid_layers):
        self.num_mid_layers = num_mid_layers

    def trainable_label(self, config):
        return (
            f"{self.tgroup_label()}-{config.downsample}"
            f"ds_{config.input_len}in"
        )

    def tgroup_label(self):
        return f"DistfieldCnnPyramid{self.num_mid_layers}M"

    def create_trainable(
        self, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(s.num_clusters() == 1 for s in split)
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s.num_clusters() for s in split]})."
            )

        in_len = config.input_len
        out_len = self.model_output_len(in_len)

        m = retinapy.models.CnnPyramid(
            in_len,
            out_len,
            self.num_mid_layers,
            objective=retinapy.models.CnnPyramid.Objective.distfield,
        )
        ds_manager = DistDataManager(
            [split],
            config.input_len,
            out_len,
            config.downsample,
            self.DIST_PREFIX_LEN,
            opt.stride,
        )
        label = self.trainable_label(config)
        res = SingleDistTrainable(
            ds_manager,
            m,
            label,
        )
        return res


class DistfieldCnnPyramidW5M(TrainableGroup):

    DIST_PREFIX_LEN = 26

    @staticmethod
    def trainable_label(config):
        return (
            f"DistfieldCnnPyramidW5M-{config.downsample}ds_{config.input_len}in"
        )

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        num_mid_layers = 5
        out_len = 120  # 992 / (2**3) - 4
        is_single_cluster = all(s.num_clusters() == 1 for s in split)
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s.num_clusters() for s in split]})."
            )

        in_len = config.input_len

        m = retinapy.models.CnnPyramid(
            in_len,
            out_len,
            num_mid_layers,
            objective=retinapy.models.CnnPyramid.Objective.distfieldw,
        )
        ds_manager = DistDataManager(
            [split],
            config.input_len,
            out_len,
            cls.DIST_PREFIX_LEN,
            config.downsample,
            opt.stride,
        )
        label = cls.trainable_label(config)
        res = SingleDistTrainable(
            ds_manager,
            m,
            label,
        )
        return res


class DistfieldUnet(TrainableGroup):
    # DIST_PREFIX_LEN = 20
    # |<--------- 126 ------------->|
    # |  26   |   80   |     20     |
    DIST_PREFIX_LEN = 26

    @classmethod
    def trainable_label(cls, config):
        return f"DistfieldUnet1-{config.downsample}ds_{config.input_len}in"

    @dataclasses.dataclass
    class ModelConfig:
        n_down: int
        n_mid: int
        n_up: int

        def __str__(self):
            return f"{self.n_down}d{self.n_mid}m{self.n_up}u"

    @classmethod
    def model_config(cls, config: Configuration) -> "ModelConfig":
        res_map = {
            (18, 992): cls.ModelConfig(7, 5, 4),
        }
        return res_map.get((config.downsample, config.input_len), None)

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(s[0].num_clusters() == 1 for s in split)
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )

        dist_prefix_len = round(
            ms_to_num_bins(cls.DIST_PREFIX_LEN, config.downsample)
        )
        mc = cls.model_config(config)
        if mc is None:
            raise ValueError(
                f"Unsupported input_len={config.input_len} and "
                f"downsample={config.downsample}"
            )
        # out_len = int(18 / config.downsample * 256) - 2
        out_len = 126
        ds_manager = DistDataManager(
            [split],
            config.input_len,
            out_len,
            config.downsample,
            train_stride=opt.stride,
            dist_prefix_len=dist_prefix_len,
        )
        m = retinapy.models.CnnUNet(
            config.input_len, mc.n_down, mc.n_mid, mc.n_up
        )
        label = cls.trainable_label(config)
        res = SingleDistTrainable(
            ds_manager,
            m,
            label,
        )
        return res


class DistfieldUnet2(TrainableGroup):

    # |<--------- 128 ------------->|
    # |  32   |   80   |     16     |
    DIST_PREFIX_LEN = 32

    @classmethod
    def trainable_label(cls, config):
        return f"DistfieldUnet2-{config.downsample}ds_{config.input_len}in"

    @dataclasses.dataclass
    class ModelConfig:
        n_down: int
        n_mid: int
        n_up: int

        def __str__(self):
            return f"{self.n_down}d{self.n_mid}m{self.n_up}u"

    @classmethod
    def model_config(cls, config: Configuration) -> "ModelConfig":
        res_map = {
            (18, 992): cls.ModelConfig(7, 5, 4),
            # (9, 1984): cls.ModelConfig(7, 1, 5),
        }
        return res_map.get((config.downsample, config.input_len), None)

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(
            sp.num_clusters() == 1 for s in split for sp in s
        )
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )

        dist_prefix_len = round(
            ms_to_num_bins(cls.DIST_PREFIX_LEN, config.downsample)
        )
        mc = cls.model_config(config)
        if mc is None:
            raise ValueError(
                f"Unsupported input_len={config.input_len} and "
                f"downsample={config.downsample}"
            )
        out_len = 128
        ds_manager = DistDataManager(
            [split],
            config.input_len,
            out_len,
            config.downsample,
            train_stride=opt.stride,
            dist_prefix_len=dist_prefix_len,
            use_augmentation=False,
        )
        m = retinapy.models.CnnUNet3(
            config.input_len, mc.n_down, mc.n_mid, mc.n_up, dist_prefix_len
        )
        label = cls.trainable_label(config)
        res = SingleDistTrainable(
            ds_manager,
            m,
            label,
            eval_mode="basic",
            # eval_mode="detail-2",
            # loss_type="L1",
        )
        return res


class DistAttCnnUnet(TrainableGroup):

    # |<--------- 128 ------------->|
    # |   2   |    5   |      1     | (encoder space)
    # |  32   |   80   |     16     |
    DIST_PREFIX_LEN = 32

    @classmethod
    def trainable_label(cls, config):
        return f"DistAttCnnUnet-{config.downsample}ds_{config.input_len}in"

    @dataclasses.dataclass
    class ModelConfig:
        n_down: int
        n_mid: int
        n_up: int

        def __str__(self):
            return f"{self.n_down}d{self.n_mid}m{self.n_up}u"

    @classmethod
    def model_config(cls, config: Configuration) -> "ModelConfig":
        res_map = {
            (18, 992): cls.ModelConfig(7, 5, 4),
        }
        return res_map.get((config.downsample, config.input_len), None)

    @classmethod
    def create_trainable(
        cls,
        splits: Sequence[RecordingTrainValTest],
        config: Configuration,
        opt,
        rec_cluster_ids: mea.RecClusterIds,
    ):
        dist_prefix_len = round(
            ms_to_num_bins(cls.DIST_PREFIX_LEN, config.downsample)
        )
        mc = cls.model_config(config)
        if mc is None:
            raise ValueError(
                f"Unsupported input_len={config.input_len} and "
                f"downsample={config.downsample}"
            )
        out_len = 128
        ds_manager = DistDataManager(
            splits,
            config.input_len,
            out_len,
            config.downsample,
            train_stride=opt.stride,
            dist_prefix_len=dist_prefix_len,
        )
        m = retinapy.models.AttCnnUNet(
            config.input_len,
            num_clusters=max(rec_cluster_ids.values()) + 1,
            n_mid=mc.n_mid,
        )

        label = cls.trainable_label(config)
        # Mean calculated from Chicken_17_08_21_Phase_00, 69 clusters from
        # exp_9_1_1. Takes about 5min to compute.
        output_mean = 4.724
        output_sd = 0.983
        res = MultiDistTrainable(
            ds_manager,
            m,
            label,
            output_mean=output_mean,
            output_sd=output_sd,
            clusters_to_eval=[22, 54, 70],
        )
        return res


class DistFilmUNet(TrainableGroup):

    # DIST_PREFIX_LEN = 20
    # |<-------------- 254 ------------->|
    # |  120       |   80   |     54     |
    # DIST_PREFIX_LEN = 120
    # |<--------- 128 ------------->|
    # |   2   |    5   |      1     | (encoder space)
    # |  32   |   80   |     16     |
    DIST_PREFIX_LEN = 32

    @classmethod
    def trainable_label(cls, config):
        return f"DistFilmUNet-{config.downsample}ds_{config.input_len}in"

    @dataclasses.dataclass
    class ModelConfig:
        n_down: int
        n_mid: int
        n_up: int

        def __str__(self):
            return f"{self.n_down}d{self.n_mid}m{self.n_up}u"

    @classmethod
    def model_config(cls, config: Configuration) -> "ModelConfig":
        res_map = {
            (18, 992): cls.ModelConfig(7, 5, 4),
        }
        return res_map.get((config.downsample, config.input_len), None)

    @classmethod
    def create_trainable(
        cls,
        splits: Sequence[RecordingTrainValTest],
        config: Configuration,
        opt,
        rec_cluster_ids: mea.RecClusterIds,
    ):
        dist_prefix_len = round(
            ms_to_num_bins(cls.DIST_PREFIX_LEN, config.downsample)
        )
        mc = cls.model_config(config)
        if mc is None:
            raise ValueError(
                f"Unsupported input_len={config.input_len} and "
                f"downsample={config.downsample}"
            )
        out_len = 128
        ds_manager = DistDataManager(
            splits,
            config.input_len,
            out_len,
            config.downsample,
            train_stride=opt.stride,
            dist_prefix_len=dist_prefix_len,
        )
        m = retinapy.models.FilmUNet(
            config.input_len,
            num_clusters=max(rec_cluster_ids.values()) + 1,
            n_mid=mc.n_mid,
        )

        label = cls.trainable_label(config)
        # Mean calculated from Chicken_17_08_21_Phase_00, 69 clusters from
        # exp_9_1_1. Takes about 5min to compute.
        output_mean = 4.724
        output_sd = 0.983
        res = MultiDistTrainable(
            ds_manager,
            m,
            label,
            eval_mode="basic-quick",
            output_mean=output_mean,
            output_sd=output_sd,
            clusters_to_eval=[22, 54, 70],
        )
        return res


class PixelCnnTGroup(TrainableGroup):
    @staticmethod
    def trainable_label(config):
        return f"PixelCnn-{config.downsample}ds_{config.input_len}in"

    @staticmethod
    def create_datasets(split, downsample, input_len, output_len, stride):
        train_val_test_datasets = [
            retinapy.dataset.SnippetDataset2(
                r,
                snippet_len=input_len + output_len,
                stride=stride,
                pad_for_dist_calc=round(
                    ms_to_num_bins(LOSS_CALC_PAD_MS, downsample)
                ),
                dist_clamp=round(ms_to_num_bins(DIST_CLAMP_MS, downsample)),
            )
            for r in split
        ]
        return train_val_test_datasets

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        #        model_out_len = {1984: 200, 992: 150, 3174: 400, 1586: 200}[
        #            config.input_len
        #        ]
        # Longer for testing.
        model_out_len = {1984: 600, 992: 400, 3174: 800, 1586: 400}[
            config.input_len
        ]
        is_single_cluster = all(s.num_clusters() == 1 for s in split)
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s.num_clusters() for s in split]})."
            )

        m = retinapy.models.PixelCnn()
        train_ds, val_ds, test_ds = cls.create_datasets(
            split,
            config.downsample,
            config.input_len,
            model_out_len,
            opt.stride,
        )
        label = cls.trainable_label(config)
        return PixelCnnTrainable(
            train_ds, val_ds, test_ds, m, label, config.input_len
        )


class LinearNonLinearTGroup(TrainableGroup):
    MIN_MS = 3

    @staticmethod
    def trainable_label(config):
        return (
            f"LinearNonLinear-{config.downsample}"
            f"ds_{config.input_len}in_{config.output_len}out"
        )

    @classmethod
    def create_trainable(
        cls, split: RecordingTrainValTest, config: Configuration, opt
    ):
        is_single_cluster = all(
            sp.num_clusters() == 1 for s in split for sp in s
        )
        if not is_single_cluster:
            raise ValueError(
                "Only 1 cluster supported. Got splits with num clusters: "
                f"({[s[0].num_clusters() for s in split]})."
            )
        if config.output_ms < cls.MIN_MS:
            logging.warning(
                f"LinearNonLinear model needs at least {cls.MIN_MS}ms output."
                " Skipping."
            )
            return None

        num_inputs = IN_CHANNELS * config.input_len
        m = retinapy.models.LinearNonlinear(in_n=num_inputs, out_n=1)
        train_ds, val_ds, test_ds = create_count_datasets(
            split, config.input_len, config.output_len, opt.stride
        )
        label = LinearNonLinearTGroup.trainable_label(config)
        return PoissonTrainable(train_ds, val_ds, test_ds, m, label)


def _train_single(t, out_dir, config, opt):
    logging.info(f"Output directory: ({out_dir})")
    # Create outdir if not exists.
    out_dir.mkdir(parents=True, exist_ok=False)
    with open(out_dir / TRAINABLE_CONFIG_FILENAME, "w") as f:
        yaml.dump(config, f)
    if opt.early_stopping:
        early_stopping = retinapy.train.EarlyStopping(min_epochs=3, patience=4)
    else:
        early_stopping = None
    retinapy.train.train(
        t,
        num_epochs=opt.epochs,
        batch_size=opt.batch_size,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        out_dir=out_dir,
        steps_til_log=opt.steps_til_log,
        steps_til_eval=opt.steps_til_eval,
        evals_til_eval_train_ds=opt.evals_til_eval_train_ds,
        early_stopping=early_stopping,
        initial_checkpoint=opt.initial_checkpoint,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        log_activations=opt.log_activations,
        eval_batch_size=opt.batch_size * 4,
    )
    logging.info(f"Finished training model")


def model_dir(
    base_dir: str | pathlib.Path,
    trainable,
    rec_name: Optional[str] = None,
    cluster_id: Optional[int] = None,
):
    """
    Determines where to store the training data.
    """
    xor_none = (rec_name is None) ^ (cluster_id is None)
    if xor_none:
        raise ValueError(
            "rec_name and cluster_id must both be None or not "
            f"None. Got ({rec_name}, {cluster_id})"
        )
    res = pathlib.Path(base_dir) / trainable.label
    if rec_name:
        assert cluster_id is not None
        res = res / f"{rec_name}_{cluster_id:04d}"
    return res


def load_config(model_dir: str | pathlib.Path):
    model_dir = pathlib.Path(model_dir)
    with open(model_dir / TRAINABLE_CONFIG_FILENAME, "r") as f:
        res = yaml.load(f, Loader=yaml.SafeLoader)
    return res


def _create_tgroups():
    """
    A hard-coded registry of available TrainableGroups.
    """
    single_cluster_tgroups = {
        "LinearNonLinear": LinearNonLinearTGroup,
        "DistFieldCnn": DistFieldCnnTGroup,
        "PixelCnn": PixelCnnTGroup,
        "DistfieldCnnPyramidW5M": DistfieldCnnPyramidW5M,
        "DistfieldUnet1": DistfieldUnet,
        "DistfieldUnet2": DistfieldUnet2,
        "PoissonNet": PoissonNet,
        "PoissonNet2": PoissonNet2,
    }
    poissonCnnTGroups = {
        tg.tgroup_label(): tg
        for tg in [PoissonCnnPyramid(n_mid) for n_mid in range(8)]
    }
    distfieldCnnTGroups = {
        tg.tgroup_label(): tg
        for tg in [DistfieldCnnPyramid(n_mid) for n_mid in range(8)]
    }
    single_cluster_tgroups.update(poissonCnnTGroups)
    single_cluster_tgroups.update(distfieldCnnTGroups)

    multi_cluster_tgroups = {
        "MultiClusterDistField": MultiClusterDistFieldTGroup,
        "Transformer": TransformerTGroup,
        "ClusterTransformer": ClusteringTGroup,
        "DistAttCnnUnet": DistAttCnnUnet,
        "DistFilmUNet": DistFilmUNet,
    }
    tgroups = {**single_cluster_tgroups, **multi_cluster_tgroups}
    return single_cluster_tgroups, multi_cluster_tgroups, tgroups


single_cluster_tgroups, multi_cluster_tgroups, tgroups = _create_tgroups()


def _train(out_dir, opt):
    """
    Load the MEA data and train all trainables that match the filter.

    This function is due a refactor.
    """
    print("Models & Configurations")
    print("=======================")

    def _match(run_id, match_str):
        m = re.search(match_str, run_id)
        return m is not None

    do_trainable = dict()
    for c in all_configs:
        for _, tg in tgroups.items():
            t_label = tg.trainable_label(c)
            do_trainable[t_label] = _match(t_label, opt.k)

    logging.info(f"Model-configs filter: {opt.k}")
    logging.info(
        "\n".join(
            [
                t_label if do_train else t_label.ljust(40) + " (skip)"
                for t_label, do_train in do_trainable.items()
            ]
        )
    )
    total_trainables = sum(do_trainable.values())
    logging.info(f"Total: {total_trainables} models to be trained.")

    # Load the data.
    # Filter recordings, if requested.
    if opt.recording_names is not None and len(opt.recording_names) == 1:
        # If only one recording, then cluster-ids can be specified.
        if opt.cluster_ids is not None:
            include_cluster_ids = set(opt.cluster_ids)
        else:
            include_cluster_ids = None
        recordings = [
            mea.single_3brain_recording(
                opt.recording_names[0],
                opt.data_dir,
                include_clusters=include_cluster_ids,
            )
        ]
    else:
        recordings = mea.load_3brain_recordings(opt.data_dir)
        ## Filter the recording with non-standard sample rate
        # Remove by popping; don't create a new list. Alternative is a
        # generator, but a generator makes subsequent behaviour such as
        # logging difficult.
        to_pop = []
        for idx, r in enumerate(recordings):
            is_sample_rate_supported = math.isclose(
                r.sensor_sample_rate,
                SUPPORTED_SENSOR_SAMPLE_RATE,
                rel_tol=1e-10,
            )
            if not is_sample_rate_supported:
                to_pop.append(idx)
                logging.info(
                    f"Recordings ({r.name}) has unsupported sample "
                    f"rate ({r.sensor_sample_rate} vs "
                    f"{SUPPORTED_SENSOR_SAMPLE_RATE}) and will be ignored."
                )
        for idx in sorted(to_pop, reverse=True):
            del recordings[idx]
    _, rec_cluster_ids = mea.load_id_info(opt.data_dir)

    num_clusters = sum(r.num_clusters() for r in recordings)
    logging.info(f"Num clusters: {num_clusters}")

    done_trainables = set()
    for c in all_configs:
        # single-cluster models.
        def _single_cluster_gen(all_splits):
            for train_val_test in all_splits:
                c_ids = train_val_test[0][0].cluster_ids
                for cid in c_ids:
                    yield tuple(
                        [part.clusters({cid}) for part in split]
                        for split in train_val_test
                    )

        for tg in single_cluster_tgroups.values():
            t_label = tg.trainable_label(c)
            if not do_trainable[t_label]:
                continue
            if t_label in done_trainables:
                continue
            logging.info(
                "Starting model training "
                f"({len(done_trainables)}/{total_trainables}): {t_label}"
            )
            min_spikes = tuple(n * opt.min_spikes for n in SPLIT_RATIO)
            # Wasteful to downsample each trainable, but not likely a big deal
            # given how comparatively long training is to the downsampling.
            # Might need to change in future.
            rec_splits = recording_splits(
                recordings,
                downsample=c.downsample,
                num_workers=opt.num_workers,
                min_spikes=min_spikes,
            )
            for idx, rsplit in enumerate(_single_cluster_gen(rec_splits)):
                t = tg.create_trainable(rsplit, c, opt)
                if t is None:
                    logging.warning(
                        f"Skipping. Model ({t_label}) isn't yet supported."
                    )
                    continue
                # Train, test and val splits should all have the same rec name,
                # so just take the name from the first one.
                rec_name = rsplit[0][0].name
                cluster_id = rsplit[0][0].cluster_ids[0]
                sub_dir = model_dir(out_dir, t, rec_name, cluster_id)
                logging.info(
                    f"Cluster {idx+1}/{num_clusters}: {rec_name} ({cluster_id})"
                )
                _train_single(t, sub_dir, c, opt)
            done_trainables.add(t_label)
        # Multi-cluster models.
        for tg in multi_cluster_tgroups.values():
            t_label = tg.trainable_label(c)
            if t_label in done_trainables:
                continue
            if not do_trainable[t_label]:
                continue
            # Wasteful to downsample each trainable, but not likely a big deal
            # given how comparatively long training is to the downsampling.
            # Might need to change in future.
            rec_splits = recording_splits(
                recordings,
                downsample=c.downsample,
                num_workers=opt.num_workers,
            )
            t = tg.create_trainable(rec_splits, c, opt, rec_cluster_ids)
            if t is None:
                logging.warning(
                    f"Skipping. Model ({t_label}) isn't yet supported."
                )
                continue
            logging.info(
                "Starting model training "
                f"({len(done_trainables)}/{total_trainables}): {t_label}"
            )
            sub_dir = out_dir / str(t_label)
            _train_single(t, sub_dir, c, opt)
            done_trainables.add(t_label)
    logging.info("Finished training all models.")


def main():
    retinapy._logging.setup_logging(logging.INFO)
    opt, opt_text = retinapy.cmdline.parse_args(*arg_parsers())
    labels = opt.labels.split(",") if opt.labels else None
    base_dir = pathlib.Path(opt.output if opt.output else DEFAULT_OUT_BASE_DIR)
    out_dir = retinapy._logging.get_outdir(base_dir, labels)
    print("Output directory:", out_dir)
    retinapy._logging.enable_file_logging(out_dir / LOG_FILENAME)
    # Record the arguments.
    with open(str(out_dir / ARGS_FILENAME), "w") as f:
        f.write(opt_text)
    # Snapshot the retinapy module source code.
    retinapy._logging.snapshot_module(out_dir)
    _train(out_dir, opt)


if __name__ == "__main__":
    main()
