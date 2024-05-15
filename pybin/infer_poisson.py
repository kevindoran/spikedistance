"""Infer spike trains and calculate metrics with a Poisson model."""

import logging
import multiprocessing as mp
import pathlib
from typing import List, Union
import numpy as np
import torch
import polars as pl
import retinapy
import retinapy.mea as mea
import retinapy.metrics
import retinapy.models
import retinapy.spikeprediction as sp
import typer
import enum

_logger = logging.getLogger(__name__)

# 1. Options
CONFIG = sp.Configuration(downsample=18, input_len=992, output_len=None)
SIGMA_RANGE = range(0, 150 + 1, 1)
DEVICE = "cuda"

# 2. Setup
torch.set_grad_enabled(False)
app = typer.Typer()


def next_spike_input( prev_spikes, stride: int, num_pred_spikes: int):
    res = torch.zeros_like(prev_spikes)
    # Window forward
    res[0:-stride] = prev_spikes[stride:]
    # Fill in new spikes
    # Evenly place `num_pred_spikes` in the last `stride` bins.
    spikes = torch.round(
        torch.linspace(
            0, stride, num_pred_spikes + 2, dtype=torch.long, device=res.device
        )[1:-1]
    )
    spikes = spikes + (len(prev_spikes) - stride)
    assert len(spikes) == num_pred_spikes
    res.index_add_(
        dim=0,
        index=spikes,
        source=torch.ones_like(spikes, dtype=prev_spikes.dtype),
    )
    return res


def infer_spiketrain(trainable, use_test_ds: bool, method = "round"):
    ds = trainable.test_ds if use_test_ds else trainable.val_ds
    actual_spikes = []
    pred_spikes = []
    model_outs = []

    def floor(m_out):
        """Floor (the mode)."""
        return int(m_out)

    def sample(m_out):
        res = np.random.default_rng().poisson(lam=m_out)
        assert res == int(res)
        return int(res)

    if method == "round":
        infer_fn = round  # builtin
    elif method == "floor":
        infer_fn = floor
    elif method == "sample":
        infer_fn = sample
    else:
        raise ValueError(
            f'Inference method must be one of "round", "floor" or "sample"'
        )

    spike_row = 4
    stride = ds.stride
    snippet_len = ds.input_len
    prev_spikes = torch.zeros(snippet_len).to("cuda")
    for idx, sample in enumerate(ds):
        # sample = X,y, with first dim being batch
        X, y = torch.utils.data.dataloader.default_collate([sample])
        X[0][spike_row] = prev_spikes
        model_out = trainable.quick_forward((X, y)).item()
        model_outs.append(model_out)
        pred_count = infer_fn(model_out)
        # Update spike prediction
        prev_spikes = next_spike_input(prev_spikes, stride, pred_count)
        pred_spikes.append(prev_spikes[-stride:])
        # Actual spikes are obtained in an atypical way: from the base dataset.
        # Spikes start after the input, up until the stride.
        spikes = ds.ds[idx]["spikes"][ds.input_len : ds.input_len + stride]
        assert (
            pred_spikes[-1].shape == spikes.shape
        ), f"({prev_spikes.shape}, {spikes.shape})"
        actual_spikes.append(spikes)
    actual_spikes = np.concatenate(actual_spikes)
    pred_spikes = torch.cat(pred_spikes, dim=0).cpu().numpy()
    model_outs = np.array(model_outs)
    assert (
        actual_spikes.shape == pred_spikes.shape
    ), f"({actual_spikes.shape}, {pred_spikes.shape})"
    return actual_spikes, pred_spikes, model_outs


def spiketrain_task(
    c_id: int,
    trainable_path: Union[str, pathlib.Path],
    rec_name: str,
    rec_cluster_ids,
    split,
    num_input_bins: int,
    config,
    opt,
    use_test_ds: bool,
    infer_method: str,
):
    print(f"Starting cid: {c_id}, output len: {config.output_len}")
    # Setup. So much setup.
    opt.stride = config.output_len
    trainable = sp.PoissonNet2.create_trainable(split, config, opt)
    model_dir = sp.model_dir(trainable_path, trainable, rec_name, c_id)
    ckpt_path = model_dir / f"checkpoint_best_loss.pth"
    retinapy.models.load_model(trainable.model, ckpt_path, map_location=DEVICE)
    trainable.model.cuda()
    trainable.model.eval()

    # NN forward
    actual, pred, m_outs = infer_spiketrain(
        trainable, use_test_ds, infer_method
    )
    _logger.info(f"num actual vs. pred spikes: {actual.sum()} vs. {pred.sum()}")
    actual_spikes = np.flatnonzero(actual) + num_input_bins
    pred_spikes = np.flatnonzero(pred) + num_input_bins
    ds = trainable.test_ds if use_test_ds else trainable.val_ds
    ground_truth = []
    for sub_ds in ds.datasets:
        rec = sub_ds.recording
        ground_truth.append(
            rec.spikes[
                num_input_bins : num_input_bins + len(sub_ds) * opt.stride, 0
            ]
        )
    ground_truth = np.concatenate(ground_truth)
    assert np.array_equal(actual, ground_truth), "Mismatch in ground truth."
    assert np.array_equal(
        np.flatnonzero(ground_truth) + num_input_bins, actual_spikes
    ), "Mismatch in ground truth (indices)."

    g_id = rec_cluster_ids[(rec_name, c_id)]
    sample_period_ms = trainable.sample_period_ms
    stats_df = pl.DataFrame(
        data=[
            [
                "Poisson",
                rec_name,
                c_id,
                g_id,
                sample_period_ms,
                num_input_bins,
                config.output_len,
                σ_ms,
                len(pred_spikes),
                len(actual_spikes),
                pred_spikes.tolist(),
            ]
            for σ_ms in SIGMA_RANGE
        ],
        orient="row",
        schema=[
            ("model", None),
            ("recording", None),
            ("cid", None),
            ("gid", None),
            ("sample_period_ms", None),
            ("num_input_bins", None),
            ("stride", None),
            ("sigma_ms", None),
            ("num_pred_spikes", None),
            ("num_actual_spikes", None),
            ("pred_spikes", pl.List(pl.Int32)),
        ],
    )

    spikes_df = pl.DataFrame(
        data=[
            [
                "Poisson",
                rec_name,
                c_id,
                g_id,
                sample_period_ms,
                num_input_bins,
                config.output_len,
                actual_spikes.tolist(),
                pred_spikes.tolist(),
                m_outs.tolist(),
            ]
        ],
        orient="row",
        schema=[
            ("model", None),
            ("recording", None),
            ("cid", None),
            ("gid", None),
            ("sample_period_ms", None),
            ("num_input_bins", None),
            ("stride", None),
            ("actual_spikes", pl.List(pl.Int32)),
            ("pred_spikes", pl.List(pl.Int32)),
            ("model_output", pl.List(pl.Float32)),
        ],
    )

    def schreiber(sigma_ms):
        return retinapy.metrics.schreiber(
            actual, pred, bin_ms=sample_period_ms, sigma_ms=sigma_ms
        )

    def smooth_pcorr(sigma_ms):
        return retinapy.metrics.smooth_pcorr(
            actual, pred, bin_ms=sample_period_ms, sigma_ms=sigma_ms
        )

    def van_rossum(sigma_ms):
        TAU_MULTIPLIER = 1
        tau_ms = sigma_ms * TAU_MULTIPLIER
        return retinapy.metrics.van_rossum(
            actual, pred, bin_ms=sample_period_ms, tau_ms=tau_ms
        )

    stats_df = stats_df.with_columns(
        [
            pl.col("sigma_ms").apply(schreiber).alias(f"schreiber"),
            pl.col("sigma_ms").apply(smooth_pcorr).alias(f"smooth_pcorr"),
            pl.col("sigma_ms").apply(van_rossum).alias(f"van_rossum"),
        ]
    ).drop("pred_spikes")

    return spikes_df, stats_df


@app.command()
@torch.no_grad()
def infer(
    trainable_path: str,
    spikes_out_path: str,
    stats_out_path: str,
    data_dir: str = typer.Option(),
    rec_name: str = typer.Option(),
    clusters: List[int] = typer.Option([]),
    out_len: List[int] = typer.Option([]),
    use_test_ds: bool = typer.Option(False),
    infer_method: str = typer.Option("round"),
    num_workers: int = typer.Option(10),
):
    trainable_path = pathlib.Path(trainable_path)
    opt = sp.opt_from_yaml(trainable_path / "args.yaml")
    # Typer will initialize cluster_ids to [] instead of None.
    cluster_ids = None if clusters == [] else set(clusters)
    _, rec_cluster_ids = mea.load_id_info(data_dir)
    rec_ds_split = sp.get_ds_split(
        data_dir, rec_name, CONFIG.downsample, num_workers, cluster_ids
    )
    cluster_ids = sorted(rec_ds_split[2][0].cluster_ids)
    _logger.info(f"Including clusters: {cluster_ids}")
    if len(out_len):
        out_len = set(out_len)

    # E.g. 1ms 10ms 50ms 100ms
    matching_configs = []
    for c in sp.all_configs:
        match_ds = c.downsample == CONFIG.downsample
        match_input_len = c.input_len == CONFIG.input_len
        match_out_len = len(out_len) == 0 or c.output_len in out_len
        sufficient_out = c.output_ms > 3
        if match_ds and match_input_len and match_out_len and sufficient_out:
            matching_configs.append(c)
    assert len(matching_configs)

    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        res_async = []
        for c_id in cluster_ids:
            for config in matching_configs:
                res_async.append(
                    pool.apply_async(
                        spiketrain_task,
                        args=(
                            c_id,
                            trainable_path,
                            rec_name,
                            rec_cluster_ids,
                            sp.single_cluster_split(rec_ds_split, c_id),
                            CONFIG.input_len,
                            config,
                            opt,
                            use_test_ds,
                            infer_method,
                        ),
                    )
                )
        spikes_dfs, stats_dfs = zip(*[r.get() for r in res_async])
    stats_df = pl.concat(stats_dfs, how="vertical")
    spikes_df = pl.concat(spikes_dfs, how="vertical")
    stats_df.write_parquet(stats_out_path)
    spikes_df.write_parquet(spikes_out_path)
    # Log.
    num_print_rows = 100
    pl.Config.set_tbl_rows(num_print_rows)
    _logger.info(stats_df.head(n=num_print_rows))


if __name__ == "__main__":
    # Set logging to level info
    logging.basicConfig(level=logging.INFO)
    app()
