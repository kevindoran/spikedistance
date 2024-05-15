"""Infer spike trains and calculate metrics with a spike distance model."""

import enum
import logging
import math
import multiprocessing as mp
import pathlib
from typing import List, Union
import numpy as np
import torch
import polars as pl
import typer
import retinapy
import retinapy.mea as mea
import retinapy.metrics
import retinapy.spikedistance as sdf
import retinapy.spikeprediction as sp

_logger = logging.getLogger(__name__)

# Options
CONFIG = sp.Configuration(downsample=18, input_len=992, output_len=None)
SIGMA_RANGE = range(0, 150 + 1, 1)
DEVICE = "cuda"
# Setup
torch.set_grad_enabled(False)
app = typer.Typer()


class ModelType(enum.Enum):
    Unet = "Unet"
    Unet2 = "Unet2"


def load_trainable(
    modelType: ModelType,
    trainable_path,
    rec_name,
    ds_split,
    cid: int,
    opt,
    device,
):
    if modelType == ModelType.Unet:
        tgroup = sp.DistfieldUnet()
    elif modelType == ModelType.Unet2:
        tgroup = sp.DistfieldUnet2()
    else:
        raise Exception()
    trainable = tgroup.create_trainable(ds_split, CONFIG, opt)
    model_dir = sp.model_dir(trainable_path, trainable, rec_name, cid)
    ckpt_path = model_dir / f"checkpoint_best_loss.pth"
    retinapy.models.load_model(trainable.model, ckpt_path, map_location=device)
    trainable.model.cuda()
    trainable.model.eval()
    return trainable


def next_spike_input(prev_spikes, stride, pred_spikes):
    # Window forward
    prev_spikes[0:-stride] = prev_spikes[stride:].clone()
    # Fill in new spikes
    prev_spikes[-stride:] = pred_spikes[0:stride].clone()
    return prev_spikes


def infer_spiketrain(
    trainable, stride, refactory, use_test_ds: bool, clamp_to_known: bool = True
):
    ds = trainable.test_ds if use_test_ds else trainable.val_ds
    ds.stride = stride
    lhs_spikes = []
    pred_spikes = []
    actual_spikes = []
    target_dists = []
    model_outs = []

    spike_row = 4

    snippet_len = ds.input_len
    prev_spikes = torch.zeros(snippet_len)  # .to("cuda")
    for idx, sample in enumerate(ds):
        sample_batch = torch.utils.data.dataloader.default_collate([sample])
        assert sample_batch["snippet"].shape[0] == 1
        sample_batch["snippet"][0, spike_row] = prev_spikes
        model_out, _ = trainable.forward(sample_batch)
        assert torch.all(~torch.isinf(model_out))
        input_spikes = sample_batch["snippet"][:, -1]
        assert input_spikes.shape == (1, snippet_len)
        lhs_spike = (
            sdf.lhs_spike(input_spikes, trainable.max_bin_dist).cpu().item()
        )
        assert lhs_spike < 0
        assert not (math.isinf(lhs_spike) or math.isnan(lhs_spike))
        lhs_spikes.append(lhs_spike)
        target_spikes = sample_batch["target_spikes"][0, 0:stride]
        # NOTE: working with distance fields, not model outputs.
        distf = trainable.nn_output_to_dist(model_out).cpu()[0]
        target_dist = sample_batch["dist"][0]
        # this was used for testing:
        # distf = sample_batch["dist"].cpu()[0]
        target_dists.append(target_dist)
        # The model is not required to model what is known: the past spikes.
        # It this case, we clip to the known maximum value. For debugging
        # purposes, it can be useful to not clamp.
        if clamp_to_known:
            v_at_0 = -(lhs_spike + ds.dist_prefix_len)
            known_max = torch.arange(v_at_0, v_at_0 + len(distf))
            display_dist = torch.clamp(distf, max=known_max)
            model_outs.append(display_dist)
        else:
            model_outs.append(distf)
        mle_spikes = sdf.predict(
            distf,
            lhs_spike,
            trainable.max_bin_dist,
            dist_prefix_len=ds.dist_prefix_len,
            refactory=refactory,
        )
        pred = mle_spikes[0:stride]
        pred_spikes.append(pred)
        actual_spikes.append(target_spikes)
        prev_spikes = next_spike_input(prev_spikes, stride, mle_spikes)
    pred_spikes = torch.cat(pred_spikes, dim=0).cpu().numpy()
    actual_spikes = torch.cat(actual_spikes, dim=0).cpu().numpy()
    # Stack model outputs
    target_dists = torch.stack(target_dists, dim=0).cpu().numpy()
    model_outs = torch.stack(model_outs, dim=0).cpu().numpy()

    assert np.array_equal(actual_spikes, ds.output_spikes()), (
        "The ground truth spikes collected from iteration should be the "
        "same as those obtained from a contiguous slice."
    )
    return lhs_spikes, actual_spikes, pred_spikes, target_dists, model_outs


def spiketrain_task(
    c_id: int,
    trainable_path: Union[str, pathlib.Path],
    rec_name: str,
    model_type: ModelType,
    rec_cluster_ids,
    opt,
    cluster_split,
    num_input_bins: int,
    stride: int,
    refactory: int,
    use_test_ds: bool,
):
    print(f"Starting cid: {c_id}, num_out: {stride}")
    trainable = load_trainable(
        model_type, trainable_path, rec_name, cluster_split, c_id, opt, DEVICE
    )
    ds = trainable.test_ds if use_test_ds else trainable.val_ds
    sample_period_ms = trainable.sample_period_ms
    # NN forward
    lhs_spikes, actual, pred, target_dists, m_outs = infer_spiketrain(
        trainable,
        stride,
        refactory,
        use_test_ds,
    )
    actual_spikes = np.flatnonzero(actual) + num_input_bins
    pred_spikes = np.flatnonzero(pred) + num_input_bins
    ground_truth = []
    for sub_ds in ds.datasets:
        rec = sub_ds.recording
        ground_truth.append(
            rec.spikes[
                num_input_bins : num_input_bins + len(sub_ds) * stride, 0
            ]
        )
    ground_truth = np.concatenate(ground_truth)
    assert np.array_equal(actual, ground_truth), "Mismatch in ground truth."
    assert np.array_equal(
        np.flatnonzero(ground_truth) + num_input_bins, actual_spikes
    ), "Mismatch in ground truth (indices)."
    _logger.info(f"actual vs. pred: {actual.shape} vs. {pred.shape}")
    dist_prefix_len = ds.dist_prefix_len

    g_id = rec_cluster_ids[(rec_name, c_id)]
    stats_df = pl.DataFrame(
        data=[
            [
                "Distfield",
                rec_name,
                c_id,
                g_id,
                sample_period_ms,
                num_input_bins,
                stride,
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
                "Distfield",
                rec_name,
                c_id,
                g_id,
                sample_period_ms,
                num_input_bins,
                stride,
                dist_prefix_len,
                lhs_spikes,
                actual_spikes.tolist(),
                pred_spikes.tolist(),
                target_dists.tolist(),
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
            ("dist_prefix_len", None),
            ("lhs_spikes", pl.List(pl.Int32)),
            ("actual_spikes", pl.List(pl.Int32)),
            ("pred_spikes", pl.List(pl.Int32)),
            ("target_dist", pl.List(pl.List(pl.Float32))),
            ("model_output", pl.List(pl.List(pl.Float32))),
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
    metrics_out_path: str,
    model_type: ModelType = typer.Option("Unet2"),
    data_dir: str = typer.Option(),
    rec_name: str = typer.Option(),
    clusters: List[int] = typer.Option([]),
    strides: List[int] = typer.Option([]),
    refactory: int = typer.Option(0),
    use_test_ds: bool = typer.Option(False),
    num_workers: int = typer.Option(10),
):
    trainable_path = pathlib.Path(trainable_path)
    opt = sp.opt_from_yaml(trainable_path / "args.yaml")
    _logger.info(
        "Using {} dataset".format("test" if use_test_ds else "validation")
    )
    # Typer will initialize cluster_ids to [] instead of None.
    cluster_ids = None if clusters == [] else set(clusters)
    _, rec_cluster_ids = mea.load_id_info(data_dir)
    rec_ds_split = sp.get_ds_split(
        data_dir, rec_name, CONFIG.downsample, num_workers, cluster_ids
    )
    cluster_ids = sorted(rec_ds_split[2][0].cluster_ids)
    _logger.info(f"Including clusters: {cluster_ids}")
    if len(strides) == 0:
        strides = [
            80,
        ]
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        res_async = []
        for c_id in cluster_ids:
            for stride in strides:
                res_async.append(
                    pool.apply_async(
                        spiketrain_task,
                        args=(
                            c_id,
                            trainable_path,
                            rec_name,
                            model_type,
                            rec_cluster_ids,
                            opt,
                            sp.single_cluster_split(rec_ds_split, c_id),
                            CONFIG.input_len,
                            stride,
                            refactory,
                            use_test_ds,
                        ),
                    )
                )
        spikes_dfs, stats_dfs = zip(*[r.get() for r in res_async])
    stats_df = pl.concat(stats_dfs, how="vertical")
    spikes_df = pl.concat(spikes_dfs, how="vertical")
    stats_df.write_parquet(metrics_out_path)
    spikes_df.write_parquet(spikes_out_path)
    # Log.
    num_print_rows = 100
    pl.Config.set_tbl_rows(num_print_rows)
    _logger.info(
        stats_df.select(
            pl.exclude(
                "model",
                "recording",
                "gid",
                "sample_period_ms",
                "smooth_pcorr",
                "num_input_bins",
            )
        ).head(n=num_print_rows)
    )


if __name__ == "__main__":
    # Set logging to level info
    logging.basicConfig(level=logging.INFO)
    app()
