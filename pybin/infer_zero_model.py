"""Calculate metrics for a zero model."""

import retinapy
import retinapy.spikeprediction as sp
import retinapy.mea as mea
import retinapy.metrics
from typing import Optional, Set, List

import multiprocessing as mp
import numpy as np
import torch
import pathlib
import typer
import polars as pl
import logging

# 1. Options
PROJECT_ROOT = pathlib.Path("./")
CONFIG = sp.Configuration(downsample=18, input_len=992, output_len=None)
DEVICE = "cuda"

# 2. Setup
torch.set_grad_enabled(False)
app = typer.Typer()


def metrics_task(rec_name, test_rec, c_id: int, rec_cluster_ids):
    """
    Calculates model metrics for a given cell.
    """
    num_input_bins = 992
    print(f"Starting cid: {c_id}")
    g_id = rec_cluster_ids[(rec_name, c_id)]
    actual = test_rec.spikes[num_input_bins:, test_rec.cid_to_idx(c_id)]
    pred = np.zeros_like(actual)
    sigma_range = range(0, 150 + 1, 1)
    sample_period_ms = 1000 / 992
    stats_df = pl.DataFrame(
        data=[
            [
                "zero",
                rec_name,
                c_id,
                g_id,
                sample_period_ms,
                0,
                0,
                σ_ms,
                0,
                len(np.flatnonzero(actual)),
            ]
            for σ_ms in sigma_range
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
    )
    return stats_df


@app.command()
@torch.no_grad()
def spiketrain_metrics(
    stats_out_path: str,
    data_dir: str = typer.Option(),
    rec_name: str = typer.Option(),
    cluster_ids: List[int] = typer.Option([]),
    num_workers: int = typer.Option(10),
):
    # Typer will initialize cluster_ids to [] instead of None.
    cluster_ids = None if cluster_ids == [] else set(cluster_ids)
    _, rec_cluster_ids = mea.load_id_info(data_dir)
    rec_ds_split = sp.get_ds_split(
        data_dir, rec_name, CONFIG.downsample, num_workers, cluster_ids
    )
    test_rec = rec_ds_split[2][0]
    cluster_ids = sorted(test_rec.cluster_ids)
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        res = []
        for c_id in cluster_ids:
            res.append(
                pool.apply_async(
                    metrics_task,
                    args=(rec_name, test_rec, c_id, rec_cluster_ids),
                )
            )
        stats_dfs = [r.get() for r in res]
    stats_df = pl.concat(stats_dfs, how="vertical")
    stats_df.write_parquet(stats_out_path)
    # Log.
    num_print_rows = 100
    pl.Config.set_tbl_rows(num_print_rows)
    print(stats_df.head(n=num_print_rows))


if __name__ == "__main__":
    # info logging
    logging.basicConfig(level=logging.INFO)
    app()
