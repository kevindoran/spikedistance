import logging
import retinapy
import retinapy.mea as mea
import numpy as np
import retinapy.spikeprediction as sp
import polars as pl
import pandas as pd

_logger = logging.getLogger(__name__)


def high_freq_filter(
    rec: mea.CompressedSpikeRecording,
) -> mea.CompressedSpikeRecording:
    """Remove any spurious high frequency virtual cells.

    A small number of cells are artifacts of the 20 Hz trigger signal."""
    MAX_FREQ = 19
    res = rec.filter_clusters(max_rate=19, min_count=0)
    _logger.info(
        "Filtering out high frequency cells: "
        f"{len(rec.cluster_ids)} -> {len(res.cluster_ids)} "
        f"({len(rec.cluster_ids) - len(res.cluster_ids)})."
    )
    return res


def spike_count_filter(
    rec: mea.CompressedSpikeRecording,
    min_train_spike_rate=0.0,
) -> mea.CompressedSpikeRecording:
    """Filter-in cells that have sufficient spikes in the test set.

    No requirements are placed on the train or validation set."""
    # Decompress then split the recording so we can count the spikes in the
    # segment that forms the test set.
    rec_dc = mea.decompress_recording(rec, downsample=18)
    train_val_test_splits = mea.mirror_split2(
        rec_dc, split_ratio=sp.SPLIT_RATIO
    )
    train_segments = train_val_test_splits[0]
    min_train_spikes = round(
        min_train_spike_rate * sum(s.duration() for s in train_segments)
    )
    train_val_test_splits = mea.remove_few_spike_clusters(
        train_val_test_splits, (min_train_spikes, 0, 0)
    )
    cids = train_val_test_splits[0][0].cluster_ids
    res = rec.clusters(set(cids) & set(rec.cluster_ids))
    _logger.info(
        "Filtering out cells with train set spike counts below threshold "
        f"({min_train_spikes} spikes): "
        f"{len(rec.cluster_ids)} -> {len(res.cluster_ids)} "
        f"({len(rec.cluster_ids) - len(res.cluster_ids)})."
    )
    return res


def main():
    chicken_recording = mea.single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        "./data/chicken_ff_noise",
    )
    frog_recording = mea.single_3brain_recording(
        "Xla_2022-04-29_Ph00_19", 
        "./data/frog_ff_noise"
    )

    min_spike_rate_per_sec = 0.75
    chicken_rec = spike_count_filter(
        high_freq_filter(chicken_recording),
        min_train_spike_rate=min_spike_rate_per_sec,
    )
    frog_rec = spike_count_filter(
        high_freq_filter(frog_recording),
        min_train_spike_rate=min_spike_rate_per_sec,
    )
    print(f"Chicken_2021_08_17 cell IDs: {sorted(chicken_rec.cluster_ids)}")
    print(f"Xla_2022-04-29_Ph00_19 cell IDs: {sorted(frog_rec.cluster_ids)}")
    assert (
        len(chicken_rec.cluster_ids) == 60 
    ), f"Expected 135, got {len(chicken_rec.cluster_ids)}"
    assert (
        len(frog_rec.cluster_ids) == 113
    ), f"Expected 110, got {len(frog_rec.cluster_ids)}"


if __name__ == "__main__":
    # Set logging to level info
    logging.basicConfig(level=logging.INFO)
    main()
