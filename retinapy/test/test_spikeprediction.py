import pytest
import retinapy.spikeprediction as sp
import retinapy.mea as mea
import itertools
import pathlib

DATA_DIR = pathlib.Path("./data/chicken_ff_noise")


@pytest.fixture
def recs():
    res = mea.load_3brain_recordings(
        DATA_DIR,
        include=["Chicken_04_08_21_Phase_01", "Chicken_04_08_21_Phase_02"],
    )
    # Filter out some clusters
    res[0] = res[0].clusters({40, 41})
    res[1] = res[1].clusters({46, 352, 703})
    return res


def test_recording_splits(rec0, rec12):
    """
    Tests that:
        1. Two recordings are split without throwing an exception.
        2. Each split has 3 parts (for train, val, test).
        3. The number of clusters filtered out is correct for a single
            recording (Chicken_17_08_21_Phase_00).
    """
    # Setup
    num_clusters = rec12.num_clusters()
    assert (
        num_clusters == 154
    ), "This should always be 154 for Chicken_17_08_21_Phase_00"
    min_spikes_opt = (10, 3, 3)
    expected_post_filter_clusters = 126

    # Test
    # 1.
    rsplits = sp.recording_splits(
        [rec0, rec12], downsample=18, num_workers=10, min_spikes=min_spikes_opt
    )
    assert len(rsplits) == 2, "Two recordings => two splits."
    # 2.
    assert len(rsplits[0]) == len(rsplits[1]) == 3, "Train, val, test."
    rec12_splits = rsplits[1]
    # 3.
    assert (
        rec12_splits[0][0].num_clusters()
        == rec12_splits[0][1].num_clusters()
        == rec12_splits[1][0].num_clusters()
        == rec12_splits[1][1].num_clusters()
        == rec12_splits[2][0].num_clusters()
    ), "All splits should have the same number of clusters."
    assert rec12_splits[0][0].num_clusters() == expected_post_filter_clusters, (
        "The MAX_SPIKE_RATE, MIN_SPIKES and SPLIT_RATIO defined in "
        "spikeprediction.py are considered constants. They determine the number"
        " of clusters that remain after filtering. The number of clusters "
        "should only change if these values are changed. For "
        "Chicken_17_08_21_Phase_00, there are 122 clusters after filtering."
    )


def test_create_multi_cluster_df_datasets(rec0, rec1, rec2):
    # Setup
    # Note down the expected number of clusters. These were calculated once
    # in a Jupyter notebook. It's not a very good ground truth, as it was using
    # the function in question; however, it does work as a check against any
    # unexpected changes.
    min_spikes_opt = (10, 3, 3)
    expected_num_filtered = 88
    expected_num_clusters = (
        rec0.num_clusters()
        + rec1.num_clusters()
        + rec2.num_clusters()
        - expected_num_filtered
    )
    downsample = 18
    rec_splits = sp.recording_splits(
            [rec0, rec1, rec2],
            downsample=downsample,
            num_workers=10,
            min_spikes=min_spikes_opt,
    )

    # Test

    train_ds, val_ds, test_ds = sp.create_multi_cluster_df_datasets(
        rec_splits,
        input_len=992,
        output_len=100,
        downsample=downsample,
        stride=17,
    )
    assert (
        expected_num_clusters
        == train_ds.num_clusters
        == val_ds.num_clusters
        == test_ds.num_clusters
    )


@pytest.mark.skip(reason="Switching to dataset manager hasn't be reflected in "
    "many of the older Trainables.")
def test_trainable_factories(recs, rec_cluster_ids):
    """
    Tests multiple functions in one go (so as to speed up tests).

    Tests that:
        1. for each tranable group, a trainable is created without error for
        a small set of different configurations.
    """
    # Setup
    downsample_factors = [89, 178]
    input_lengths_ms = [992, 1586]
    output_lenghts_ms = [1, 50]
    configs = tuple(
        sp.Configuration(*tple)
        for tple in itertools.product(
            downsample_factors,
            input_lengths_ms,
            output_lenghts_ms,
        )
    )
    # Get default options from arg parser.
    parser, _ = sp.arg_parsers()
    default_opts = parser.parse_args([])
    # For the models that only support a single cluster:
    single_cluster = [recs[0].clusters({40})]

    # Test
    for config in configs:
        rsplits = list(
            sp.recording_splits(
                recs, downsample=config.downsample, num_workers=10
            )
        )
        single_cluster_splits = list(
            sp.recording_splits(
                single_cluster, downsample=config.downsample, num_workers=10
            )
        )[0]
        assert (
            sp.LinearNonLinearTGroup.create_trainable(
                single_cluster_splits, config, default_opts
            )
        ) is not None
        assert (
            sp.DistFieldCnnTGroup.create_trainable(
                single_cluster_splits, config, default_opts
            )
        ) is not None
        assert (
            sp.PoissonCnnPyramid(num_mid_layers=5).create_trainable(
                single_cluster_splits, config, default_opts
            )
        ) is not None
        assert (
            sp.MultiClusterDistFieldTGroup.create_trainable(
                rsplits, config, default_opts, rec_cluster_ids
            )
        ) is not None
        assert (
            sp.TransformerTGroup.create_trainable(
                rsplits, config, default_opts, rec_cluster_ids
            )
        ) is not None
        assert (
            sp.ClusteringTGroup.create_trainable(
                rsplits, config, default_opts, rec_cluster_ids
            )
        ) is not None
