import pytest
import retinapy.dataset as dataset
import retinapy.mea as mea
import numpy as np
import pathlib


DATA_DIR = pathlib.Path("./data/chicken_ff_noise")


def test_DistDataset(rec12_1kHz):
    # Setup
    snippet_len = 2000
    mask_begin = 1000
    mask_end = 1500
    max_dist = 500
    pad = 500
    mask_shape = (mask_end - mask_begin,)

    # Test
    # 1. The dataset should be created correctly.
    exp12_1kHz_cluster_13 = rec12_1kHz.clusters({13})
    ds = dataset.DistDataset(
        exp12_1kHz_cluster_13, snippet_len, mask_begin, mask_end, pad, max_dist
    )
    # 2. The dataset should have the correct length.
    assert len(ds) == len(rec12_1kHz) - snippet_len - pad + 1

    # 3. The dataset should return the correct snippet and distance arrays.
    sample = ds[0]
    masked_snippet = sample["snippet"]
    target_spikes = sample["target_spikes"]
    dist = sample["dist"]
    # 3.1. The shapes should be correct.
    # The first dimension: LEDs (4) and spikes (1).
    assert masked_snippet.shape == (mea.NUM_STIMULUS_LEDS + 1, snippet_len)
    assert target_spikes.shape == mask_shape
    assert dist.shape == mask_shape
    # 3.2 The target spikes should be float array, with mostly zeros.
    # Note: this was changed from int. float resulted in higher examples/sec
    # when training.
    assert target_spikes.dtype == float
    known_spike_count = 10
    assert np.sum(target_spikes) == known_spike_count
    # 3.3 The distance arrays should be 0.25 where there are spikes, and >=1 
    # where there are no spikes.
    assert np.all(dist[np.where(target_spikes == 1)] == 0.25)
    assert np.all(dist[np.where(target_spikes == 0)] >= 1.0)
    # 3.4 No distance in the distance arrays should be larger than max_dist.
    assert np.max(dist) <= max_dist


@pytest.fixture
def two_exps():
    # The data directory contains ID information, so global IDs will be loaded
    # and used.
    recs = mea.load_3brain_recordings(
        DATA_DIR,
        include=["Chicken_04_08_21_Phase_01", "Chicken_20_08_21_Phase_00"],
    )
    dc_recs = mea.decompress_recordings(recs, downsample=18)
    return dc_recs


@pytest.mark.parametrize("stride", [1, 3, 17])
def test_ConcatDistDataset(two_exps, np_rng, stride):
    """Test the concatenated dist dataset.

    Tests that:
        1. Construction from two distance array datasets throws no errors and 
            has expected length.
        2. Samples with same timestep but different cluster have equal stimulus.
        3. All samples for a single cluster have the correct (and same)
            "cluster_id". For this test, this will be the global cluster ID,
            as ID information is available in the data directory.

    These tests are run parameterized for different strides.
    """
    # Setup
    num_timestep_trials = 1000
    num_cluster_trials = 50
    snippet_len = 1000
    mask_begin = 800
    mask_end = 1000
    pad = 100
    dist_clamp = 500
    dist_ds1 = dataset.DistDataset(
        two_exps[0],
        snippet_len,
        mask_begin,
        mask_end,
        pad,
        dist_clamp,
        stride=stride,
        enable_augmentation=False,
    )
    dist_ds2 = dataset.DistDataset(
        two_exps[1],
        snippet_len,
        mask_begin,
        mask_end,
        pad,
        dist_clamp,
        stride=stride,
        enable_augmentation=False,
    )

    # Test
    # 1. The dataset should be created correctly.
    concat_ds = dataset.ConcatDistDataset([dist_ds1, dist_ds2])
    assert len(concat_ds) == len(dist_ds1) + len(dist_ds2)

    # 2. Two clusters from the same recording should share the same stimulus.
    # 2.1 For the first recording.
    num_timesteps1 = dist_ds1.ds._num_strided_timesteps
    test_idxs = np_rng.integers(0, num_timesteps1, num_timestep_trials)
    for idx in test_idxs:
        s1 = concat_ds[idx]["snippet"][0 : mea.NUM_STIMULUS_LEDS, :]
        s2 = concat_ds[idx + num_timesteps1]["snippet"][
            0 : mea.NUM_STIMULUS_LEDS, :
        ]
        np.testing.assert_allclose(s1, s2, err_msg=f"idx={idx}")
    # 2.2 For the second recording.
    num_timesteps2 = dist_ds2.ds._num_strided_timesteps
    ds_2_start_idx = len(dist_ds1)
    test_idxs = np_rng.integers(
        ds_2_start_idx, ds_2_start_idx + num_timesteps2, num_timestep_trials
    )
    for idx in test_idxs:
        s1 = concat_ds[idx]["snippet"][0 : mea.NUM_STIMULUS_LEDS, :]
        s2 = concat_ds[idx + num_timesteps2]["snippet"][
            0 : mea.NUM_STIMULUS_LEDS, :
        ]
        np.testing.assert_allclose(s1, s2, err_msg=f"idx={idx}")

    # 3. A cluster's snippets should have the same 'cluster_id'.
    # 3.1 For the first recording.
    cluster_idxs = np_rng.integers(
        0, len(dist_ds1.recording.cluster_gids), num_cluster_trials
    )
    for c_idx in cluster_idxs:
        offset = c_idx * dist_ds1.ds._num_strided_timesteps
        test_idxs = np_rng.integers(0, num_timesteps1, num_timestep_trials)
        gid = dist_ds1.recording.cluster_gids[c_idx]
        for idx in test_idxs:
            c_idx_from_ds = concat_ds[idx + offset]["cluster_id"]
            assert c_idx_from_ds == gid
    # 3.2 For the second recording.
    cluster_idxs = np_rng.integers(
        0, len(dist_ds2.recording.cluster_gids), num_cluster_trials
    )
    for c_idx in cluster_idxs:
        offset = (
            len(dist_ds1) + c_idx * dist_ds2.ds._num_strided_timesteps
        )
        test_idxs = np_rng.integers(0, num_timesteps2, num_timestep_trials)
        gid = dist_ds2.recording.cluster_gids[c_idx]
        for idx in test_idxs:
            c_idx_from_ds = concat_ds[idx + offset]["cluster_id"]
            assert c_idx_from_ds == gid


@pytest.mark.parametrize("stride", [1, 3, 17, 40, 80])
@pytest.mark.parametrize(
    "input_len",
    [
        992,
        1984,
    ],
)
def test_BasicDistDataset_output_spikes(rec12_1kHz_c13, stride, input_len):
    """
    Tests that the dataset correctly iterates over the whole recording to
    produce a continuous output spike sequence.

    This tests both output_spikes() and the behaviour of the dataset.
    """
    _len = 9*1001 # ~9s
    rec_part = rec12_1kHz_c13[:_len]
    out_prefix_len = 26
    out_len = 128
    assert out_len >= out_prefix_len + stride
    ds = dataset.BasicDistDataset(
        [rec_part],
        input_len=input_len,
        output_len=out_len,
        pad=600,
        dist_prefix_len=out_prefix_len,
        dist_clamp=600,
        stride=stride,
    )

    output_spikes_slice = ds.output_spikes()

    output_spikes_iter = []
    for i in range(len(ds)):
        output_spikes_iter.append(ds[i]["target_spikes"][0:stride])
    output_spikes_iter = np.concatenate(output_spikes_iter)

    np.testing.assert_array_equal(
        output_spikes_slice, output_spikes_iter
    )
