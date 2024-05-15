import math

import numpy as np
import numpy.ma as ma
import numpy.testing
import pandas as pd

import pytest
import retinapy.mea as mea
import pathlib


DATA_DIR = pathlib.Path("./data/chicken_ff_noise")
FF_NOISE_PATTERN_PATH = DATA_DIR / "stimulus_pattern.npy"
FF_SPIKE_RESPONSE_PATH = DATA_DIR / "spike_response.pickle"
FF_SPIKE_RESPONSE_PATH_ZIP = DATA_DIR / "spike_response.pickle.zip"
FF_RECORDED_NOISE_PATH = DATA_DIR / "recorded_stimulus.pickle"
FF_RECORDED_NOISE_PATH_ZIP = DATA_DIR / "recorded_stimulus.pickle.zip"


def test_load_stimulus_pattern():
    noise = mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH)
    known_shape = (24000, 4)
    assert noise.shape == known_shape


def test_load_response():
    for path in (FF_SPIKE_RESPONSE_PATH, FF_SPIKE_RESPONSE_PATH_ZIP):
        response = mea.load_response(path)
        known_index_names = ["Cell index", "Stimulus ID", "Recording"]
        assert response.index.names == known_index_names
        known_shape = (4417, 2)
        assert response.shape == known_shape


def test_load_recorded_stimulus():
    for path in (FF_RECORDED_NOISE_PATH, FF_RECORDED_NOISE_PATH_ZIP):
        res = mea.load_recorded_stimulus(path)
        known_index_names = ["Stimulus_index", "Recording"]
        assert res.index.names == known_index_names
        known_shape = (18, 8)
        assert res.shape == known_shape


@pytest.fixture
def stimulus_pattern():
    return mea.load_stimulus_pattern(FF_NOISE_PATTERN_PATH)


@pytest.fixture
def recorded_stimulus():
    return mea.load_recorded_stimulus(FF_RECORDED_NOISE_PATH)


@pytest.fixture
def response_data():
    return mea.load_response(FF_SPIKE_RESPONSE_PATH)


def test_recording_names(response_data):
    known_list = [
        "Chicken_04_08_21_Phase_01",
        "Chicken_04_08_21_Phase_02",
        "Chicken_05_08_21_Phase_00",
        "Chicken_05_08_21_Phase_01",
        "Chicken_06_08_21_2nd_Phase_00",
        "Chicken_06_08_21_Phase_00",
        "Chicken_11_08_21_Phase_00",
        "Chicken_12_08_21_Phase_00",
        "Chicken_12_08_21_Phase_02",
        "Chicken_13_08_21_Phase_00",
        "Chicken_13_08_21_Phase_01",
        "Chicken_14_08_21_Phase_00",
        "Chicken_17_08_21_Phase_00",
        "Chicken_19_08_21_Phase_00",
        "Chicken_19_08_21_Phase_01",
        "Chicken_20_08_21_Phase_00",
        "Chicken_21_08_21_Phase_00",
    ]
    rec_list = mea.recording_names(response_data)
    assert rec_list == known_list


def test_cluster_ids(response_data):
    # fmt: off
    known_list = [12, 13, 14, 15, 17, 25, 28, 29, 34, 44, 45, 50, 60, 61, 80,
                  82, 99, 114, 119, 149, 217, 224, 287, 317, 421, 553, 591]
    # fmt: on
    recording_name = "Chicken_21_08_21_Phase_00"
    cluster_ids = mea._cluster_ids(response_data, recording_name)
    assert cluster_ids == known_list


def test_load_3brain_recordings():
    # Test
    res = mea.load_3brain_recordings(
        DATA_DIR,
        include=[
            "Chicken_04_08_21_Phase_01",
            "Chicken_04_08_21_Phase_02",
        ],
    )
    assert len(res) == 2


def test_filter_clusters(rec0):
    """
    Tests that:
        1. Filter by min count works for one example.
        2. Filter by min rate works for one example.
        3. Filter by max rate works for one example.
    """
    # Setup
    num_clusters = len(rec0.cluster_ids)

    # Test 1
    expected_num_dropped = 32
    num_filtered_clusters = len(rec0.filter_clusters(min_count=100).cluster_ids)
    assert (num_clusters - num_filtered_clusters) == expected_num_dropped

    # Test 2
    expected_num_dropped = 3
    num_filtered_clusters = len(
        rec0.filter_clusters(min_rate=1 / 50).cluster_ids
    )
    assert (num_clusters - num_filtered_clusters) == expected_num_dropped

    # Test 3
    expected_num_dropped = 8
    num_filtered_clusters = len(rec0.filter_clusters(max_rate=5.0).cluster_ids)
    assert (num_clusters - num_filtered_clusters) == expected_num_dropped


def test_split(dc_rec12):
    """Tests splitting a recording into multiple parts.

    Tests that:
        1. A split works.
        2. zero-value ratio causes an error.
    """
    # Test 1
    # Setup
    splits = (3, 1, 1)
    expected_len = 892863
    assert len(dc_rec12) == expected_len
    expected_split_lens = [535716 + 3, 178572, 178572]
    expected_split_lens_reversed = [178572 + 3, 178572, 535716]
    assert len(dc_rec12) == sum(expected_split_lens)
    # Test
    res = mea.split(dc_rec12, splits)
    assert len(res) == 3, "There should be 3 splits."
    assert [
        len(s) for s in res
    ] == expected_split_lens, "Splits should be the correct length."
    # Do it again but reversed.
    res = mea.split(dc_rec12, splits[::-1])
    assert len(splits) == 3, "There should be 3 splits."
    assert [
        len(s) for s in res
    ] == expected_split_lens_reversed, "Splits should be the correct length."

    # Test 2
    with pytest.raises(ValueError):
        mea.split(dc_rec12, (0, 1, 1))


def test_mirror_split(dc_rec12):
    """Tests splitting a recording into multiple parts (the mirrored version).

    Tests that:
        1. a (3, 1, 1) and (1, 1, 3) split works.
            - no errors
            - number of output datasets is correct
            - dataset sizes are as expected
        2. zero-value ratio causes an error.
        3. requesting less than two splits causes an error.
    """
    # Test 1
    # Setup
    splits = (3, 1, 1)
    expected_len = 892863
    assert len(dc_rec12) == expected_len
    # Note how the remainders fall in different places compared to the
    # non-mirrored split. This is due to the mirrored split calling split
    # twice, under the hood.
    expected_split_lens = [535716 + 1, 178572, 178572 + 2]
    expected_split_lens_reversed = [178572 + 1, 178572, 535716 + 2]
    assert len(dc_rec12) == sum(expected_split_lens)
    # Test
    res = mea.mirror_split(dc_rec12, splits)
    assert len(res) == 3, "There should be 3 splits."
    assert [len(s) for s in res] == [
        1,
        1,
        1,
    ], "Concatenation means there should be only single segments."
    assert [
        len(s[0]) for s in res
    ] == expected_split_lens, "Splits should be the correct length."
    # Do it again but reversed.
    res = mea.mirror_split(dc_rec12, splits[::-1])
    assert len(splits) == 3, "There should be 3 splits."
    assert [len(s) for s in res] == [
        1,
        1,
        1,
    ], "Concatenation means there should be only single segments."
    assert [
        len(s[0]) for s in res
    ] == expected_split_lens_reversed, "Splits should be the correct length."
    # Test 2
    with pytest.raises(ValueError):
        mea.split(dc_rec12, (0, 1, 1))

    # Test 3
    with pytest.raises(ValueError):
        mea.split(dc_rec12, (1,))
    with pytest.raises(ValueError):
        mea.split(dc_rec12, tuple())


def test_mirror_split__2(dc_rec12):
    """
    Invariant test for mirror_split.

    The other test inspects a specific case; here we test various splits and
    check for expected invariants.

    Tests that:
        - split lengths sum up to the length of the original.
        - splits have the same number of clusters.
        - the values (stimulus and spikes) in the first half of the first
            split match the beginning of the original.
    """
    # Setup
    seed = 123
    rng = np.random.default_rng(seed)
    num_trials = 20
    orig_len = len(dc_rec12)
    num_splits = rng.integers(low=2, high=10, size=num_trials)
    ratios = [
        rng.integers(low=1, high=20, size=num_splits[i])
        for i in range(num_trials)
    ]

    def check_sizes(splits):
        # The combined splits have the same number of timesteps as the original.
        total_len = sum([len(s[0]) for s in splits])
        assert total_len == orig_len
        # The cluster count is also the same.
        expected_num_clusters = len(dc_rec12.cluster_ids)
        for split in splits:
            assert len(split[0].cluster_ids) == expected_num_clusters

    def check_split1_values(splits):
        split1_len = splits[0][0].stimulus.shape[0]
        test_len = split1_len // 2
        orig_stim = dc_rec12.stimulus[:test_len]
        orig_spikes = dc_rec12.spikes[:test_len]
        np.testing.assert_array_equal(
            orig_stim, splits[0][0].stimulus[:test_len]
        )
        np.testing.assert_array_equal(
            orig_spikes, splits[0][0].spikes[:test_len]
        )

    # Test
    for i in range(num_trials):
        splits = mea.mirror_split(dc_rec12, ratios[i].tolist())
        check_sizes(splits)
        check_split1_values(splits)


def test_remove_few_spike_clusters(dc_rec12):
    """
    Tests removing clusters with few spikes.

    Tests that: for a single set of arguments, the filtering works and matches
    a precomputed result.
    """
    # Setup
    splits = (7, 2, 1)
    min_counts = (10, 5, 5)
    num_clusters = len(dc_rec12.cluster_ids)
    expected_num_removed = 27
    train_test_val_splits = mea.split(dc_rec12, splits)
    # Convert to a list of lists, as this is what's expected hereafter.
    train_test_val_splits = [[s] for s in train_test_val_splits]

    # Test
    filtered = mea.remove_few_spike_clusters(train_test_val_splits, min_counts)
    num_filtered_clusters = [s[0].num_clusters() for s in filtered]
    assert num_filtered_clusters == [num_clusters - expected_num_removed] * 3


def test_decompress_recording(rec12):
    """
    Test decompressing a recording.

    Tests that:
        1. Simple decompression works (basic checks).
        2. Multiple spikes in the same bucket is allowed.
    """
    # Test 1
    downsample = 18
    orig_freq = rec12.sensor_sample_rate
    expected_sample_rate = orig_freq / downsample
    res = mea.decompress_recording(rec12, downsample)
    assert res.sample_rate == pytest.approx(expected_sample_rate)

    # Test 2
    downsample = 100
    res = mea.decompress_recording(rec12, downsample)
    max_per_bucket = np.max(res.spikes)
    assert max_per_bucket == 3


def test_single_3brain_recording(
    stimulus_pattern, recorded_stimulus, response_data
):
    """
    Tests that:
        1. A recording can be loaded without errors (very basic checks).
        2. Filtering by cluster id works.
        3. Requesting non-existing clusters raises an error.
    """
    # Test 1
    # Setup
    expected_num_samples = 16071532
    # Test
    rec = mea._single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
    )
    assert rec.num_sensor_samples == expected_num_samples
    assert rec.stimulus_pattern.shape[1] == mea.NUM_STIMULUS_LEDS
    # Note: the sampling frequency in the Pandas dataframe isn't as accurate
    # as the ELECTRODE_FREQ value.
    assert rec.sensor_sample_rate == pytest.approx(mea.ELECTRODE_FREQ)

    # Test 2
    # Setup
    cluster_ids = {13, 14, 202, 1485}
    # Test
    rec = mea._single_3brain_recording(
        "Chicken_17_08_21_Phase_00",
        stimulus_pattern,
        recorded_stimulus,
        response_data,
        cluster_ids,
    )
    assert len(rec.spike_events) == len(cluster_ids)
    assert len(rec.cluster_ids) == len(cluster_ids)
    assert set(rec.cluster_ids) == cluster_ids

    # Test 3
    # Setup
    partially_existing_clusters = {
        13,  # Exists
        14,  # Exists
        18,  # Does not exist
    }
    # Test
    with pytest.raises(ValueError):
        mea._single_3brain_recording(
            "Chicken_17_08_21_Phase_00",
            stimulus_pattern,
            recorded_stimulus,
            response_data,
            partially_existing_clusters,
        )


def test_decompress_stimulus():
    """
    Tests that:
        1. Decompression with no downsample works.
        2. Decompression with downsample works.
            - This test overlaps a bit with test_downsample_stimulus
        3. Invalid trigger start should cause an exception.
    """
    # Setup
    # fmt: off
    stimulus_pattern = np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0, 1, 0]]).T
    trigger_events = np.array([0, 4, 5, 7, 12])
    num_sensor_samples = 20
    expected_output = np.array([
    # idx: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
    # val: 0  0  0  0  1  2  2  3  3  3  3  3  4  4  4  4  4  4  4  4
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]).T
    expected_downsampled = np.array([
    # idx: 0     2     4     6     8     10    12    14    16    18
    # val: 0     0     1     2     3     3     4     4     4     4 
          [0.7, 1.1,   1,    1,    1,    1,  0.3,    0,    0,    0],
          [0.7, 1.1, 0.7,    0,    0,    0,  0.7,    1.1,  1,    1],
          [0.7,   1, 0.5,  0.7,    0,    0,  0.7,    1.1,  1,    1]]).T
    # fmt: on
    # Notice how the first downsampled value is pulled towards zero. This
    # is due to the "constant" option sent to scipy.resample_poly, which is
    # called by scipy.resample with "0" as the constant value. There are other
    # options, such as "mean" too. If we want a different constant value
    # or want "mean" padding, we should call scipy.resapmle_poly directly.

    # Test 1
    downsample = 1
    res = mea.decompress_stimulus(
        stimulus_pattern, trigger_events, num_sensor_samples, downsample
    )
    np.testing.assert_array_equal(res, expected_output)

    # Test 2
    downsample = 2
    res = mea.decompress_stimulus(
        stimulus_pattern, trigger_events, num_sensor_samples, downsample
    )
    # Test approx with numpy
    numpy.testing.assert_allclose(res, expected_downsampled, atol=0.1)

    # Test 3
    # Non-zero starting trigger is invalid.
    trigger_events_invalid = np.array([2, 4, 5, 7, 12])
    with pytest.raises(ValueError):
        mea.decompress_stimulus(
            stimulus_pattern,
            trigger_events_invalid,
            num_sensor_samples,
            downsample,
        )


def test_factors_sorted_by_count():
    """
    Tests that:
        1. A simple case:
            1. There are no duplicates.
            2. The factors are sorted by count.
        2. Setting a limit will cause the list to be truncated.
        3. If no factorizations meet the limit requirement, a fallback is
            returned.
        4. For a prime, the parameter is returned as is.
    """
    # Test 1
    num = 12
    expected = ((2, 2, 3), (2, 6), (3, 4), (12,))
    # Test 1
    res = mea.factors_sorted_by_count(num)
    assert set(res) == set(expected)

    # Test 2
    num = 12
    expected = ((2, 2, 3), (3, 4))
    res = mea.factors_sorted_by_count(num, limit=5)
    assert set(res) == set(expected)
    expected = ((2, 2, 3), (2, 6), (3, 4))
    res = mea.factors_sorted_by_count(num, limit=6)
    assert set(res) == set(expected), "The limit should be inclusive."

    # Test 3
    num = 2 * 2 * 13 * 13
    # Note: note how the result is not really ideal (13, 13, 4) would be better.
    expected = ((2, 2, 13, 13),)
    res = mea.factors_sorted_by_count(num, limit=5)
    assert set(res) == set(expected)

    # Test 4
    num = 89
    expected = ((89,),)
    res = mea.factors_sorted_by_count(num)
    assert res == expected


def test_downsample_stimulus():
    # Setup
    # fmt: off
    orig_signal = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1])
    # TODO: is this satisfactory?
    expected_decimate_by_2 = np.array(
            [0.012, -0.019,  0.029, -0.060,
             0.739,  1.085,  0.937,  1.081,
             0.249, -0.076,  0.058, -0.077,
             0.750,  1.077,  0.942,  1.077,
             0.250, -0.077,  0.058, -0.076,
             0.749,  1.081,  0.937,  1.085])
    expected_decimate_by_4 = np.array(
            [0.030, -0.065,  0.595,  1.148,
             0.364, -0.120,  0.620,  1.136,
             0.369, -0.120,  0.614,  1.148])
    # fmt: on

    # Test
    decimated_by_2 = mea.downsample_stimulus(orig_signal, 2)
    numpy.testing.assert_allclose(
        decimated_by_2, expected_decimate_by_2, atol=0.002
    )
    decimated_by_4 = mea.downsample_stimulus(orig_signal, 4)
    numpy.testing.assert_allclose(
        decimated_by_4, expected_decimate_by_4, atol=0.002
    )


def test_compress_spikes():
    """
    Tests that the basic spikes to spike-indexes function works.
    """
    # Setup
    spikes = np.array([0, 0, 1, 0, 0, 2, 3])
    expected_indexes = np.array([2, 5, 5, 6, 6, 6])

    # Test
    res = mea.compress_spikes(spikes)
    np.testing.assert_array_equal(res, expected_indexes)


def test_decompress_spikes1():
    # Setup
    downsample_by = 9
    num_sensor_samples = 123
    # fmt: off
    spike_times1 = np.array([8, 9, 30, 40, 50, 70, 80, 90, 100, 110])
    spike_times2 = np.array([0, 1, 8, 9, 10, 27, 30, 40, 50, 70, 80, 90, 100, 110])

    spike_counts1 = np.array([1, 1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0])
    spike_counts2 = np.array([3, 2,  0,  2,  1,  1,  0,  1,  1,  0,  1,  1,  1,  0])
    # fmt: on
    expected_output_len = math.ceil(num_sensor_samples / downsample_by)
    assert len(spike_counts1) == len(spike_counts2) == expected_output_len

    # Test 1
    spikes = mea.decompress_spikes(
        spike_times1, num_sensor_samples, downsample_by
    )
    assert np.array_equal(spikes, spike_counts1)

    # Test 2
    # Test the case where two spikes land in the same bucket.
    # There should *not* be an error thrown, even though two samples land in
    # the same bucket.
    spikes = mea.decompress_spikes(
        spike_times2, num_sensor_samples, downsample_by
    )
    assert np.array_equal(spikes, spike_counts2)


def test_decompress_spikes2(response_data):
    """
    Test decompress_spikes on actual recordings.

    Not much is actually checked though.
    """
    # Setup
    cluster_id = 36
    spikes_row = response_data.xs(
        (cluster_id, "Chicken_17_08_21_Phase_00"),
        level=("Cell index", "Recording"),
    ).iloc[0]
    spikes = spikes_row["Spikes"].compressed()

    assert spikes.shape == (2361,)
    num_sensor_samples = spikes[-1] + 1000

    # Test
    mea.decompress_spikes(spikes, num_sensor_samples)
    mea.decompress_spikes(spikes, num_sensor_samples, downsample_factor=18)


def test_spike_snippets():
    """
    Tests that the spike snippets are extracted correctly.

    Specifically, the following cases are considered:

        1. The spike needs no padding.
        2. The spike needs padding at the beginning.
        3. The spike happens in the first sample.
        4. The spike needs padding at the end.
        5. The spike happens in the last sample.
    """
    # Setup
    # -----
    # fmt: off
    # Test:   1, 2, 3, 4, 5
    spikes = [4, 1, 0, 6, 7]
    # The numbers in comments refer to the 5 tests below.
    stimulus = np.array(
        [
            [1, 1, 1, 1],  #     |  3
            [1, 1, 1, 1],  #     2  |
            [0, 1, 1, 1],  #  -  |  -
            [0, 0, 1, 1],  #  |  -
            [0, 0, 0, 1],  #  1       -
            [0, 0, 0, 0],  #  |       |  -
            [1, 0, 0, 0],  #  -       4  |
            [1, 1, 0, 0],  #          |  5
        ]              
    )
    # fmt: on
    total_len = 5
    pad = 2

    def _test(spike, expected_snippet):
        """Collects the repetitive call and test into a function."""
        snippets = mea.spike_snippets(
            stimulus,
            [spike],
            total_len,
            pad,
        )
        assert (
            len(snippets) == 1
        ), "One spike was given, expect only one snippet."
        snippet = snippets[0]
        numpy.testing.assert_array_equal(snippet, expected_snippet)

    # Test 1: case where no padding is needed.
    _test(
        spikes[0],
        expected_snippet=np.array(
            [
                [0, 1, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
            ]
        ),
    )

    # Test 2: sample is near the beginning and needs padding.
    _test(
        spikes[1],
        expected_snippet=np.array(
            [
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 0, 1, 1],
            ]
        ),
    )

    # Test 3: sample is _at_ the beginning and needs padding.
    _test(
        spikes[2],
        expected_snippet=np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 1],
            ]
        ),
    )

    # Test 4: sample is near the end and needs padding.
    _test(
        spikes[3],
        expected_snippet=np.array(
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
    )

    # Test 5: sample is _at_ the end and needs padding.
    _test(
        spikes[4],
        expected_snippet=np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
    )


def test_labeled_spike_snippets():
    """
    Create a fake response `DataFrame` and check that the spike snippets are
    calculated correctly.
    """
    # Setup
    snippet_len = 7
    snippet_pad = 2
    # Fake stimulus pattern.
    stimulus_pattern = np.array(
        [
            [0, 0, 0, 1],  # 0
            [0, 0, 1, 0],  # 1
            [0, 0, 1, 1],  # 2
            [0, 1, 0, 0],  # 3
            [0, 1, 0, 1],  # 4
            [0, 1, 1, 0],  # 5
            [0, 1, 1, 1],  # 6
            [1, 0, 0, 0],  # 7
            [1, 0, 0, 1],  # 8
            [1, 0, 1, 0],  # 9
            [1, 0, 1, 1],  # 10
            [1, 1, 0, 0],  # 11
        ]
    )

    # Fake response
    rec_name1 = "Chicken1"
    rec_name2 = "Chicken2"
    cluster_ids1 = [25, 40]
    cluster_ids2 = [17, 40]
    spike_events1 = np.array(
        [
            [1, 8],
            [6, 9],
        ],
        dtype=int,
    )
    spike_events2 = np.array(
        [
            [11],
            [9],
        ],
        dtype=int,
    )
    # Make SpikeRecordings.
    stimulus_events = np.arange(len(stimulus_pattern))
    recording1 = mea.CompressedSpikeRecording(
        rec_name1,
        stimulus_pattern,
        stimulus_events,
        spike_events1,
        cluster_ids1,
        sensor_sample_rate=1,
        num_sensor_samples=len(stimulus_pattern),
    )
    recording2 = mea.CompressedSpikeRecording(
        rec_name2,
        stimulus_pattern,
        stimulus_events,
        spike_events2,
        cluster_ids2,
        sensor_sample_rate=1,
        num_sensor_samples=len(stimulus_pattern),
    )

    # The expected snippets.
    expected_spike_snippets1 = np.array(
        [
            [
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 1],  # 0
                [0, 0, 1, 0],  # 1 <-- spike
                [0, 0, 1, 1],  # 2
                [0, 1, 0, 0],  # 3
            ],
            [
                [0, 1, 0, 1],  # 4
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8 <- spike
                [1, 0, 1, 0],  # 9
                [1, 0, 1, 1],  # 10
            ],
            [
                [0, 0, 1, 1],  # 2
                [0, 1, 0, 0],  # 3
                [0, 1, 0, 1],  # 4
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6 <- spike
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
            ],
            [
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
                [1, 0, 1, 0],  # 9 <- spike
                [1, 0, 1, 1],  # 10
                [1, 1, 0, 0],  # 11
            ],
        ]
    )
    expected_spike_snippets2 = np.array(
        [
            [
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
                [1, 0, 1, 0],  # 9
                [1, 0, 1, 1],  # 10
                [1, 1, 0, 0],  # 11 <- spike
                [0, 0, 0, 0],  # pad
                [0, 0, 0, 0],  # pad
            ],
            [
                [0, 1, 1, 0],  # 5
                [0, 1, 1, 1],  # 6
                [1, 0, 0, 0],  # 7
                [1, 0, 0, 1],  # 8
                [1, 0, 1, 0],  # 9 <- spike
                [1, 0, 1, 1],  # 10
                [1, 1, 0, 0],  # 11
            ],
        ]
    )
    expected_cluster_ids1 = np.array([25, 25, 40, 40])
    expected_cluster_ids2 = np.array([17, 40])

    # Test 1 (rec_name1)
    spike_snippets, cluster_ids = mea.labeled_spike_snippets(
        recording1,
        snippet_len,
        snippet_pad,
    )
    for idx, (spwin, cluster_ids) in enumerate(
        zip(spike_snippets, cluster_ids)
    ):
        np.testing.assert_equal(spwin, expected_spike_snippets1[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids1[idx])

    # Test 2 (rec_name2)
    spike_snippets, cluster_ids = mea.labeled_spike_snippets(
        recording2, snippet_len, snippet_pad
    )
    for idx, (spwin, cluster_ids) in enumerate(
        zip(spike_snippets, cluster_ids)
    ):
        np.testing.assert_equal(spwin, expected_spike_snippets2[idx])
        np.testing.assert_equal(cluster_ids, expected_cluster_ids2[idx])
