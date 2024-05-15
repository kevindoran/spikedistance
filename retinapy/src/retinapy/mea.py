from collections import namedtuple
from collections import deque
import concurrent.futures
import itertools
import json
import logging
import math
import pathlib
from typing import (
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    Union,
)
import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import bidict


_logger = logging.getLogger(__name__)


ELECTRODE_FREQ = 17852.767845719834  # Hz
NUM_STIMULUS_LEDS = 4
STIMULUS_PATTERN_FILENAME = "stimulus_pattern.npy"
SPIKE_RESPONSE_FILENAME = "spike_response.pickle"
RECORDED_STIMULUS_FILENAME = "recorded_stimulus.pickle"
REC_IDS_FILENAME = "recording_ids.json"
REC_CLUSTER_IDS_FILENAME = "recording_cluster_ids.json"


RecClusterId: TypeAlias = Tuple[str, int]
# A dictionary, recording-name <-> id
RecIds: TypeAlias = bidict.BidirectionalMapping[str, int]
# A dictionary, (recording-name, cluster_id) <-> global cluster id
RecClusterIds: TypeAlias = bidict.BidirectionalMapping[RecClusterId, int]

Stimulus = namedtuple(
    "Stimulus", ["name", "wavelength", "channel", "display_hex", "import_name"]
)


stimuli = [
    Stimulus("red", 630, 0, "#FE7C7C", "/Red_Noise"),
    Stimulus("green", 530, 1, "#8AFE7C", "/Green_Noise"),
    Stimulus("blue", 480, 2, "#7CFCFE", "/Blue_Noise"),
    Stimulus("uv", 420, 3, "#7C86FE", "/UV_Noise"),
]
_stimulus_map = {s.name: s for s in stimuli}


def stimulus_by_name(name: str) -> Stimulus:
    """Return the stimulus info for the stimulus with the given name."""
    return _stimulus_map[name]


class CompressedSpikeRecording:
    """
    Data for a single recording, with spikes and stimulus stored as events.

    Storage as events means that a 10 bin long recording with a single spike
    at bin 5 would be have spike data = [5], as opposed to
    [0,0,0,0,1,0,0,0,0,0]. The recordings are stored in this format anyway,
    so this class is mostly just a wrapper around some Pandas dataframes. The
    SpikeRecording class is the decompressed version of this class.

    Some benefits of this class:

        - We are not sure what changes to the underlying dataframe we will be
          making in future, so this class provides a layer of abstraction.
        - The recording data is split into a triplet: (stimulus pattern,
          stimulus recording, spike recording) which typically needs to be
          moved around together. So combining them in a single place makes
          sense.
        - It's easy to make mistakes querying dataframes with the Pandas API.
          The queries can be done once here.


    Future work:

        - Currently, it's assumed that the stimulus pattern is full-field.
          Switching to a 2D stimulus might require some changes. At least we
          will need to decide the X-Y coordinate order and (0,0) position.
    """

    def __init__(
        self,
        name: str,
        stimulus_pattern: np.ndarray,
        stimulus_events: np.ndarray,
        spike_events: List[np.ndarray],
        cluster_ids: List[int],
        sensor_sample_rate: float,
        num_sensor_samples: int,
        cluster_gids: Optional[List[int]] = None,
    ):
        if len(spike_events) != len(cluster_ids):
            raise ValueError(
                f"Mismatch between number of cluster-spikes "
                f"({len(spike_events)}) and number of cluster ids "
                f"({len(cluster_ids)})."
            )
        self.name = name
        self.stimulus_pattern = stimulus_pattern
        self.cluster_ids = cluster_ids
        self.stimulus_events = stimulus_events
        self.spike_events = spike_events
        self.sensor_sample_rate = sensor_sample_rate
        self.num_sensor_samples = num_sensor_samples
        self.cluster_gids = (
            cluster_gids if cluster_gids is not None else cluster_ids
        )

    def __str__(self):
        res = (
            f"Recording: {self.name}, "
            f"sensor sample rate: {self.sensor_sample_rate} Hz, "
            f"num samples: {self.num_sensor_samples}, "
            f"duration: {self.duration():.1f} seconds, "
            f"stimulus pattern shape: {self.stimulus_pattern.shape},"
            f"num clusters: {len(self.cluster_ids)}."
        )
        return res

    def duration(self):
        """Duration of the recording in seconds."""
        return self.num_sensor_samples / self.sensor_sample_rate

    def clusters(self, cluster_ids: Set[int]) -> "CompressedSpikeRecording":
        """Returns a new recording with only the specified clusters."""
        if not cluster_ids.issubset(self.cluster_ids):
            raise ValueError(
                f"Cluster ids ({cluster_ids}) are not a subset of "
                f"the cluster ids in this recording ({self.cluster_ids})."
            )
        cluster_indices = [self.cluster_ids.index(i) for i in cluster_ids]
        spike_events = [self.spike_events[i] for i in cluster_indices]
        cluster_gids = [self.cluster_gids[i] for i in cluster_indices]
        return CompressedSpikeRecording(
            self.name,
            self.stimulus_pattern,
            self.stimulus_events,
            spike_events,
            list(cluster_ids),
            self.sensor_sample_rate,
            self.num_sensor_samples,
            cluster_gids,
        )

    def num_clusters(self) -> int:
        return len(self.cluster_ids)

    def filter_clusters(
        self,
        min_rate: Optional[float] = None,
        max_rate: Optional[float] = None,
        min_count: Optional[int] = None,
    ) -> "CompressedSpikeRecording":
        """Filter out clusters with spike rates outside the given range.

        A new recording is returned.
        """
        matching_clusters = set()

        def _spike_rate(spike_events):
            assert len(spike_events.shape) == 1
            res = len(spike_events) / self.duration()
            return res

        for i in range(len(self.spike_events)):
            min_rate_match = (
                min_rate is None
                or _spike_rate(self.spike_events[i]) >= min_rate
            )
            max_rate_match = (
                max_rate is None
                or _spike_rate(self.spike_events[i]) <= max_rate
            )
            min_count_match = (
                min_count is None or len(self.spike_events[i]) >= min_count
            )
            is_match = min_rate_match and max_rate_match and min_count_match
            if is_match:
                matching_clusters.add(self.cluster_ids[i])
        return self.clusters(matching_clusters)

    def cid_to_idx(self, cid: int) -> int:
        return self.cluster_ids.index(cid)


class SpikeRecording:
    """
    Data for a recording, with a spike and stimulus value for each time bin.

    The class allows indexing by time and cluster, so an object may represent
    a part of a recording, and isn't limited to be a whole recording.

    A 10 bin long recording with a single spike at bin 5 would be have spike
    data = [0,0,0,0,1,0,0,0,0,0], as opposed to [5]. The stimulus is stored
    similarly.

    This class is mostly just a wrapper around the already decompressed data.
    It's useful for the same reasons as the CompressedSpikeRecording class.

    Future work:

        - I'm slowly adding functionality to make this class more array like.
          The idea here is that the spike data and stimulus data have a common
          time dimension, which it is useful index and slice over, but
          otherwise their dimensions don't match. xarray would be a suitable
          off-the-shelf solution for this. xarray is probably best, but while
          the required functionality of this class is still minimal, the
          current solution seems fine, and saves having to require knowledge of
          yet another indexing API in addition to the currently used numpy,
          Pandas and PyTorch.
    """

    name: str
    stimulus: np.ndarray
    spikes: np.ndarray
    cluster_ids: List[int]
    """
    The globally unique cluster IDs. These will be the same as the cluster
    IDs if a cluster_gid_map is not provided when initialing the Recording.
    """
    cluster_gids: List[int]
    sample_rate: float

    def __init__(
        self,
        name,
        stimulus,
        spikes,
        cluster_ids: List[int],
        sample_rate: float,
        cluster_gids: Optional[List[int]] = None,
    ):
        if len(stimulus) != len(spikes):
            raise ValueError(
                f"Length of stimulus ({len(stimulus)}) and length"
                f" of response ({len(spikes)}) do not match."
            )
        if spikes.shape[1] != len(cluster_ids):
            raise ValueError(
                f"Mismatch between number of cluster-spikes "
                f"({spikes.shape[1]}) and number of cluster ids "
                f"({len(cluster_ids)})."
            )
        self.name = name
        self.stimulus = stimulus
        self.spikes = spikes
        self.cluster_ids = cluster_ids
        self.sample_rate = sample_rate
        self.cluster_gids = (
            cluster_gids if cluster_gids is not None else cluster_ids
        )

    def __str__(self):
        res = (
            f"Recording: {self.name}, "
            f"sample rate: {self.sample_rate:.1f} Hz "
            f"({1000/self.sample_rate:.3f} ms per sample), "
            f"duration: {self.duration():.1f} seconds, "
            f"stimulus shape: {self.stimulus.shape}, "
            f"spikes shape: {self.spikes.shape}, "
            f"num clusters: {len(self.cluster_ids)}."
        )
        return res

    def __len__(self):
        assert len(self.stimulus) == len(self.spikes)
        return len(self.stimulus)

    def duration(self) -> float:
        return self.stimulus.shape[0] / self.sample_rate

    def num_clusters(self) -> int:
        return len(self.cluster_ids)

    def __getitem__(self, key):
        """Return a new recording with only data for the given time bin."""
        return SpikeRecording(
            self.name,
            self.stimulus[key],
            self.spikes[key],
            self.cluster_ids,
            self.sample_rate,
            self.cluster_gids,
        )

    def clusters(self, cluster_ids: Set[int]):
        """Returns a new recording with only the specified clusters."""
        if not cluster_ids.issubset(self.cluster_ids):
            raise ValueError(
                f"Cluster ids ({cluster_ids}) are not a subset of "
                f"the cluster ids in this recording ({self.cluster_ids})."
            )
        cluster_indices = [self.cluster_ids.index(i) for i in cluster_ids]
        spikes = self.spikes[:, cluster_indices]
        gids = [self.cluster_gids[i] for i in cluster_indices]
        return SpikeRecording(
            self.name,
            self.stimulus,
            spikes,
            list(cluster_ids),
            self.sample_rate,
            gids,
        )

    def extend(self, recording):
        """Add the given recording to the end of this one.

        This was added in order to create test, train and validation sets that
        pick their data from multiple parts of the same recording.
        """
        if self.sample_rate != recording.sample_rate:
            raise ValueError(
                f"Sample rates do not match. Got ({self.sample_rate}) and "
                f"({recording.sample_rate})."
            )
        if self.stimulus.shape[1] != recording.stimulus.shape[1]:
            raise ValueError(
                f"The stimulus must have the same number of LEDs. Got "
                f"({self.stimulus.shape[1]}) and "
                f"({recording.stimulus.shape[1]})."
            )
        if not np.array_equal(self.cluster_ids, recording.cluster_ids):
            raise ValueError(
                f"Cluster ids do not match. Got ({self.cluster_ids}) and "
                f"({recording.cluster_ids})."
            )
        if not np.array_equal(self.cluster_gids, recording.cluster_gids):
            raise ValueError(
                f"Cluster global ids do not match. Got ({self.cluster_gids}) "
                f"and ({recording.cluster_gids})."
            )
        self.stimulus = np.concatenate((self.stimulus, recording.stimulus))
        self.spikes = np.concatenate((self.spikes, recording.spikes))
        return self

    def spike_snippets(self, cid: int, total_len: int, post_spike_len: int):
        res = spike_snippets(
            self.stimulus,
            compress_spikes(self.spikes[:, self.cluster_ids.index(cid)]),
            total_len,
            post_spike_len,
        )
        return res

    def all_spike_snippets(self, total_len: int, post_spike_len: int):
        snippets_by_cluster = {}
        # Can this be done in a single call?
        # Could make compress_spikes operate on 2d array, then sent all
        # spikes to spike_snippets then split with np.split().
        for c in range(self.spikes.shape[1]):
            snippets_by_cluster[self.cluster_ids[c]] = spike_snippets(
                self.stimulus,
                compress_spikes(self.spikes[:, c]),
                total_len,
                post_spike_len,
            )
        return snippets_by_cluster

    def cid_to_idx(self, cid: int) -> int:
        return self.cluster_ids.index(cid)


def split(recording: SpikeRecording, split_ratio: Sequence[int]):
    """Split a recording into multiple recordings.

    Args:
        split_ratio: a list of weightings that determines how much data to
            give each split. For example, you might use the triplet (3, 1, 1)
            to create a train-val-test split with 60% of the data for training,
            20% for validation, and 20% for testing.
    """
    if len(split_ratio) < 2:
        raise ValueError("Can't split a recording into fewer than 2 parts.")
    if not all([r > 0 for r in split_ratio]):
        raise ValueError(f"Split ratios must be positive. Got ({split_ratio}).")
    divisions = sum(split_ratio)
    num_per_division, remainder = divmod(len(recording), divisions)
    splits = []
    slice_start = slice_end = 0
    for i in range(len(split_ratio)):
        if i == 0:
            # Give all of the remainder to the first split.
            slice_end = split_ratio[i] * num_per_division + remainder
        else:
            slice_end += split_ratio[i] * num_per_division
        splits.append(recording[slice_start:slice_end])
        slice_start = slice_end
    total_len = sum([len(s) for s in splits])
    assert total_len == len(
        recording
    ), f"Split lengths do not match ({total_len}) vs ({len(recording)})."
    return splits


def mirror_split(recording: SpikeRecording, split_ratio: Sequence[int]):
    """Split data from "outside-in".

    This approach of splitting the data tries to address the issue that the
    response of the retina will change over time. Training the model with the
    earlier responses and validating or testing with the later responses will
    likely lead to reduced accuracy that is not the fault of the model.

    One way to ammeliorate this issue is to give each split a piece of the
    earlier data and a piece of the later data. There is still some flexibility
    in how to make this choice. If there are three splits (train, val, test),
    then they will be made as follows:

        +-------+------+----+----+------+-------+
        | train | val  |  test   | val  | train |
        +-------+------+----+----+------+-------+

    The benefit of this approach is that each split is not exposed soley to
    one end of the data. Additionally, the test data, which will often appear
    in reports will still exist as a continuous block of data.

    Args:
        split_ratio: a list of weightings that determines how much data to
            give each split. For example, you might use the triplet (3, 1, 1)
            to create a train-val-test split with 60% of the data for training,
            20% for validation, and 20% for testing.
    """
    if not all([r > 0 for r in split_ratio]):
        raise ValueError(f"Split ratios must be positive. Got ({split_ratio}).")
    recording_half1 = recording[: len(recording) // 2]
    recording_half2 = recording[len(recording) // 2 :]
    assert len(recording_half1) + len(recording_half2) == len(recording)
    splits_half1 = split(recording_half1, split_ratio)
    splits_half2 = split(recording_half2, tuple(reversed(split_ratio)))
    splits = [
        s1.extend(s2) for (s1, s2) in zip(splits_half1, reversed(splits_half2))
    ]
    total_len = sum([len(s) for s in splits])
    assert total_len == len(
        recording
    ), f"Split lengths do not match ({total_len}) vs ({len(recording)})."
    # Finally, to match the datatype of mirror_split2, each element is a list.
    splits = [[s] for s in splits]
    return splits


def mirror_split2(recording: SpikeRecording, split_ratio: Sequence[int]):
    """Split data from "outside-in".

    Same as mirror_split1, but returns lists of recordings per-split instead of
    concatenating them together.

    Args:
        split_ratio: a list of weightings that determines how much data to
            give each split. For example, you might use the triplet (3, 1, 1)
            to create a train-val-test split with 60% of the data for training,
            20% for validation, and 20% for testing.
    """
    if not all([r > 0 for r in split_ratio]):
        raise ValueError(f"Split ratios must be positive. Got ({split_ratio}).")
    mirrored_ratios = []
    # [7, 2, 1] -> [7, 2, 2, 2, 7]
    mirrored_ratios.extend(split_ratio[:-1])
    mirrored_ratios.append(split_ratio[-1] * 2)
    mirrored_ratios.extend(reversed(split_ratio[:-1]))
    split_parts = split(recording, mirrored_ratios)
    assert len(split_parts) % 2 == 1, "Expecting odd number of splits"
    # [first, last], [second, second-last], ...
    # zip returns generator(tuple), and we want list[list[]].
    splits = [list(pair) for pair in zip(split_parts[:len(split_parts)//2], 
                      reversed(split_parts[len(split_parts)//2:]))]
    splits.append([split_parts[len(split_parts) // 2]])
    total_len = sum([len(sp) for s in splits for sp in s])
    assert total_len == len(
        recording
    ), f"Split lengths do not match ({total_len}) vs ({len(recording)})."
    return splits


def remove_few_spike_clusters(splits : List[List[SpikeRecording]], min_counts):
    """Skip clusters that have few spikes in one of the data splits."""
    if len(splits) != len(min_counts):
        raise ValueError(
            "splits and min_counts should have the same shape. Got "
            f"({len(splits)} vs, {len(min_counts)})."
        )
    num_cids = len(splits[0][0].cluster_ids)
    to_keep = np.ones(shape=num_cids, dtype=bool)
    for rs, min_count in zip(splits, min_counts):
        spikes = np.concatenate([r.spikes for r in rs])
        np.logical_and(np.sum(spikes, axis=0) >= min_count, to_keep, to_keep)
    filtered_splits = tuple(
        [s.clusters(set(np.array(s.cluster_ids)[to_keep])) for s in split_parts]
        for split_parts in splits
    )
    return filtered_splits


def load_stimulus_pattern(file_path: Union[pathlib.Path, str]) -> np.ndarray:
    file_path = pathlib.Path(file_path)
    res = np.load(file_path)
    return res


def load_response(
    file_path: Union[pathlib.Path, str], keep_kernels=False
) -> pd.DataFrame:
    """Loads the spike data from a Pickle file as a Pandas DataFrame.

    The input path should point to standard pickle file (zipped or not).

    Dataframe structure
    -------------------
    The dataframe uses a multiindex: ['Cell index', 'Stimulus ID', 'Recording'].
        - The cell index refers to the id assigned by the spike sorter.
        - TODO: The stimulus ID is what?
        - The 'Recording' index is a human readable string to identify a
            recording session.

    The Dataframe has 2 columns: ['Kernel', 'Spikes'].
        - The each row contains a (2000,4) shaped Numpy array holding the
          precomputed response kernel for the matching cell-stimulus-recording.
          The kernel represents 2000 miliseconds, with the spike located at
          1000 ms.
        - The 'Spikes' column contains a (17993,) shaped *masked* Numpy array,
          holding a variable number of integers. Each integer reprents the
          time (in sensor readings) at which the spike occurred.
    """
    res = pd.read_pickle(file_path)
    if not keep_kernels:
        res.drop("Kernel", axis=1)
    return res


def load_recorded_stimulus(file_path: Union[pathlib.Path, str]) -> pd.DataFrame:
    """Load the spike data from a Pickle file as a Pandas DataFrame."""
    res = pd.read_pickle(file_path, compression="infer")
    return res


def stim_and_spike_rows(
    rec_name: str, stimulus_df: pd.DataFrame, response_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Return the stimulus row and spike rows corresponding to a single recording.
    """
    if "Stimulus_index" in stimulus_df.index.names:
        stimulus_df = stimulus_df.droplevel("Stimulus_index")
    if "Stimulus ID" in response_df.index.names:
        response_df = response_df.droplevel("Stimulus ID")
    if stimulus_df.index.names != ["Recording"]:
        raise ValueError(
            'Expected only a single index level, "Recording". '
            f"Got {stimulus_df.index.names}."
        )
    if response_df.index.names != ["Cell index", "Recording"]:
        raise ValueError(
            'Expected two index levels, "Cell index" and "Recording".'
            f"Got {response_df.index.names}."
        )

    stim_row = stimulus_df.loc[rec_name]
    if len(stim_row.shape) != 1:
        _logger.warning(
            "There should be only one FF_Noise stimulus recording. "
            f"Got ({len(stim_row)}). For recording name: ({rec_name}). "
            f"Using the first only."
        )
        stim_row = stim_row.iloc[0]
        # raise ValueError(
        #     "There should be only one stimulus recording. "
        #     f"Got ({len(stim_row)}). For recording name: ({rec_name})"
        # )
    response_rows = response_df.xs(rec_name, level="Recording")
    return stim_row, response_rows


def _assert_dir_exists(p: pathlib.Path):
    if not p.exists():
        raise ValueError(f"Directory does not exist: ({p}).")
    if not p.is_dir():
        raise ValueError(f"Path is not a directory: ({p}).")


def _assert_file_exists(p: pathlib.Path):
    if not p.exists():
        raise ValueError(f"File does not exist: ({p}).")
    if not p.is_file():
        raise ValueError(f"Path is not a file: ({p}).")


def _load_data_files(data_dir: pathlib.Path):
    stimulus_pattern_path = data_dir / STIMULUS_PATTERN_FILENAME
    stimulus_recording_path = data_dir / RECORDED_STIMULUS_FILENAME
    response_recording_path = data_dir / SPIKE_RESPONSE_FILENAME
    _assert_file_exists(stimulus_pattern_path)
    _assert_file_exists(stimulus_recording_path)
    _assert_file_exists(response_recording_path)
    stim_pattern = load_stimulus_pattern(stimulus_pattern_path)
    stim_recordings = load_recorded_stimulus(stimulus_recording_path)
    response_recordings = load_response(response_recording_path)
    return stim_pattern, stim_recordings, response_recordings


def _rec_cluster_ids(recs: Iterable[CompressedSpikeRecording]):
    """Create a rec_name -> cluster_id -> ID mapping."""
    res = {}
    tally = 0
    for r in recs:
        res[r.name] = {}
        for c_id in r.cluster_ids:
            res[r.name][c_id] = tally
            tally += 1
    return res


def save_id_info(
    recs: Iterable[CompressedSpikeRecording],
    rec_id_map: Dict[str, int],
    data_dir: pathlib.Path | str,
):
    """
    Saves two dictionaries to disk:
        1. recording-name -> recording-id
        2. recording-name -> cluster-id -> ID.

    The purpose of the loading and saving of id information is to maintain a
    consistent way of referring to recordings and clusters.
    An example of where this is needed is when training a multi-cluster model
    on a subset of the data. This subset could have been selected manually
    and/or automatically by the filtering proceedure that removes unsuitable
    clusters, such as those with too few spikes. It is important to be able to
    obtain the embeddings for the clusters that were trained on, but not the
    ones that were not. Without a global ID, a snapshot of the data along with
    the filtering routine used while training would be needed.
    given to the cell data by the spike sorter is also not app
    """
    data_dir = pathlib.Path(data_dir)
    _assert_dir_exists(data_dir)
    with open(data_dir / REC_IDS_FILENAME, "w") as f:
        json.dump(rec_id_map, f)
    recs = sorted(recs, key=lambda r: rec_id_map[r.name])
    with open(data_dir / REC_CLUSTER_IDS_FILENAME, "w") as f:
        json.dump(_rec_cluster_ids(recs), f)


def has_id_info(data_dir: pathlib.Path | str) -> bool:
    data_dir = pathlib.Path(data_dir)
    dir_exists = data_dir.exists() and data_dir.is_dir()
    rec_id_file_exists = (data_dir / REC_IDS_FILENAME).exists()
    cluster_id_file_exists = (data_dir / REC_CLUSTER_IDS_FILENAME).exists()
    res = dir_exists and rec_id_file_exists and cluster_id_file_exists
    return res


def load_id_info(data_dir: pathlib.Path | str) -> Tuple[RecIds, RecClusterIds]:
    """Loads two dictionaries from disk, stored as JSON.

    The two mappings are:
        1. recording-name -> recording-id
        2. (recording-name, cluster-id) -> ID.
    """
    data_dir = pathlib.Path(data_dir)
    _assert_dir_exists(data_dir)
    rec_id_path = data_dir / REC_IDS_FILENAME
    rec_cluster_id_path = data_dir / REC_CLUSTER_IDS_FILENAME
    _assert_file_exists(rec_id_path)
    _assert_file_exists(rec_cluster_id_path)
    with open(rec_id_path, "r") as f:
        rec_ids = json.load(f)
        # Convert from normal to bi-directional dictionary.
        rec_ids = bidict.bidict(rec_ids)
    with open(rec_cluster_id_path, "r") as f:
        rec_cluster_ids = json.load(f)
    # The recording-cluster map will be flattened to take tuples.
    flat_rec_cluster_ids = bidict.bidict()
    for r_name, d1 in rec_cluster_ids.items():
        for c_id, c_id_flat in d1.items():
            # Don't forget to convert c_id to int. It became a string as
            # JSON only allows string dictionary keys.
            c_id = int(c_id)
            assert type(c_id) == type(c_id_flat) == int
            flat_rec_cluster_ids[(r_name, c_id)] = c_id_flat
    return rec_ids, flat_rec_cluster_ids


def load_3brain_recordings(
    data_dir: Union[str, pathlib.Path],
    include: Optional[Iterable[str]] = None,
    num_workers: int = 4,
) -> List[CompressedSpikeRecording]:
    """
    Creates a CompressedSpikeRecording for each recording in the 3Brain data.

    Args:
        include: a list of recording names to include.
        num_workers: how many threads can be used to load the data.
    """
    # Load the data. Do it once here.
    data_dir = pathlib.Path(data_dir)
    (
        stimulus_pattern,
        stimulus_recordings,
        response_recordings,
    ) = _load_data_files(data_dir)
    if has_id_info(data_dir):
        _, rec_cluster_ids = load_id_info(data_dir)
    else:
        rec_cluster_ids = None
    rec_names = recording_names(response_recordings)
    # Collect the recordings to load.
    to_load = []
    for rec_name in rec_names:
        do_load = include is None or rec_name in include
        if do_load:
            to_load.append(rec_name)
    done = []

    def _load(rec_name):
        rec_obj = _single_3brain_recording(
            rec_name,
            stimulus_pattern,
            stimulus_recordings,
            response_recordings,
            rec_cluster_ids=rec_cluster_ids,
        )
        return rec_obj

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        future_to_name = {
            executor.submit(_load, rec_name): rec_name for rec_name in to_load
        }
        for future in concurrent.futures.as_completed(future_to_name):
            rec_name = future_to_name[future]
            try:
                data = future.result()
            except Exception as exc:
                print(f"Loading {rec_name} generated an exception: {exc}.")
            else:
                done.append(data)
    return done


def single_3brain_recording(
    rec_name: str,
    data_dir: Union[pathlib.Path, str],
    include_clusters: Optional[Set[int]] = None,
) -> CompressedSpikeRecording:
    data_dir = pathlib.Path(data_dir)
    (
        stimulus_pattern,
        stimulus_recordings,
        response_recordings,
    ) = _load_data_files(data_dir)
    if has_id_info(data_dir):
        _, rec_cluster_ids = load_id_info(data_dir)
    else:
        rec_cluster_ids = None
    return _single_3brain_recording(
        rec_name,
        stimulus_pattern,
        stimulus_recordings,
        response_recordings,
        include_clusters,
        rec_cluster_ids
    )


def _single_3brain_recording(
    rec_name: str,
    stimulus_pattern: np.ndarray,
    stimulus_recordings: pd.DataFrame,
    response_recordings: pd.DataFrame,
    include_clusters: Optional[Set[int]] = None,
    rec_cluster_ids: Optional[RecClusterIds] = None,
) -> CompressedSpikeRecording:
    stimulus_row, response_rows = stim_and_spike_rows(
        rec_name, stimulus_recordings, response_recordings
    )
    # Handle stimulus.
    if stimulus_row["Stimulus_name"] != "FF_Noise":
        raise ValueError(
            f'Only stimulus type "FF_NOISE" is currently '
            f'supported. Got ({stimulus_row["Stimulus_name"]})'
        )
    if include_clusters and not include_clusters.issubset(
        set(response_rows.index)
    ):
        raise ValueError(
            f"Cluster ids ({include_clusters}) are not a subset of "
            f"the cluster ids in this recording {tuple(response_rows.index)}."
        )
    # TODO: is 'End_Fr' inclusive? If so, add 1 below. Assuming yes for now.
    stimulus_events = stimulus_row["Trigger_Fr_relative"].astype(int)
    num_samples = stimulus_events[-1] + 1
    # Remove the last event, as it's the end.
    stimulus_events = stimulus_events[:-1]
    sensor_sample_rate = stimulus_row["Sampling_Freq"]

    # Handle spikes.
    spikes_per_cluster = []
    cluster_ids = []
    for cluster_id, cluster_row in response_rows.iterrows():
        if include_clusters is not None and cluster_id not in include_clusters:
            continue
        spikes = cluster_row["Spikes"].compressed().astype(int)
        assert not np.ma.isMaskedArray(spikes), "Don't forget to decompress!"
        spikes_per_cluster.append(spikes),
        cluster_ids.append(cluster_id)

    # Optionally populate global cluster ids.
    if rec_cluster_ids is not None:
        gids = []
        for cid in cluster_ids:
            gids.append(rec_cluster_ids[(rec_name, cid)])
    else:
        gids = None

    # Create return object.
    res = CompressedSpikeRecording(
        rec_name,
        stimulus_pattern,
        stimulus_events,
        spikes_per_cluster,
        cluster_ids,
        sensor_sample_rate,
        num_samples,
        cluster_gids=gids,
    )
    return res


def decompress_recordings(
    recordings: Iterable[CompressedSpikeRecording],
    downsample: int = 1,
    num_workers=10,
) -> Deque[SpikeRecording]:
    """
    Decompress multiple recordings.

    For when you want multithreading and don't want to handle futures, use this.
    The downside is that you can't chain the results; you get the results all
    in one go. Chaining is useful if the next step involves filtering, and the
    decompressed recordings are too large to fit in memory.
    """

    def _decompress(rec):
        return decompress_recording(rec, downsample)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
        res = deque(executor.map(_decompress, recordings))
        executor.shutdown(wait=True)
    return res


def decompress_recording(
    recording: CompressedSpikeRecording, downsample: int
) -> SpikeRecording:
    """
    Decompress a compressed recording.

    The result holds numpy arrays where the first dimension is time.

    Without downsampling, a single recording take up more than 1 gigabyte of
    memory. It's quite convenient to set the  downsample to 18, as this will
    cause the resulting sample rate to be 991.8 Hz, which is the closest you
    can get to 1 kHz, given the 3Brain electrode's original frequency.
    """
    sample_rate = recording.sensor_sample_rate / downsample
    stimulus = decompress_stimulus(
        recording.stimulus_pattern,
        recording.stimulus_events,
        recording.num_sensor_samples,
        downsample,
    )
    spikes = np.stack(
        [
            decompress_spikes(s, recording.num_sensor_samples, downsample)
            for s in recording.spike_events
        ],
        axis=1,
    )
    res = SpikeRecording(
        recording.name,
        stimulus,
        spikes,
        recording.cluster_ids,
        sample_rate,
        recording.cluster_gids,
    )
    return res


def decompress_stimulus(
    stimulus_pattern: np.ndarray,
    trigger_events: np.ndarray,
    total_length: int,
    downsample: int,
) -> np.ndarray:
    if trigger_events[0] != 0:
        raise ValueError(
            "The trigger events are expected to start at zero, "
            f"but the first trigger was at ({trigger_events[0]})."
        )
    if len(trigger_events) > len(stimulus_pattern):
        raise ValueError(
            "Recorded stimulus is longer than the stimulus "
            f"pattern. ({len(trigger_events)} > {len(stimulus_pattern)})"
        )
    # TODO: check assumption! Assuming that the stimulus does not continue
    # after the last trigger event. This makes the last trigger special in that
    # it doesn't mark the start of a new stimulus output.
    # TODO: Marvin says that the last trigger event isn't the end of the
    # recording, but the last stimulus event before the end of the recording.
    # This introduces issues in that it's not clear how long this final
    # stimulus event is. I'm keeping the functionality as it is now, as I think
    # a better solution is to encode the recording end in the last stimulus
    # event, which would make the existing code below work fine.
    _logger.info(
        f"Starting: decompressing stimulus. Resulting shape ({total_length})."
    )
    # If this becomes a bottleneck, there are some tricks to reach for:
    # https://stackoverflow.com/questions/60049171/fill-values-in-numpy-array-that-are-between-a-certain-value
    num_channels = stimulus_pattern.shape[1]
    res = np.empty(shape=(total_length, num_channels))
    # Pair each trigger with the next trigger, and do this for all but the last.
    slices = np.stack((trigger_events[:-1], trigger_events[1:]), axis=1)
    for idx, s in enumerate(slices):
        res[np.arange(*s)] = stimulus_pattern[idx]
    last_trigger = trigger_events[-1]
    res[last_trigger:] = stimulus_pattern[len(slices)]
    _logger.info(
        f"Finished: decompressing stimulus. "
        f"The last trigger was at ({trigger_events[-1]}) making its "
        f"duration ({res.shape[0]} - {trigger_events[-1]} = "
        f"{res.shape[0] - trigger_events[-1]}) samples."
    )
    res = downsample_stimulus(res, downsample)
    return res


def rebin_spikes(spike_idxs: np.ndarray, downsample_factor: int) -> np.ndarray:
    """Calculate the spike indices after downsampling.

    Args:
        spike_idxs: The spike indices before downsampling.

    This method is very simple: just floor divide the indicies. It's good to
    have a dedicated function just so we are clear what the behaviour is,
    and to insure that we are doing it consistently.
    """
    res = np.floor_divide(spike_idxs, downsample_factor)
    return res


def decompress_spikes(
    spikes: Union[np.ndarray, np.ma.MaskedArray],
    num_sensor_samples: int,
    downsample_factor: int = 1,
) -> np.ndarray:
    """
    Fills an integer array counting the number of spikes that occurred.

    Setting downsample_factor to an integer greater than 1 will result in
    the spikes being counted in larger bin sizes that the original sensor
    sample period. So we are not talking about signal downsampling, Nyquist
    rates etc., rather we are talking about histogram binning where the bin
    size is scaled by downsample_factor. This behaviour is similar to Pandas's
    resample().sum() pattern.

    Binning behaviour
    -----------------
    As only integer values are accepted for downsample_factor, the binning is
    achieved by floor division of the original spike index. Examples:
        1. Input: [0, 0, 0, 1, 1], downsample_factor=2, output: [0, 1, 1]
        1. Input: [0, 0, 0, 1, 1, 1], downsample_factor=2, output: [0, 1, 2]
    """
    if np.ma.isMaskedArray(spikes):
        spikes = spikes.compressed()
    downsampled_spikes = rebin_spikes(spikes, downsample_factor)
    res = np.zeros(
        shape=[
            math.ceil(num_sensor_samples / downsample_factor),
        ],
        dtype=int,
    )
    np.add.at(res, downsampled_spikes, 1)
    return res


def factors_sorted_by_count(
    n, limit: Optional[int] = None
) -> Tuple[Tuple[int, ...]]:
    """
    Calulates factor decomposition with sort and limit.

    This method is used to choose downsampling factors when a single factor
    is too large. The decompositions are sorted by the number of factors in a
    decomposition. With this in mind, when a factorization cannot be found
    that has all factors

    Args:
        n: The number to decompose.
        limit: The maximum number of factors allowed in a single decomposition.
            This is an inclusive limit. If there are no decompositions t
    """

    def _factors(n):
        # Use a set to avoid duplicate factors.
        res = {(n,)}
        f1 = n // 2
        while f1 > 1:
            f2, mod = divmod(n, f1)
            if not mod:
                res.add(tuple(sorted((f1, f2))))
                for (a, b) in itertools.product(_factors(f1), _factors(f2)):
                    sub_factors = tuple(sorted(a + b))
                    res.add(sub_factors)
            f1 -= 1
        return res

    factors = _factors(n)
    sorted_by_count = tuple(sorted(factors, key=lambda x: len(x)))
    # If there is a limit set, remove any factor decompositions that contain a
    # factor larger than the limit.
    if limit:
        factors_filtered = tuple(f for f in factors if max(f) <= limit)
        if not factors_filtered:
            _logger.info(
                f"No factor decomposition exists with factors under "
                f"{limit}. Returning the decomposition with the "
                f"most factors."
            )
            sorted_by_count = (sorted_by_count[-1],)
        else:
            sorted_by_count = factors_filtered
    return sorted_by_count


def downsample_stimulus(stimulus: np.ndarray, factor: int) -> np.ndarray:
    """
    Filter (low-pass) a stimulus and then decimate by a factor.

    This is needed to prevent aliasing.

    Resources on filtering
    ----------------------
    https://dsp.stackexchange.com/questions/45446/pythons-tt-resample-vs-tt-resample-poly-vs-tt-decimate
    https://dsp.stackexchange.com/questions/83696/downsample-a-signal-by-a-non-integer-factor
    https://dsp.stackexchange.com/questions/83889/decimate-a-signal-whose-values-are-calculated-not-stored?noredirect=1#comment176944_83889
    """
    if factor == 1:
        return stimulus
    time_axis = 0
    _logger.info(
        f"Starting: downsampling by {factor}. Initial length "
        f"{stimulus.shape[time_axis]:,}."
    )
    # SciPy recommends to never exceed 13 on a single decimation call.
    # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.decimate.html
    MAX_SINGLE_DECIMATION = 13
    sub_factors = factors_sorted_by_count(factor, limit=MAX_SINGLE_DECIMATION)[
        0
    ]
    for sf in sub_factors:
        _logger.info(f"Starting: decimating by {sf}")
        stimulus = scipy.signal.decimate(
            stimulus, sf, ftype="fir", axis=time_axis
        )
        _logger.info(f"Finished: decimating by {sf}")
    _logger.info(
        f"Finished: downsampling. Resulting length "
        f"({stimulus.shape[time_axis]:,})."
    )
    return stimulus


def recording_names(response: pd.DataFrame) -> list:
    """Return the list of recording names."""
    rec_list = response.index.get_level_values("Recording").unique().tolist()
    return rec_list


def _list_to_index_map(l) -> Dict[str, int]:
    """
    Returns a map from list elements to their index in the list.
    """
    return {x: i for i, x in enumerate(l)}


def _cluster_ids(response: pd.DataFrame, rec_name: str) -> list:
    """Returns the list of cluster ids for a given recording.

    Stimulus ID = 7 is currently ignored (actually, anything other than 1).

    This function isn't used anymore, I don't think. Keeping it around for
    reference, for a little while.
    """
    # Note: I'm not sure if it's better to request by recording name or
    # the recording id.
    stimulus_id = 1  # not 7.
    cluster_ids = (
        response.xs(
            (stimulus_id, rec_name),
            level=("Stimulus ID", "Recording"),
            drop_level=True,
        )
        .index.get_level_values("Cell index")
        .unique()
        .tolist()
    )
    return cluster_ids


def spike_window(
    spike_idxs: Union[int, Sequence[int]], total_len: int, post_spike_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the window endpoints around a spike, in samples of the stimulus.
    """
    if total_len < post_spike_len + 1:
        raise ValueError(
            f"Total snippet length must be at least 1 greater than the "
            f"post-spike length + 1. Got total_len ({total_len}) "
            f"and post_spike_len ({post_spike_len})."
        )
    spikes = np.array(spike_idxs)
    # Calculate the snippet start and end.
    # The -1 appears as we include the spike sample in the snippet.
    win_start = (spikes + post_spike_len) - (total_len - 1)
    win_end = spikes + post_spike_len + 1
    return win_start, win_end


def spike_snippets(
    stimulus: np.ndarray,
    spike_idxs: np.ndarray,
    total_len: int,
    post_spike_len: int,
) -> np.ndarray:
    """
    Return a subset of the stimulus around the spike points.

    Args:
        stimulus: The decompressed stimulus.
        spikes: Spike counts, re-binned to the stimulus sample rate.
        total_len: The length of the snippet in stimulus frames.
        post_spike_len: The number of frames to pad after the spike.
    Returns:
        A Numpy array of shape (spikes.shape[0], total_len, NUM_STIMULUS_LEDS).

    Note 1: The total length describes the snippet length inclusive of the post
        spike padding.
    Note 2: If a spike bin has 2 spikes, then the snippet they share will be
        added to the output twice.
    Note 3: If a spike happens early enough that the snippet would start before
        the stimulus, then the snippet will be padded with zeros. This applies
        to the end of the stimulus as well.

    Single spike example
    ====================

        frame #:  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
        ===========================================================================
        stimulus: |   0   |   1   |   1   |   0   |   0   |   1   |   0   |   0   |
                  |   1   |   1   |   0   |   0   |   1   |   0   |   0   |   0   |
                  |   1   |   1   |   1   |   0   |   0   |   0   |   0   |   1   |
                  |   0   |   0   |   0   |   0   |   1   |   0   |   1   |   1   |
        ===========================================================================
        spikes:   |   0   |   0   |   0   |   0   |   1   |   0   |   0   |   0   |

    The slice with parameters:
        - total length = 5
        - post spike length = 1

    Would be:

                  |   1   |   1   |   0   |   0   |   1   |
                  |   1   |   0   |   0   |   1   |   0   |
                  |   1   |   1   |   0   |   0   |   0   |
                  |   0   |   0   |   0   |   1   |   0   |
    """
    if np.ma.isMaskedArray(spike_idxs):
        raise ValueError(
            "spikes must be a standard numpy array, not a masked array."
        )
    # 1. Get the spike windows.
    win_start, _ = spike_window(spike_idxs, total_len, post_spike_len)
    # 2. Pad the stimulus in case windows go out of range.
    if np.any(win_start < 0) or np.any(
        win_start >= (stimulus.shape[0] - total_len)
    ):
        stimulus = np.pad(
            stimulus, ((total_len, total_len), (0, 0)), "constant"
        )
        # 3. Offset the windows, which is needed due to the padding.
        win_start += total_len
    # 4. Extract the slice.
    # The padded_stimulus is indexed by a list arrays of the form:
    #    (win_start[0], win_start[0]+1, win_start[0]+2, ..., win_start[0]+total_len)
    #    (win_start[1], win_start[1]+1, win_start[1]+2, ..., win_start[1]+total_len)
    #    ...
    #    (win_start[num_spikes-1], win_start[num_spikes-1]+1, win_start[num_spikes-1]+2, ..., win_start[num_spikes-1]+total_len)
    snippets = stimulus[np.asarray(win_start)[:, None] + np.arange(total_len)]
    return snippets


def compress_spikes(spikes: np.ndarray) -> np.ndarray:
    """Converts a 1D spike array to an index of spike array.

    Example:
        [0, 0, 1, 0, 0, 2, 3] => [2, 5, 5, 6, 6, 6]
    """
    nonzero_idxs = np.squeeze(np.nonzero(spikes))
    # Alternative:
    #   return np.repeat(np.arange(len(spikes)), spikes)
    return np.repeat(nonzero_idxs, spikes[nonzero_idxs])


def labeled_spike_snippets(
    rec: CompressedSpikeRecording,
    snippet_len: int,
    snippet_pad: int,
    downsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Caluclate spike snippets for a recording, paired with the cluster ids.

    Args:
        snippet_len: the length of the snippet.
        snippet_pad: the number of timesteps to include after the spike.

    Returns: two np.ndarrays tuple. The first element contains
    the spike snippets and the second element contains ids of the clusters.
    """
    stimulus = decompress_stimulus(
        rec.stimulus_pattern,
        rec.stimulus_events,
        rec.num_sensor_samples,
        downsample,
    )

    # Gather the snippets and align them with a matching array of cluster ids.
    snippets = []
    cluster_ids = []
    for idx, cluster_id in enumerate(rec.cluster_ids):
        spike_idxs = rebin_spikes(rec.spike_events[idx], downsample)
        snips = spike_snippets(
            stimulus,
            spike_idxs,
            snippet_len,
            snippet_pad,
        )
        cluster_ids.extend(
            [
                cluster_id,
            ]
            * len(snips)
        )
        snippets.extend(snips)
    snippets = np.stack(snippets)
    cluster_ids = np.array(cluster_ids)
    return snippets, cluster_ids
