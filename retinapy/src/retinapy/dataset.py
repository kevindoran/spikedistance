"""
A lot of experimentation ends up here in dataset.py. Not much will generalize 
to other projects.
"""

import bisect
import math
from typing import Dict, Iterable, List, Optional, Tuple, TypeAlias, Union
import numpy as np
import torch
import retinapy.mea as mea
import retinapy.spikedistance as sdf


def _num_snippets(num_timesteps: int, snippet_len: int, stride: int) -> int:
    """
    Returns the number of snippets that can be extracted from a recording with
    the given number of timesteps, snippet length, and stride.
    """
    # The most likely place for insidious bugs to hide is here. So try two
    # ways.
    # 1. Find the last strided timestep, then divide by stride.
    last_nonstrided = num_timesteps - snippet_len
    last_strided_timestep = last_nonstrided - (last_nonstrided % stride)
    num_snippets1 = last_strided_timestep // stride + 1
    # 2. Same deal, but mindlessly copying the formula from Pytorch docs.
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # Padding is 0, and dilation is 1.
    num_snippets2 = math.floor(
        (num_timesteps - (snippet_len - 1) - 1) / stride + 1
    )
    assert num_snippets1 == num_snippets2, (
        f"Strided snippet calculation is wrong. {num_snippets1} != "
        f"{num_snippets2}."
    )
    return num_snippets1


class SingleClusterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        recording: mea.SpikeRecording,
        cluster_idx: int,
        snippet_len: int,
        stride: int = 1,
    ):
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.recording = recording
        self.cluster_idx = cluster_idx
        self.cluster_gid = recording.cidx_to_gid(cluster_idx)
        self.snippet_len = snippet_len
        self.num_clusters = len(recording.cluster_ids)
        self.num_timesteps = len(recording) - snippet_len + 1
        assert (
            self.num_timesteps > 0
        ), "Snippet length is longer than the recording."
        self.stride = stride

    def __len__(self):
        """
        Calculates the number of samples in the dataset.
        """
        return self._num_strided_timesteps

    def __getitem__(self, idx):
        """
        Returns the snippet at the given index.
        """
        start_time_idx = self.stride * idx
        end_time_idx = start_time_idx + self.snippet_len
        assert end_time_idx <= len(
            self.recording
        ), f"{end_time_idx} <= {len(self.recording)}"
        rec = self.recording.stimulus[start_time_idx:end_time_idx].T
        spikes = self.recording.spikes[
            start_time_idx:end_time_idx, self.cluster_idx
        ]
        res = {
            "stimulus": rec,
            "spikes": spikes,
            "cluster_id": self.cluster_gid,
        }
        return res

    @property
    def sample_rate(self):
        return self.recording.sample_rate

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride: int):
        self._stride = stride
        self._num_strided_timesteps = _num_snippets(
            len(self.recording), self.snippet_len, stride
        )

    @property
    def num_strided_timesteps(self):
        return self._num_strided_timesteps


class SnippetDataset(torch.utils.data.Dataset):
    _num_strided_timesteps: int
    _stride: int
    num_clusters: int

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len: int,
        stride: int = 1,
        shuffle_stride: bool = False,
    ):
        """
        Args:
            recording: the recording to extract snippets from.
            snippet_len: the number of bins to include in each snippet.
            stride: viewing the output as a window, stride is the number of
                bins to move the window forward by between samples.
            shuffle_stride: when False, start positions are skipped by the
                nature of stride. If True, start positions are uniformly sampled
                from the range [pos, pos + stride). This is useful as a means
                of tuning the batch size. When stride is 1, a single batch
                can be considered to contain a lot of very similar data. This
                can lead to models becoming overfitted before a single batch
                ends. Setting stride to a larger value can help with this.
                When setting stride > 1, we can jitter the start position as
                so as to not throw away data.
        """
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.recording = recording
        self.snippet_len = snippet_len
        self.num_clusters = len(recording.cluster_ids)
        self.num_timesteps = len(recording) - snippet_len + 1
        self.shuffle_stride = shuffle_stride
        assert (
            self.num_timesteps > 0
        ), "Snippet length is longer than the recording."
        # _num_strided_timesteps is set in the setter for stride.
        self.stride = stride

    def __len__(self):
        """
        Calculates the number of samples in the dataset.
        """
        res = self._num_strided_timesteps * self.num_clusters
        return res

    def _decode_index(self, index: int) -> Tuple[int, int, int]:
        """
        Decodes the index into the timestep and cluster id.

        The data is effectively a 2D array with dimensions (time, cluster).
        The index is the flattened index of this array, and so the timestep
        increases as the index increases and wraps to the next cluster id when
        it reaches the end of the recording.
        """
        timestep_idx = self.stride * (index % self._num_strided_timesteps)
        # Shuffle?
        if self.shuffle_stride and self.stride > 1:
            timestep_idx += torch.randint(
                0, self.stride, (1,), dtype=torch.long
            ).item()
            timestep_idx = min(
                timestep_idx, len(self.recording) - self.snippet_len - 1
            )
        cluster_idx = index // self._num_strided_timesteps
        assert (
            cluster_idx < self.num_clusters
        ), f"{cluster_idx} > {self.num_clusters}"
        cluster_id = self.recording.cluster_gids[cluster_idx]
        return timestep_idx, cluster_idx, cluster_id

    def __getitem__(self, idx):
        """
        Returns the snippet at the given index.
        """
        if idx >= len(self):
            raise IndexError()
        start_time_idx, cluster_idx, cluster_id = self._decode_index(idx)
        end_time_idx = start_time_idx + self.snippet_len
        assert end_time_idx <= len(
            self.recording
        ), f"{end_time_idx} <= {len(self.recording)}"
        rec = self.recording.stimulus[start_time_idx:end_time_idx].T
        spikes = self.recording.spikes[start_time_idx:end_time_idx, cluster_idx]
        res = {"stimulus": rec, "spikes": spikes, "cluster_id": cluster_id}
        return res

    @property
    def sample_rate(self):
        return self.recording.sample_rate

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride: int):
        self._stride = stride
        self._num_strided_timesteps = _num_snippets(
            len(self.recording), self.snippet_len, stride
        )

    @property
    def num_strided_timesteps(self):
        return self._num_strided_timesteps


class BulkSnippetDataset(torch.utils.data.Dataset):
    """
    Snippet dataset where all clusters are presented together.

    Samples are a dictionary of the form
    {
        stimulus: (5, snippet_len),
        spikes: (num_clusters,))
    }
    The cluster ids are not included, as they are always the same. They can
    be obtained from the recording.
    """

    _num_strided_timesteps: int
    _stride: int
    num_clusters: int

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len: int,
        stride: int = 1,
        shuffle_stride: bool = False,
    ):
        """
        Args:
            recording: the recording to extract snippets from.
            snippet_len: the number of bins to include in each snippet.
            stride: viewing the output as a window, stride is the number of
                bins to move the window forward by between samples.
            shuffle_stride: when False, start positions are skipped by the
                nature of stride. If True, start positions are uniformly sampled
                from the range [pos, pos + stride). This is useful as a means
                of tuning the batch size. When stride is 1, a single batch
                can be considered to contain a lot of very similar data. This
                can lead to models becoming overfitted before a single batch
                ends. Setting stride to a larger value can help with this.
                When setting stride > 1, we can jitter the start position as
                so as to not throw away data.
        """
        if snippet_len > len(recording):
            raise ValueError(
                f"Snippet length ({snippet_len}) is larger than "
                f"the recording length ({len(recording)})."
            )
        self.recording = recording
        self.snippet_len = snippet_len
        self.num_clusters = len(recording.cluster_ids)
        self.num_timesteps = len(recording) - snippet_len + 1
        self.shuffle_stride = shuffle_stride
        assert (
            self.num_timesteps > 0
        ), "Snippet length is longer than the recording."
        # _num_strided_timesteps is set in the setter for stride.
        self.stride = stride

    def __len__(self):
        """
        Calculates the number of samples in the dataset.
        """
        res = self._num_strided_timesteps
        return res

    def _decode_index(self, index: int) -> int:
        """
        Decodes the index into the starting timestep.
        """
        timestep_idx = self.stride * index
        if self.shuffle_stride and self.stride > 1:
            timestep_idx += torch.randint(
                0, self.stride, (1,), dtype=torch.long
            ).item()
            timestep_idx = min(
                timestep_idx, len(self.recording) - self.snippet_len - 1
            )
        return timestep_idx

    def __getitem__(self, idx):
        """
        Returns the snippet at the given index.
        """
        start_time_idx = self._decode_index(idx)
        end_time_idx = start_time_idx + self.snippet_len
        assert end_time_idx <= len(
            self.recording
        ), f"{end_time_idx} <= {len(self.recording)}"
        rec = self.recording.stimulus[start_time_idx:end_time_idx].T
        spikes = self.recording.spikes[start_time_idx:end_time_idx, :]
        res = {"stimulus": rec, "spikes": spikes}
        return res

    @property
    def stride(self):
        return self._stride

    @stride.setter
    def stride(self, stride: int):
        self._stride = stride
        self._num_strided_timesteps = _num_snippets(
            len(self.recording), self.snippet_len, stride
        )

    @property
    def num_strided_timesteps(self):
        return self._num_strided_timesteps


class SpikeCountDataset(torch.utils.data.Dataset):
    """
    Dataset that pairs a stimulus+spike history with a future spike count.

    The spike count is for a configurable duration after the end of the
    history snippet.

    This (X,y) style dataset is intended to for basic comparison between
    different spike count models.
    """

    def __init__(
        self,
        recordings: List[mea.SpikeRecording],
        input_len: int,
        output_len: int,
        stride: int = 1,
    ):
        # Check that recording chunks sufficiently match.
        _sample_rates = [rec.sample_rate for rec in recordings]
        _stim_shape = [rec.stimulus.shape[1] for rec in recordings]
        def _all_same(items):
            return all(i == items[0] for i in items)
        if not _all_same(_sample_rates):
            raise ValueError("All recordings must have the same sample rate."
                             f" Got ({_sample_rates})")
        if not _all_same(_stim_shape):
            raise ValueError("All recordings must have the same stimulus shape."
                             f" Got ({_stim_shape})")

        self._sample_rate = _sample_rates[0]
        self._recordings = recordings
        self.output_len = output_len
        self.input_len = input_len
        self.total_len = input_len + output_len
        self.ds = ConcatDataset(
            [
                SnippetDataset(rec, self.total_len, stride)
                for rec in self._recordings
            ]
        )

    def __len__(self):
        """
        Calculates the number of samples in the dataset.

        There will be one sample for every timestep in the recording.
        """
        return len(self.ds)

    @property
    def recordings(self):
        return self._recordings

    @property
    def datasets(self):
        return self.ds.datasets

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def stride(self):
        return self.ds.stride

    @stride.setter
    def stride(self, stride: int):
        self.ds.stride = stride

    def __getitem__(self, idx):
        """
        Returns the (X,y) sample at the given index.

        Index is one-to-one with the timesteps in the recording.
        """
        sample = self.ds[idx]
        rec = sample["stimulus"]
        spikes = sample["spikes"]
        X_stim = rec[:, 0 : self.input_len]
        X_spikes = spikes[0 : self.input_len]
        X = np.vstack((X_stim, X_spikes))
        Y = spikes[self.input_len :]
        assert Y.shape == (self.output_len,)
        y = np.sum(Y)
        return X, y


class BulkSpikeCountDataset(torch.utils.data.Dataset):
    """
    SpikeCountDataset but with all clusters in a single sample.
    """

    def __init__(
        self,
        recording: mea.SpikeRecording,
        input_len: int,
        output_len: int,
        stride: int = 1,
    ):
        self.output_len = output_len
        self.input_len = input_len
        self.total_len = input_len + output_len
        self.recording = recording
        self.ds = BulkSnippetDataset(recording, self.total_len, stride)

    @staticmethod
    def from_standard(
        ds: SpikeCountDataset,
    ) -> "BulkSpikeCountDataset":
        return BulkSpikeCountDataset(
            ds.recording, ds.input_len, ds.output_len, ds.stride
        )

    def __len__(self):
        """
        Calculates the number of samples in the dataset.

        There will be one sample for every timestep in the recording.
        """
        return len(self.ds)

    def __getitem__(self, idx):
        """
        Returns the (X,y) sample at the given index.

        X's shape (4, snippet_len); it doesn't have any spikes. y's shape
        is (num_clusters,).

        """
        sample = self.ds[idx]
        rec = sample["stimulus"]
        spikes = sample["spikes"]
        X = rec[:, 0 : self.input_len]
        Y = spikes[:, self.input_len :]
        y = np.sum(Y)
        return X, y


class BasicDistDataset(torch.utils.data.Dataset):
    """
    Dataset that pairs a stimulus+spike history with a future distance array.

    Future stimulus information is not made available.

    Output is a dictionary:

        "snippet": (5, input_len), float
            - Returned as is from the underlying SnippetDataset.
               - the first 4 channels are the stimulus, containing float values
                 in the range [0, 1].
               - the 5th channel contains spike history, a binary array of
                 {0, 1}.
        "dist": (output_len,), float
            - A distance array of length output_len, starting from
              time_bin=0 - dist_prefix_len. The values are *not* the log
              values. The values are clipped to dist_clamp.
        "target_spikes": (output_len,), float
            - Same as the 5th channel of the "snippet" array, but for the
              output_len bins starting from t=0.
        "cluster_id": int
            - The cluster ID of the cell that the snippet is for.

    No normalization
    ----------------
    No normalizing is done by the dataset; do it yourself in a trainable or
    model. Why?

        - changing norm will affect all previously trained models.
        - normalized spikes would no longer be binary {0, 1}, which is
            unintuitive for subsequent analysis.
        - stimulus norm is not the same for all recordings.
    """

    def __init__(
        self,
        recordings: List[mea.SpikeRecording],
        input_len: int,
        output_len: int,
        pad: int,
        dist_prefix_len: int,
        dist_clamp: float,
        stride: int = 1,
        shuffle_stride: bool = False,
        use_augmentation: bool = False,
    ):
        if output_len < dist_prefix_len:
            raise ValueError(
                "The output must be longer than it's offset relative to t=0."
                f"Got output_len ({output_len}) < dist_prefix_len "
                f"({dist_prefix_len})."
            )
        self.input_len = input_len
        self.output_len = output_len
        self._recordings = recordings
        for r in recordings:
            assert r.stimulus.shape[1] == recordings[0].stimulus.shape[1]
            assert r.sample_rate == recordings[0].sample_rate
        self._sample_rate = self._recordings[0].sample_rate
        self.num_stim_channels = self._recordings[0].stimulus.shape[1]
        self.pad = pad
        self.dist_prefix_len = dist_prefix_len
        self.dist_clamp = dist_clamp
        self.use_augmentation = use_augmentation
        # self.ds = SnippetDataset(
        #     recordings[0],
        #     self.input_len + self.output_len - self.dist_prefix_len + self.pad,
        #     stride,
        #     shuffle_stride=shuffle_stride,
        # )
        self.ds = ConcatDataset(
            [
                SnippetDataset(
                    rec,
                    self.input_len
                    + self.output_len
                    - self.dist_prefix_len
                    + self.pad,
                    stride,
                    shuffle_stride=shuffle_stride,
                )
                for rec in self._recordings
            ]
        )

    def __len__(self):
        return len(self.ds)

    @property
    def recordings(self):
        return self._recordings

    @property
    def datasets(self):
        return self.ds.datasets

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def stride(self):
        return self.ds.stride

    @stride.setter
    def stride(self, stride: int):
        self.ds.stride = stride

    def output_spikes(self, cluster_id=0) -> np.ndarray:
        """
        Returns the underlying 1D spike array that is equivalent to
        concatenating the spike portion of the dataset's outputs in ascending
        order. The equivalent concatenation must truncate the output to
        the stride length.

        This is useful for obtaining the full length ground truth spike array
        without having to iterate over the dataset. It is also useful to have
        this as a function so that it can be tested for any indexing issues—such
        issues could render evaluation metrics incorrect.

        If there are multiple recording chunks, then they will be concatenated
        together.
        """
        res = []
        start = self.input_len
        future_out_len = self.output_len - self.dist_prefix_len
        if self.stride > future_out_len:
            raise ValueError(
                "Stride must be less-equal the length of the future portion of "
                "the output in order to recover the unbroken spike sequence "
                f"via susseccive concatenations. Got stride ({self.stride}) vs."
                f" ({future_out_len})."
            )
        for ds in self.datasets:
            end = start + self.stride * len(ds)
            res.append(ds.recording.spikes[start:end, cluster_id])
        res = np.concatenate(res)
        return res

    def _augment_stimulus(self, stimulus):
        """
        Augment a stimulus portion of a sample.
        """
        NOISE_SD = 0.2
        NOISE_MU = 0
        STIM_MASK_RATE = 0.01
        MIXUP_P = 0.2
        MASK3_VALUE = -3
        # Mixup
        mix_idx = np.random.randint(0, len(self))
        mixup = self.ds[mix_idx]["stimulus"][:, 0 : stimulus.shape[1]]
        stimulus = stimulus * (1 - MIXUP_P) + mixup * MIXUP_P
        # Whole block scale.
        mu = 1.0
        sd = 0.10
        scale = np.random.normal(mu, sd, size=(1,))
        # Whole block offset.
        mu = 0.0
        sigma = 0.10
        offset_noise = np.random.normal(mu, sigma, size=(1,))
        # Per bin noise.
        max_length = stimulus.shape[1]
        left, right = (0, max_length - 1)
        bin_noise = np.random.normal(
            NOISE_MU,
            NOISE_SD,
            size=(self.num_stim_channels, (right - left)),
        )
        stimulus = stimulus * scale + offset_noise
        stimulus[:, left:right] += bin_noise
        # Mask some parts.
        mask_indicies = np.nonzero(
            np.random.binomial(1, p=STIM_MASK_RATE, size=len(stimulus))
        )
        stimulus[:, mask_indicies] = MASK3_VALUE
        return stimulus

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        +---------------------------+
        |  a) input stimulus        |
        +---------------------------+
        |  b) input spike           |
        +---------------------------+
                                        this gap is the output offset,
                              |<--->|   typically negative.

                              |-----+---------------+---------+
                              |   c) dist target*   | d) pad* |
                              |-----+---------------+---------+
                                    | e) out spikes |
                                    +---------------+


        Note (c*): the target distance array is for the time interval
        [0-dist_prefix_len, 0-dist_prefix_len+output_len). For example,
        if the prefix length is 32 and the output length is 128, this would be
        [-32, 96). While the interval extends into the past by the prefix
        length, only future spikes are considered to calculate the distance
        array. The alternative of including spikes that land in [-prefix_len, 0)
        would be fine too, and this approach has been tried, but the inference
        is harder. It's harder because spikes within [-prefix_len, 0) will fix
        an upper limit on the target distance array, and the model will always
        predict lower than this, making inference susceptible to incorrectly
        place spikes close to zero.

        Note (d*): there is an extra bit of spike data used when creating
        a sample, here called a pad. The pad is used to calculate the ground
        truth distance array. This bit of data is not placed in the sample that
        is returned.

        """
        sample = self.ds[idx]
        stimulus = sample["stimulus"][:, 0 : self.input_len]
        # Switch to float for spikes.
        spikes = np.array(sample["spikes"], copy=True, dtype=float)
        dist_input_spikes = np.concatenate(
            [
                np.zeros(self.dist_prefix_len, dtype=float),
                spikes[self.input_len :],
            ]
        )
        dist = sdf.distance_arr(dist_input_spikes, self.dist_clamp)[
            0 : self.output_len
        ]
        in_spikes = spikes[0 : self.input_len]
        # Target spikes only include the future spikes (t>0).
        target_spikes = spikes[
            self.input_len : self.input_len + self.output_len
        ]
        if self.use_augmentation:
            stimulus = self._augment_stimulus(stimulus)
        # Returning a dictionary is more flexible than returning a tuple, as
        # we can add to the dictionary without breaking existing consumers of
        # the dataset.
        res = {
            "snippet": np.vstack((stimulus, in_spikes)),
            "dist": dist,
            "target_spikes": target_spikes,
            "cluster_id": sample["cluster_id"],
        }
        return res


class DistDataset(torch.utils.data.Dataset):
    """
    Dataset producing a snippet (spikes and stimulus) and a spike distance
    array. A portion of the spikes in the snippet are masked. The spike distance
    array will be created for the masked portion of the snippet.

    The intended usecase of this dataset is to predict spike activity given
    stimulus and spike history.
    """

    # Mask value should be negative. Zero represents no spikes, and 1+ represent
    # a spike count which can be greater than 1!
    MASK_VALUE = -1
    MASK2_VALUE = -0.5
    MASK3_VALUE = 10
    # Do not! set a seed within the dataset. Process forking leads to identical
    # seeds.
    # RNG_SEED = 123

    # TODO: make configurable
    NOISE_SD = 0.7
    NOISE_MU = 0
    NOISE_JITTER = 2
    DROP_RATE = 0.0
    STIM_MASK_RATE = 0.2
    SPIKE_MASK_RATE = 0.0
    MIXUP_P = 0.2

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len,
        mask_begin: int,
        mask_end: int,
        pad: int,
        dist_clamp: float,
        stride: int = 1,
        enable_augmentation: bool = True,
        allow_cheating: bool = False,
    ):
        self.enable_augmentation = enable_augmentation
        self.num_stim_channels = recording.stimulus.shape[1]
        self.pad = pad
        self.dist_clamp = dist_clamp
        self.ds = SnippetDataset(
            recording,
            snippet_len + self.pad,
            stride,
            shuffle_stride=False,
        )
        self.mask_slice = slice(mask_begin, mask_end)
        self.allow_cheating = allow_cheating

        # Don't do this! Mean will depend on train,val,test!
        # self.stim_mean = np.expand_dims(recording.stimulus.mean(axis=0), -1)
        # self.stim_sd = np.expand_dims(recording.stimulus.std(axis=0), -1)

        # The stimulus mean will be dominated by the mask
        mask_len = mask_end - mask_begin
        self.spike_mean = mask_len * self.MASK_VALUE / snippet_len
        self.spike_sd = (
            mask_len * (self.MASK_VALUE - self.spike_mean) ** 2
            + (snippet_len - mask_len) * self.spike_mean**2
        ) / snippet_len

    def __len__(self):
        return len(self.ds)

    @property
    def recording(self):
        return self.ds.recording

    @property
    def sample_rate(self):
        return self.recording.sample_rate

    @property
    def stride(self):
        return self.ds.stride

    # Setter property for stride
    @stride.setter
    def stride(self, stride: int):
        self.ds.stride = stride

    @classmethod
    def mask_start(cls, spikes):
        mask_val = cls.MASK_VALUE
        mask_start_idx = np.min(np.flatnonzero(spikes == mask_val))
        return mask_start_idx

    def _augment_stimulus(self, stimulus):
        """
        Augment a stimulus portion of a sample.
        """
        # Mixup
        mix_idx = np.random.randint(0, len(self))
        mixup = self.ds[mix_idx]["stimulus"][:, 0 : stimulus.shape[1]]
        stimulus = stimulus * (1 - self.MIXUP_P) + mixup * self.MIXUP_P
        # Whole block scale.
        mu = 1.0
        sd = 0.10
        scale = np.random.normal(mu, sd, size=(1,))
        # Whole block offset.
        mu = 0.0
        sigma = 0.10
        offset_noise = np.random.normal(mu, sigma, size=(1,))
        # Per bin noise.
        max_length = stimulus.shape[1]
        center, length = np.random.randint(low=0, high=max_length, size=(2,))
        # left = max(0, center - length // 2)
        # right = min(max_length - 1, center + length // 2 + 1)
        left, right = (0, max_length - 1)
        bin_noise = np.random.normal(
            self.NOISE_MU,
            self.NOISE_SD,
            size=(self.num_stim_channels, (right - left)),
        )
        stimulus = stimulus * scale + offset_noise
        stimulus[:, left:right] += bin_noise
        # Mask some parts.
        mask_indicies = np.nonzero(
            np.random.binomial(1, p=self.STIM_MASK_RATE, size=len(stimulus))
        )
        stimulus[:, mask_indicies] = self.MASK3_VALUE
        return stimulus

    def _augment_spikes(self, spikes):
        """
        Augment the spike portion of a sample.

        Call this on the model input portion of the spike data, and not the
        portion that we are trying to predict.
        """
        spike_indicies = np.nonzero(spikes)
        spikes[spike_indicies] = 0
        # Add jitter
        if self.NOISE_JITTER > 0:
            jitter = np.random.randint(
                -self.NOISE_JITTER, self.NOISE_JITTER, len(spike_indicies)
            )
            spike_indicies = np.clip(
                spike_indicies + jitter, 0, len(spikes) - 1
            )
            # Drop some spikes.
            new_spikes = np.random.binomial(
                1, p=(1 - self.DROP_RATE), size=len(spike_indicies)
            )
            spikes[spike_indicies] = new_spikes
        # Mask some parts.
        mask_indicies = np.nonzero(
            np.random.binomial(1, p=self.SPIKE_MASK_RATE, size=len(spikes))
        )
        spikes[mask_indicies] = self.MASK2_VALUE
        return spikes

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        +---------------------------------+-------------+
        |  a) input stimulus                            |
        +---------------------------------+-------------+---------+
        |  b) input spike data            | c) masked   | d) pad* |
        +---------------------------------+-------------+---------+

        Note (d*): there is an extra bit of spike data used when creating
        a sample, here called a pad. The pad is used to calculate the ground
        truth distance array. This bit of data is not placed in the sample that
        is returned.
        """
        # 1. Get the snippet. Make it extra long, for the distance array calc.
        sample = self.ds[idx]
        extra_long_stimulus = sample["stimulus"]
        extra_long_spikes = sample["spikes"]
        cluster_id = sample["cluster_id"]
        # For some unknown reason, the following copy call makes
        # training about 5x faster, and it has no effect when called on the
        # stimulus array. Maybe related to the copy that is done below for
        # target_spikes?
        extra_long_spikes = np.array(extra_long_spikes, copy=True, dtype=float)

        # 2. Optional augmentation.
        if self.enable_augmentation:
            extra_long_stimulus = self._augment_stimulus(extra_long_stimulus)
            extra_long_spikes[0 : self.mask_slice.start] = self._augment_spikes(
                extra_long_spikes[0 : self.mask_slice.start]
            )
        # 3. Calculate the distance array.
        dist = sdf.distance_arr(extra_long_spikes, self.dist_clamp)
        # With the distance array calculated, we can throw away the extra bit.
        dist = dist[self.mask_slice]
        target_spikes = np.array(extra_long_spikes[self.mask_slice], copy=True)
        if not self.allow_cheating:
            extra_long_spikes[self.mask_slice] = self.MASK_VALUE
        # 4. Remove the extra padding that was used to calculate the distance arrays.
        stimulus = extra_long_stimulus[:, 0 : -self.pad]
        spikes = extra_long_spikes[0 : -self.pad]
        # 5. Stack
        snippet = np.vstack((stimulus, spikes))
        # Returning a dictionary is more flexible than returning a tuple, as
        # we can add to the dictionary without breaking existing consumers of
        # the dataset.
        res = {
            "snippet": snippet,
            "dist": dist,
            "target_spikes": target_spikes,
            "cluster_id": cluster_id,
        }
        return res


class SnippetDataset2(torch.utils.data.Dataset):
    """
    Produces stack([snippets, dist_arr]).

    Like SnippetDataset, there is no separation between input and output
    sections of the snippets. This is in contrast to the DistDataset, which has
    input spikes and output distance array with a clear boundary between what
    is input and what is output. Maybe rename the latter to be
    SnippetInDistFieldOut or something?
    """

    def __init__(
        self,
        recording: mea.SpikeRecording,
        snippet_len: int,
        pad_for_dist_calc: int,
        dist_clamp: float,
        stride: int = 1,
        rec_cluster_ids: Optional[mea.RecClusterIds] = None,
    ):
        self.num_stim_channels = recording.stimulus.shape[1]
        self.pad_for_dist_calc = pad_for_dist_calc
        self.dist_clamp = dist_clamp
        self.snippet_len = snippet_len
        self.ds = SnippetDataset(
            recording,
            snippet_len + self.pad_for_dist_calc,
            stride,
            rec_cluster_ids,
        )

    def __len__(self):
        return len(self.ds)

    @property
    def recording(self):
        return self.ds.recording

    @property
    def sample_rate(self):
        return self.recording.sample_rate

    @property
    def stride(self):
        return self.ds.stride

    # Setter property for stride
    @stride.setter
    def stride(self, stride: int):
        self.ds.stride = stride

    def normalize_stimulus(self, stimulus):
        """
        Normalize a stimulus portion of a sample.
        """
        # Hard-code mean of 0.5, justified as it is binary noise with p=0.5.
        stim_mean = 0.5
        stim_sd = 0.5  # 0.25 variance for bernoilli with p=0.5
        res = (stimulus - stim_mean) / stim_sd
        return res

    def __getitem__(self, idx):
        """
        Returns the sample at the given index.

        +----------------------------------------+
        |  a) input stimulus                     |
        +----------------------------------------+---------+
        |  b) dist array                         | d) pad* |
        +----------------------------------------+---------+

        Note (d*): there is an extra bit of spike data used when creating
        a sample, here called a pad. The pad is used to calculate the ground
        truth distance array. This bit of data is not placed in the sample that
        is returned.
        """
        # 1. Get the snippet. Make it extra long, for the distance array calc.
        sample = self.ds[idx]
        extra_long_stimulus = sample["stimulus"]
        extra_long_spikes = sample["spikes"]
        cluster_id = sample["cluster_id"]
        # TODO: test again in this situation for speed-up.
        # extra_long_spikes = np.array(extra_long_spikes, copy=True, dtype=float)

        # 2. Optional augmentation.
        # TODO: augmentation

        # 3. Calculate the distance array.
        dist = sdf.distance_arr(extra_long_spikes, self.dist_clamp)
        # With the distance array calculated, we can throw away the extra bit.
        dist = dist[0 : self.snippet_len]
        # 4. Remove the extra padding that was used to calculate the distance arrays.
        stimulus = extra_long_stimulus[:, 0 : -self.pad_for_dist_calc]
        # 5. Normalize
        stimulus_norm = self.normalize_stimulus(stimulus)
        # 6. Stack
        snippet = np.vstack((stimulus_norm, dist))
        # Returning a dictionary is more flexible than returning a tuple, as
        # we can add to the dictionary without breaking existing consumers of
        # the dataset.
        res = {
            "snippet": snippet,
            "cluster_id": cluster_id,
        }
        return res


class ConcatDataset(torch.utils.data.Dataset):
    """
    Dataset that concatenates datasets and inserts a dataset label.

    This is an edited version of PyTorch's ConcatDataset, with the addition
    of a dataset index being included in each sample. This is useful for
    making a multi-recording dataset easily from a list of single recording
    datasets. The PyTorch implementation wasn't sufficient, as we want to
    include information of which recording a sample belongs to.

    As stride affects dataset sizes, we add the setting and getting of strides
    here.
    """

    datasets: List[torch.utils.data.Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r = []
        s = 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(
        self,
        datasets: Iterable[torch.utils.data.Dataset],
        label_key: Optional[str] = None,
    ) -> None:
        """
        Args:
            label_key: If not None, the dataset index will be included in each
                sample, under this key. If None, the dataset index will not be
                included. Example, label_key = "id". This was originally added
                to allow the recording id to be included in the sample when
                multiple recordings form a concatenated dataset. However,
                to allow for more flexibility in what recordings are loaded in
                train vs. test time, this responsibility has been delegated to
                the underlying per-recording dataset. We do however need to
                enable the recording identifier, as it is disabled by default.
                At the moment, this is expected to be done to each dataset
                before they are passed in here. With label_key = None, this
                class is equivalent to the PyTorch ConcatDataset. Might end
                up removing it if the label key isn't needed anymore.
        """
        super().__init__()
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(
                d, torch.utils.data.IterableDataset
            ), "ConcatDataset does not support IterableDataset"
        self.label_key = label_key
        assert len(self.datasets) > 0, (
            "datasets should not be an empty " "iterable"
        )
        self._update_sizes()

    def _update_sizes(self):
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def stride(self):
        res = self.datasets[0].stride
        assert all([ds.stride == res for ds in self.datasets])
        return res

    @stride.setter
    def stride(self, stride: int):
        for ds in self.datasets:
            ds.stride = stride
        self._update_sizes()

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample = self.datasets[dataset_idx][sample_idx]
        if self.label_key:
            assert isinstance(
                sample, dict
            ), "Sample must be a dictionary in order to add a label."
            sample[self.label_key] = dataset_idx
        return sample


class ConcatDistDataset(ConcatDataset):
    """
    A concatenation of DistDatasets, possibly coming from multiple recordings.

    The DistDatasets might represent multiple contiguous chunks from the same
    recording.

    This class exists to expose some methods like sample_rate() that are both
    present and consistent among all contained datasets. Using these methods
    is tedious if using LabeledConcatDataset directly. It's also brittle, as
    the LabeledConcatDataset can't (but should) act as a drop in replacement
    for a single DistDataset.

    While we are at it, we can encapsulate the setting of the "rec_id" as the
    custom label key.
    """

    def __init__(self, datasets: Iterable[DistDataset]) -> None:
        super().__init__(datasets, label_key=None)
        # Check for consistency
        sample_rate = self.datasets[0].sample_rate
        for ds in self.datasets:
            assert ds.sample_rate == sample_rate, (
                "All datasets must have the same sample rate. Got "
                f"({sample_rate}) and ({ds.sample_rate})."
            )

    @property
    def sample_rate(self):
        return self.datasets[0].sample_rate

    @property
    def stride(self):
        return self.datasets[0].stride

    @stride.setter
    def stride(self, stride):
        for ds in self.datasets:
            ds.stride = stride
        self._calc_size()

    @property
    def num_recordings(self):
        return len(self.datasets)

    @property
    def num_clusters(self):
        cids = set.union(*[set(d.recording.cluster_gids) for d in self.datasets])
        res = len(cids)
        return res
