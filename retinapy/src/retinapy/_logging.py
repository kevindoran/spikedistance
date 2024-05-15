from collections import defaultdict
import collections.abc
import inspect
import io
import json
import logging
import logging.handlers
import math
from numbers import Number
import pathlib
import shutil
import sys
import time
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import PIL
import PIL.Image
import einops
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard as tb
import plotly
import plotly.graph_objects
import retinapy.models

"""
Logging
=======

The logging functionality in this module handles things like:

    - where to log
    - using Python's logging module
    - timing things 
    - measuring things with numbers
        - increasing vs. decreasing
        - accumulated calculation vs. one-off
        - best-so-far tracking *
    - saving Pytorch models *
    - logging training/evaluation data *

The stars are placed next to the categories that are training-loop specific.
If a logging routine needs to know what epoch it is, then it's very much
inseparable from the training-loop. I'm pointing this out, as it's conceivable
that these two categories of logging will be separated into different modules
at some point; the more general logging functions might be used for other
projects.

A lot more can be said about the training-loop logging.

tl;dr 
-----
Trainables return a dictionary like {"metrics": .., "images": ..., "figures": ...,}

Training-loop logging
=====================
Logging for a training-loop is a great source of edge cases, thwarting
attempts to create separation between parts of the training apparatus. 
If you blur your eyes a bit, the concerns of different parts of the training 
aparatus form a top-down gradient. At the top, the mundane but important things
like where to store data is handled. In the middle, we have the training-loop
management which knows how to setup and run the endless loop over the data
to train and periodiacally evaluate a model, yet it doesn't care at all about
what model it is actually training. At the very bottom is the actual Pytorch
model. It has been given quite a simple life thanks to the work of the 
Trainable—the great encapsulator of entropy—that knows how to pull a sample from
the dataset and feed it to the model. If you want to re-use a PyTorch Dataset
for multiple models, you really do need something in the middle than
massages the data into each model. But that is not the only work of the 
Trainable—the Trainable is the only one who has enough context to report on how 
good or bad the training is actually progressing, so it gets the task of
gathering up the data to be logged. And this is where logging comes in to make 
things difficult. The Trainable should be free to log whatever it wants, from
metrics like loss, to images and figures and weight snapshots. But all of
this data cannot be handed back to the training-loop without being specific 
about what each piece of data is, as different types of data need to be 
handled differently. For example, the TensorBoard API requires datatypes to
be separated, as the API calls are different for different types, for example,
SummaryWriter.add_scalar() vs. SummaryWriter.add_histogram(). The W&B API takes
a different approach and accepts data as a dictionary where the values are
object instances of wandb classes, each data type using a different class.

The consequence of Tensorboard's approach is that you need to attach some sort
of metadata to your data that will allow you to create some big if-else block 
that routes data to the suitable Tensorboard function call. Alternatively,
you could call the Tensorboard API directly in the Trainable. This latter 
approach feels like it would be frustrating, as suddenly, your Trainable 
object needs to know about the training-loop (what step etc). What if you
want to run the evaluation outside of a training loop, say at test time? So
many little details to handle. So your choice with tensorboard is to either
log data from very deep in your stack, or pass around some metadata that allows
you to build a big if-else block at some point.

In comparison, if one imagined committing fully to W&B, the Trainable
can create the wandb objects directly and return a data dictionary that doesn't 
need to be inspected at any point before handing off to the wandb API. This
is very seductive. W&B has nicely observed that the decision of what type
of data to be logged is made when you collect the data, and shouldn't need to
be made again when you go to actually call your logging utility. You can
view the set of wandb datatypes here: 

    https://github.com/wandb/wandb/blob/latest/wandb/__init__.py

Another interesting file is the one below, which contains the `WBValue` class
definition. The `WBValue` class is the abstract parent class for anything 
that can be logged with W&B api. It pressages some of the abstractions one 
might end up having to make.

https://github.com/wandb/wandb/blob/d622ee37b232e54addcd48e9f92d9198a3e2790b/wandb/sdk/data_types/base_types/wb_value.py#L56

Another class, `Media` shows how the `WBValue` parent class separates as a tree
into different types of data formats.

https://github.com/wandb/wandb/blob/d622ee37b232e54addcd48e9f92d9198a3e2790b/wandb/sdk/data_types/base_types/media.py#L31

What to actually do
-------------------
I don't want to stop using Tensorboard, I don't want to call tensorboard 
logging calls directly in the Trainable, and I want the option of using 
alternative logging tools like W&B in the future. So it seems like there
will be a if-else block for data routing. What is left is to choose what is the
form of metadata this is switched upon. We could go with classes, like how
wandb seems to do it, or we could go with just strings like "metrics", "images", 
etc. I think this latter approach is nicer; I don't want to have to create 
new classes just for the sake of being distinguished in some if-else chain.
Having said that, classes will probably wrap most of the data types anyway,
for example, numeric data is already wrapped in the Metric class which 
records the increasing or decreasing nature of a metric.

The Tensorboard and wandb API documentation homepages:

    https://pytorch.org/docs/stable/tensorboard.html
    https://docs.wandb.ai/
"""


_logger = logging.getLogger(__name__)


def setup_logging(level):
    """Default logging setup.

    The default setup:
        - makes the root logger use the specified level
        - adds stdout as a handler
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    root_logger.addHandler(console_handler)


def enable_file_logging(log_path: Union[pathlib.Path, str]):
    """Enable logging to a file.

    Rolling logging is used—additional files have ".1", ".2", etc. appended.
    """
    root_logger = logging.getLogger()
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=(2 ** (10 * 2) * 5), backupCount=3
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d: [%(levelname)8s] - %(message)s"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def get_outdir(
    base_dir: Union[pathlib.Path, str], labels: Optional[Sequence[str]] = None
):
    """Returns the directory in which all logging should be stored.

    The directory will have the form:

        /basedir/label1/label2/label3/<num>

    Where <num> is incremented so that a fresh directory is always created.
    """
    base_dir = pathlib.Path(base_dir)
    if labels is None:
        labels = []
    base_dir = base_dir.joinpath(*labels)
    if not base_dir.exists():
        base_dir.mkdir(parents=True)
    count = 0
    folder_path_f = lambda count: base_dir / str(count)
    while folder_path_f(count).exists():
        count += 1
    is_probably_too_many = count > 1000
    if is_probably_too_many:
        _logger.warn(f"Reached {count} output sub-folders.")
    out_dir = folder_path_f(count)
    out_dir.mkdir()
    return out_dir


def snapshot_module(out_dir):
    """Snapshot the retinapy module source code."""
    import retinapy

    module_dir = pathlib.Path(inspect.getfile(retinapy)).parent
    shutil.copytree(
        module_dir,
        out_dir / "src_snapshot" / "retinapy",
        ignore=shutil.ignore_patterns("*pyc"),
    )


class Meter:
    """An online sum and avarage meter."""

    def __init__(self, name=None):
        self.reset()
        self.name = name

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def __str__(self):
        res = f"{self.name} " if self.name else "Meter"
        res += f"(average -- total) : {self.avg:.4f} -- ({self.sum:.4f})"
        return res


class MovingAverageMeter:
    def __init__(self, beta: float, name=None):
        """
        beta: the weighting given to the new value.
        """
        self.reset()
        self.beta = beta
        self.name = name
        self._avg = None

    def reset(self):
        self._avg = None
        self.count = 0
        self.sum = 0

    @property
    def avg(self) -> float:
        return self._avg if self._avg is not None else 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        if self._avg is None:
            self._avg = val
        else:
            b = self.beta**n
            self._avg = (1-b) * self._avg + b * val

    def __str__(self):
        res = f"{self.name} " if self.name else "Meter"
        res += f"(smoothed -- total) : {self.avg:.4f} -- ({self.sum:.4f})"
        return res


class Timer:
    """A little timer to track repetitive things.

    Rates/durations are monitored with exponential moving averages.

    All times are in seconds (fractional).

    There is two main use cases:

    The timer supports the context manager protocol, so you can use it like:

        timer = Timer()
        with timer:
            # Do something
        print(f"Duration: {timer.elapsed()}")

    The original motivation for this feature was to track how long it took to
    run validation, while keeping a rolling average of the validation time.
    """

    def __init__(self):
        self.w = 0.1
        self._loop_start = None
        self._rolling_duration = None
        self._first_start = None
        self._prev_elapsed = 0

    @staticmethod
    def create_and_start() -> "Timer":
        timer = Timer()
        timer.restart()
        return timer

    def restart(self) -> Optional[float]:
        """Start or restart the timer.

        Returns the duration of the previous loop, or None on the first call.
        """
        t_now = time.monotonic()
        # First call?
        if self._first_start is None:
            self._first_start = t_now
        # Has the timer been stopped? If so, start it again.
        if self._loop_start is None:
            self._loop_start = t_now
            return
        assert self._loop_start is not None
        # From second call, we have an elasped time.
        self._prev_elapsed = t_now - self._loop_start
        self._increment_rolling(self._prev_elapsed)
        self._loop_start = t_now
        return self._prev_elapsed

    def _increment_rolling(self, elapsed):
        if self._rolling_duration is None:
            self._rolling_duration = elapsed
        else:
            assert self._rolling_duration is not None
            self._rolling_duration = (
                self.w * elapsed + (1 - self.w) * self._rolling_duration
            )

    def stop(self) -> Optional[float]:
        """Stop the timer.

        It's fine to stop a timer that hasn't been started.
        """
        if self._loop_start is None:
            return
        self._prev_elapsed = time.monotonic() - self._loop_start
        self._increment_rolling(self._prev_elapsed)
        self._loop_start = None
        return self._prev_elapsed

    def elapsed(self) -> float:
        """
        The current elapsed time, or previous if the timer is stopped.

        If the timer has never been started, an error will be raised.
        """
        if self._loop_start is None:
            if self._prev_elapsed is None:
                raise ValueError("Not yet started.")
            return self._prev_elapsed
        return time.monotonic() - self._loop_start

    def total_elapsed(self) -> float:
        if self._first_start is None:
            raise ValueError("Not yet started")
        return time.monotonic() - self._first_start

    def rolling_duration(self) -> float:
        if self._rolling_duration is None:
            return self.elapsed()
        else:
            return self._rolling_duration

    def __enter__(self):
        self.restart()

    def __exit__(self, *args):
        self.stop()


###############################################################################
# Training-loop specific
# Below, the logging utilities are specific to model training.
###############################################################################


def loss_metric(loss):
    """
    A convenience function for creating a loss metric.

    Loss metric is always a good-increasing metric that is checkpointed.
    """
    return Metric("loss", loss, increasing=False, checkpointed=True)


class Metric:
    """A quantity like like loss or accuracy is tracked as a Metric.

    In addition to the value itself, functions that consume metrics typically
    need to know:
        - a name for the metric
        - whether a higher value is better or lower is better

    Class vs. tuples
    ----------------
    Tuples could be used to pass around these values. However, a Metric class
    makes the implicit explicit, and hopefully makes things easier to both
    understand and use. A further benefit of a class is that there is an
    option to make the Meter class be a metric (either duck typing or via
    sub-classing) in order to avoid the very mild annoyance of having to
    meter.avg every time you want to make a metric from a meter.
    """

    def __init__(
        self,
        name: str,
        value: Number,
        increasing: bool = True,
        checkpointed: bool = False,
    ):
        """
        Args:
            checkpointed: Whether we should record checkpoints for the best
                values of this metric.
        """
        self.name = name
        self.value = value
        self.increasing = increasing
        self.checkpointed = checkpointed

    def __str__(self):
        return f"{self.name}={self.value:.5f}"

    def is_better(self, other):
        """Whether the current metric is better than the other metric.

        Args:
            than (Metric or Number): The metric to compare to.

        Returns:
            True if this metric is better than the other metric.
        """
        if isinstance(other, Metric):
            if self.increasing != other.increasing:
                raise ValueError(
                    "Cannot compare an increasing metric to a decreasing one:"
                    f" ({str(self)}, {str(other)})"
                )
            other_val = other.value
        else:
            other_val = other
        if self.increasing:
            return self.value > other_val
        else:
            return self.value < other_val


"""
Previously, the generic class was used for logging data. However, to use this
the evaluate() would return a map whose keys were used to interpret the type
of data. For example, "plotly" meant the value of the dict contained plotly
figures. This was fine until in addition to the input/output figures, a
latent space figure was also to be recorded. So either the structure of the
dict values needed another level, or the keys didn't have to one-to-one with
a data type. Going with the latter option, we could add a "dtype" string
parameter to the LogData object, or we could use separate classes. Both seem
fine. It seems likely that different datatypes might need different or extra
fields, so lets go with classes for now. As the results dict keys are now
freed up to be used as labels, the data objects themselves don't need a label.

class LogData:
    def __init__(self, label: str, content: Any):
        self.label = label
        self.content = content
"""


class PlotlyFigureList:
    """Wrap one or more Plotly figures for logging (e.g. via Tensorboard)."""

    def __init__(self, figs):
        self.figs = figs


class PlotlyVideo:
    """Wrap a list of Plotly figures for logging as a video (e.g. via Tensorboard)."""

    def __init__(self, figs):
        self.figs = figs


class Embeddings:
    """Wrap an embedding matrix for logging (e.g. via Tensorboard)."""

    def __init__(self, embeddings: torch.Tensor, labels: Iterable[str]):
        """Args:
        embedding: A 2D tensor of shape (num_embeddings, embedding_dim).
        """
        self.embeddings = embeddings
        self.labels = labels
        # Not yet connected.
        self.label_img = None


class MetricTracker:
    """Monitors metrics and the epochs they were seen in.

    It's common to write a conditional like:

        if new_metric.is_better(existing_best):
            do_something()
            existing_best = new_metric

    This class does that here, so other classes like checkpointers don't have
    to. This class originally came about by gutting the functionality from
    the ModelCheckpointer when it was needed elsewhere.
    """

    # Class variables
    HISTORY_FILENAME = "metrics.csv"
    BEST_FILENAME = "best_metrics.json"

    # Instance variables
    best_metrics: Dict[str, Number]
    best_metric_epochs = Dict[str, int]
    _best_this_epoch = Sequence[Metric]
    history = Dict[int, Dict[str, Number]]

    def __init__(self, out_dir):
        self.out_dir = pathlib.Path(out_dir)
        # Some recording keeping here. Could probably just use a single
        # authorative dataframe, and have functions query it. But for now,
        # I'll stick with this.
        self.best_metrics = dict()
        self.best_metric_epochs = dict()
        self._best_this_epoch = []
        # To allow new metrics to be added at any time, we won't use any
        # fixed table structure. Instead, just a dicts of dicts, one entry
        # for each epoch recorded. It's a dict of dicts rather than a list
        # of dicts, as we don't want to enforce that the tracker be updated
        # on every epoch. Another detail: we don't store Metric objects, but
        # just the values. The benefit of this is an easy conversion to
        # pandas dataframes.
        self.history = defaultdict(dict)

    def history_path(self) -> pathlib.Path:
        return self.out_dir / self.HISTORY_FILENAME

    def best_path(self) -> pathlib.Path:
        return self.out_dir / self.BEST_FILENAME

    def on_epoch_end(self, metrics: Sequence[Metric], epoch: int):
        """Record this latest epoch's metrics.

        The function name hints that I'm coming around to the idea that
        having training callbacks is eventually going to happen.
        """
        self._best_this_epoch = []
        for metric in metrics:
            if not metric.checkpointed:
                continue
            # Add to history
            self.history[epoch][metric.name] = metric.value
            if math.isnan(metric.value):
                _logger.warn(f"Metric ({metric.name}) is NaN.")
                continue
            current_best = self.best_metrics.get(metric.name, None)
            # If new or better metric encountered, hardlink to epoch checkpoint.
            if current_best is None or metric.is_better(current_best):
                # Log a message if a new metric is encountered.
                if current_best is None:
                    _logger.info(
                        "New metric encountered "
                        f"({metric.name} = {metric.value:.5f}) "
                    )
                else:
                    assert metric.is_better(current_best)
                    gt_lt = ">" if metric.increasing else "<"
                    _logger.info(
                        f"Improved metric ({metric.name}): ".ljust(40)
                        + f"{metric.value:.5f} {gt_lt} {current_best:.5f} "
                        f"(epoch {epoch} {gt_lt} "
                        f"{self.best_metric_epochs[metric.name]})"
                    )
                self.best_metrics[metric.name] = metric.value
                self.best_metric_epochs[metric.name] = epoch
                self._best_this_epoch.append(metric)
        self._write_history()
        self._write_best()
        self._log_best()
        return self._best_this_epoch

    def _write_history(self):
        """Write the metric history as a CSV file."""
        with open(self.history_path(), "w") as f:
            self.history_as_dataframe().to_csv(f, index_label="epoch")

    def _write_best(self):
        with open(self.best_path(), "w") as f:
            json.dump(self.best_metrics, f, indent=2)

    def _log_best(self):
        """Log the best metrics to the console."""
        _logger.info("Best metrics:")
        _logger.info(
            json.dumps(self.best_metrics, indent=2, ensure_ascii=False)
        )

    def improved_metrics(self) -> Sequence[Metric]:
        """
        Returns a list of metrics that have improved since the last update.
        """
        return self._best_this_epoch

    def history_as_dataframe(self):
        """Return the history of metrics as a pandas dataframe.

        The epoch number becomes the index.
        """
        return pd.DataFrame.from_dict(self.history, orient="index")


def print_metrics(metrics):
    _logger.info(" | ".join([f"{m.name}: {m.value:.5f}" for m in metrics]))


def plotly_fig_to_array(fig) -> np.ndarray:
    """Convert plotly figure to numpy array.

    From:
        https://community.plotly.com/t/converting-byte-object-to-numpy-array/40189/3
    """
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = PIL.Image.open(buf)
    array_rgba = np.asarray(img)
    array_rgb = array_rgba[:, :, 0:3]
    return array_rgb


class TbLogger(object):
    """Manages logging to TensorBoard."""

    def __init__(self, tensorboard_dir: Union[str, pathlib.Path]):
        self.writer = tb.SummaryWriter(str(tensorboard_dir))

    @staticmethod
    def tag(label: str, log_group: str):
        res = f"{label}/{log_group}"
        return res

    def log(self, n_iter: int, data, log_group: str):
        """Log a mixture of different types of data."""
        for k, v in data.items():
            # The infamous if-else
            if k == "metrics":
                self.log_metrics(n_iter, v, log_group)
            elif isinstance(v, PlotlyFigureList):
                self.log_plotly(k, n_iter, v, log_group)
            elif isinstance(v, Embeddings):
                self.log_embeddings(k, n_iter, v, log_group)
            elif isinstance(v, PlotlyVideo):
                self.log_plotly_video(k, n_iter, v, log_group)
            else:
                raise ValueError(
                    "Logging the given data is unsupported " f"({data})."
                )

    def log_metrics(self, n_iter: int, metrics, log_group: str):
        for metric in metrics:
            self.writer.add_scalar(
                self.tag(metric.name, log_group), metric.value, n_iter
            )

    def log_scalar(self, n_iter: int, name: str, val, log_group: str):
        self.writer.add_scalar(self.tag(name, log_group), val, n_iter)

    def log_plotly(
        self,
        label: str,
        n_iter: int,
        figs: Union[PlotlyFigureList, Iterable[plotly.graph_objects.Figure]],
        log_group: str,
    ):
        if isinstance(figs, PlotlyFigureList):
            figs = figs.figs
        if not isinstance(figs, collections.abc.Iterable):
            raise ValueError("Expected a iterable of plots.")
        else:
            figs = figs
        plots_as_arrays = [plotly_fig_to_array(p) for p in figs]
        plots_as_array = np.stack(plots_as_arrays)
        self.writer.add_images(
            self.tag(label, log_group),
            plots_as_array,
            n_iter,
            dataformats="NHWC",
        )

    def log_plotly_video(
        self,
        label: str,
        n_iter: int,
        fig_list: PlotlyVideo,
        log_group: str,
    ):
        if not isinstance(fig_list.figs, collections.abc.Iterable):
            raise ValueError("Expected a iterable of plots.")
        plots_as_array = np.array(
            [plotly_fig_to_array(p) for p in fig_list.figs]
        )
        plots_as_tensor = torch.from_numpy(plots_as_array)
        plots_as_5d_tensor = einops.rearrange(
            plots_as_tensor, "(N F) H W C -> N F C H W", N=1
        )
        self.writer.add_video(
            self.tag(label, log_group), plots_as_5d_tensor, n_iter, fps=10
        )

    def log_embeddings(
        self, label: str, n_iter: int, embedding: Embeddings, log_group: str
    ):
        self.writer.add_embedding(
            embedding.embeddings,
            metadata=embedding.labels,
            label_img=embedding.label_img,
            global_step=n_iter,
            tag=self.tag(label, log_group),
        )


class ModelSaver:
    """Saves and loads model checkpoints.

    Keeps a history of checkpoints, including the checkpoints where the best
    metrics were observed.

    Inspired by both pytorch-image-models:
      https://github.com/rwightman/pytorch-image-models/blob/fa8c84eede55b36861460cc8ee6ac201c068df4d/timm/utils/checkpoint_saver.py#L21
    and PyTorch lightning:
      https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/callbacks/model_checkpoint.html#ModelCheckpoint

    A number of differences:

        - We support multiple "best" metrics. This is important
          as when the model can get high accuracy (e.g. 99.9%) in the presence
          of unbalanced data, other metrics such as correlation measures can
          be very low. When analysing the results, it can be useful to have
          the models that are best at each metric.
        - Non-linear checkpoint history. One of the very annoying things about
          the pytorch-image-model checkpointing is that it only keeps recent
          checkpoints. I'm not sure what Pytorch lightning does.
    """

    CKPT_FILENAME_FORMAT = "checkpoint_epoch-{epoch}.pth"
    LAST_CKPT_FILENAME = "checkpoint_last.pth"
    BEST_CKPT_FILENAME_FORMAT = "checkpoint_best_{metric_name}.pth"
    RECOVERY_CKPT_FILENAME = "recovery.pth"

    def __init__(
        self,
        save_dir: Union[str, pathlib.Path],
        model,
        optimizer=None,
        scheduler=None,
        max_history: int = 10,
    ):
        if max_history < 1:
            raise ValueError(
                f"max_history must be greater than zero. Got ({max_history})"
            )
        self.save_dir = pathlib.Path(save_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoints_by_epoch = dict()
        self.max_history = max_history

    @property
    def last_path(self):
        res = self.save_dir / self.LAST_CKPT_FILENAME
        return res

    @property
    def recovery_path(self):
        res = self.save_dir / self.RECOVERY_CKPT_FILENAME
        return res

    def epoch_path(self, epoch: int):
        res = self.save_dir / self.CKPT_FILENAME_FORMAT.format(epoch=epoch)
        return res

    def metric_path(self, metric_name: str):
        res = self.save_dir / self.BEST_CKPT_FILENAME_FORMAT.format(
            metric_name=metric_name
        )
        return res

    def _save_epoch_checkpoint(self, epoch: int):
        # Save the checkpoint as an epoch checkpoint.
        epoch_path = self.epoch_path(epoch)
        retinapy.models.save_model(
            self.model, epoch_path, self.optimizer, self.scheduler
        )
        self.checkpoints_by_epoch[epoch] = epoch_path
        # Link to "last" checkpoint.
        _logger.info(f"Updating symlink ({str(self.last_path)})")
        self.last_path.unlink(missing_ok=True)
        # Note: a tricky aspect of path_A.symlink_to(path_B) is that path_A
        # will be assigned to point to path_A / path_B. And so, we usually
        # will want to call path_A.symlink_to(path_B.name).
        self.last_path.symlink_to(epoch_path.name)
        assert self.last_path.exists()

    def _save_metrics_checkpoint(
        self, improved_metrics: Sequence[Metric], epoch: int
    ):
        for metric in improved_metrics:
            assert not math.isnan(metric.value)
            best_path = self.metric_path(metric.name)
            best_path.unlink(missing_ok=True)
            _logger.info(
                f"Updating (best {metric.name}) checkpoint, {best_path}"
            )
            # Hard link to the epoch checkpoint.
            self.epoch_path(epoch).link_to(best_path)
            # TODO: move to "hardlink_to" when we upgrade to Python 3.10.
            #   best_path.hardlink_to(self.epoch_path(epoch)

    def save_checkpoint(self, epoch: int, improved_metrics: Sequence[Metric]):
        self._save_epoch_checkpoint(epoch)
        self._save_metrics_checkpoint(improved_metrics, epoch)

        # Clean up history, if necessary.
        self._remove_epoch_checkpoints()

    @staticmethod
    def inverse_cumulative_exp(area: float, half_life: float = 0.5):
        """This is the inverse of the cumulative exponential function (base 2).

        f(t) = 2 ** (t / half_life)
        F(t) = int_{0}^{t} f(x) dx
        InvCumExp(x) = F^{-1}(x)   <--- this is what we want.
        """
        # L is used to normalize the area under the curve to 1.
        L = 1 / (2 ** (1 / half_life) - 1)
        # Inv function.
        t = half_life * math.log(area / L + 1, 2)
        return t

    def _remove_epoch_checkpoints(self):
        """
        Remove old checkpoints if we have hit the checkpoint history limit.

        Removal tries to be "smart" by spreading out the removal. Why? Because
        if we keep the N-most recent checkpoints, we can't go back further
        than N epochs prior. This is definitely a problem, as often, when we
        notice an issue with training, such as NaN values, we want to go back
        more than N epochs prior.

        The solution taken here is to break up the checkpoint history into
        weighted areas, and to remove checkpoints if two fall into the same
        region. The areas are weighted exponentially, so that we keep a larger
        number of more recent checkpoints, and fewer older checkpoints.
        """
        assert self.max_history > 0
        if len(self.checkpoints_by_epoch) <= self.max_history:
            # We haven't reached the max number of checkpoints yet.
            return
        # Remove the newest checkpoint that doesn't fit under the easing curve.
        saved_epochs = sorted(self.checkpoints_by_epoch.keys())
        max_epoch = saved_epochs[-1]
        to_remove = None
        prev_zone = -1
        to_remove = saved_epochs[0]
        for e in saved_epochs:
            zone = self.inverse_cumulative_exp(e / max_epoch)
            zone_int = math.floor(zone * self.max_history)
            if zone_int == prev_zone:
                to_remove = e
                break
            prev_zone = zone_int

        assert to_remove is not None, "There must be checkpoints already saved."
        # Remove the identified checkpoint.
        file_to_remove = self.checkpoints_by_epoch.pop(to_remove)
        # Unlink old checkpoint. Note that it might still be linked as the best
        # checkpoint, and thus the file might not be deleted.
        file_to_remove.unlink()
        _logger.info(f"Unlinked old checkpoint: ({str(file_to_remove)})")

    def save_recovery(self):
        """Trigger the saving of a checkpoint to the recovery path.

        The recovery checkpoint is the only checkpoint taken mid-epoch. It is
        created periodically to allow training to restart in the event of an
        error.
        """
        _logger.info(
            "Creating recovery checkpoint: " f"({str(self.recovery_path)})"
        )
        retinapy.models.save_model(
            self.model, self.recovery_path, self.optimizer, self.scheduler
        )

    def delete_recovery(self):
        """Delete the recovery checkpoint.

        This can be manually whenever the recovery checkpoint becomes unneeded,
        typically once training is finished.
        """
        if self.recovery_path.exists():
            if not self.recovery_path.is_file():
                raise Warning(
                    "Recovery path exists but is not a file "
                    f"({str(self.recovery_path)})"
                )
            self.recovery_path.unlink()
