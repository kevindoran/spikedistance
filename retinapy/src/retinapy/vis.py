"""
This module is starting out as somewhere to create the most commonly figures 
needed in notebooks. Going forward, it's hoped that some of the plots or 
images created here can be recorded automatically when training a model.
"""

import concurrent.futures
import functools
import logging
import math
import pathlib
from typing import Optional
from typing import Iterable, Union
import PIL.Image
import numpy as np
import scipy
import plotly
import plotly.graph_objects as go
import plotly.subplots
import retinapy
import retinapy.dataset
import retinapy.mea as mea
import retinapy.spikeprediction as sp
import scinot
import sklearn.manifold


_logger = logging.getLogger(__name__)


def create_title(title: str, subtitle: Optional[str] = None) -> str:
    """
    Returns a styled title string for use in plotly.

    Subtitle is optional.
    """
    if subtitle is None:
        res = f"{title}"
    else:
        res = f'{title}<br><span style="font-size:90%; color:grey">{subtitle}</span>'
    return res


def create_axis_title(title: str, units: str) -> str:
    """Returns a styled title string for use in plotly."""
    return f"{title}<span style='font-size:90%;white-space:pre;color:grey'>  ({units})</span>"


def default_fig_layout():
    """Some default fig options.

    For quite a few of these, like margin, I don't quite understand their
    full implications. This is part of the reason it's useful to move them
    here, as they will likely change over time.
    """
    res = {
        "title_x": 0.5,
        "title_pad": dict(l=0, r=0, b=0, t=0),
        # The title and xaxis label fit within the margin, so it needs
        # to be big enough to fit them. Top margin of 50 or so works for a
        # title with no subtitle. 70 works for a title with a subtitle.
        # 50 is okay for bottom, but it's a little too tight in Jupyter Lab,
        # so making it a tad bigger (60).
        "margin": {"l": 0, "r": 0, "t": 50, "b": 60, "pad": 0},
    }
    return res


def kernel(
    kernel: np.ndarray,
    t_0: int,
    bin_duration_ms,
    vline: Optional[int] = None,
    line_width: float = 3.0,
):
    """
    Args:
        kernel: The kernel to plot. This is a 2D array with shape
            (n_timesteps, n_channels). There can be 1-4 channels.
        t_0: the index of the bin that corresponds to t=0.
    """
    fig = go.Figure()
    xs = (np.arange(kernel.shape[0]) - t_0) * bin_duration_ms
    if vline is not None:
        fig.add_vline(
            x=-t_0,
            line_width=line_width,
            line_dash="dot",
            line_color="grey",
            annotation_text="-100ms",
            annotation_position="bottom right",
        )
    for idx, stim in enumerate(mea.stimuli):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kernel[:, idx],
                line_color=stim.display_hex,
                mode="lines",
            )
        )
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        margin=dict(l=1, r=1, b=1, t=25, pad=10),
        yaxis_fixedrange=True,
        showlegend=False,
        title={"text": "Kernel"},
        xaxis={"title": "time (ms), with spike at 0"},
        yaxis={"title": "Stimulus", "range": [0, 1]},
    )
    return fig

def kernel_flat(
    kernel: np.ndarray,
    t_0: int,
    bin_duration_ms,
    vline: Optional[int] = None,
    line_width: float = 3.0):
    """
    Args:
        kernel: The kernel to plot. This is a 2D array with shape
            (n_timesteps, n_channels). There can be 1-4 channels.
        t_0: the index of the bin that corresponds to t=0.
    """
    fig = go.Figure()
    xs = (np.arange(kernel.shape[0]) - t_0) * bin_duration_ms
    if vline is not None:
        fig.add_vline(
            x=-t_0,
            line_width=line_width,
            line_dash="dot",
            line_color="grey",
            annotation_text="-100ms",
            annotation_position="bottom right",
        )
    
    # Create subplots
    fig = plotly.subplots.make_subplots(rows=1, cols=4, 
                                        shared_yaxes=True)
    
    for idx, stim in enumerate(mea.stimuli):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kernel[:, idx],
                line_color=stim.display_hex,
                mode="lines",
            ),
            row=1,
            col=idx + 1,
        )
    
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        margin=dict(l=1, r=1, b=1, t=25, pad=10),
        yaxis_fixedrange=True,
        showlegend=False,
        title={"text": "Kernel"},
        xaxis={"title": "time (ms), with spike at 0"},
        yaxis={"title": "Stimulus", "range": [0, 1]},
        grid={"rows": 1, "columns": 4},  # Add grid layout for subplots
    )
    
    return fig

def kernel2(
    ds_rec: mea.SpikeRecording,
    c_idx: int,
    snippet_len: int,
    snippet_pad: int,
    style: str,
):
    snippets = mea.spike_snippets(
        ds_rec.stimulus,
        mea.compress_spikes(ds_rec.spikes[:, c_idx]),
        snippet_len,
        snippet_pad,
    )
    allowed_styles = ("mini", "no-text", "normal")
    if style not in allowed_styles:
        raise ValueError(f"style must be one of {allowed_styles}.")
    # Kernel and spike counts.
    c_id = ds_rec.cluster_ids[c_idx]
    ker = snippets.mean(axis=0)
    num_spikes = len(snippets)
    # Make figure.
    bin_duration_ms = 1000 / ds_rec.sample_rate
    t_0 = snippet_len - snippet_pad
    # Use thicker lines for the no-text version.
    line_width = 4 if style == "no-text" else 2
    # Create the kernel.
    fig = kernel(
        ker, t_0=t_0, bin_duration_ms=bin_duration_ms, line_width=line_width
    )
    if style == "mini":
        fig.update_layout(
            {
                "yaxis": {"visible": False},
                "xaxis": {"title": None},
                "title": {
                    "text": (
                        f'<span style="font-size:75%">{ds_rec.name}</span><br>'
                        f'<span style="font-size:95%">c{c_id}</span> '
                        '<span style="font-size:85%">'
                        f"({scinot.format(num_spikes, 2)} spikes</span>)"
                    ),
                },
                "width": 200,
                "height": 200,
                "margin": {"l": 0},
            }
        )
    elif style == "no-text":
        fig.update_layout(
            {
                # Slightly narrower y-axis.
                "yaxis": {"visible": False, "range": [0.15, 0.75]},
                "xaxis": {"visible": False, "title": None},
                "title": None,
                "width": 200,
                "height": 200,
                "margin": {"l": 0, "r": 0, "t": 0, "pad": 0},
            }
        )
    else:
        fig.update_layout(
            {
                "title": {
                    "text": create_title(
                        "STA kernel",
                        f"rec: {ds_rec.name}, cluster: {c_id}",
                    )
                }
            }
        )
    return fig


class KernelPlots:
    """Creates and saves kernel plots as images to disk.

    Motivation:
    This class was created to speed up the process of creating kernel plots when
    they were needed in bulk for hover-over tooltips.
    """

    FILE_PATH_PATTERN = "{rec_name}_c{cluster_id}.png"
    OUT_DIR = "kernel_plots"

    def __init__(self, img_dir: Union[str, pathlib.Path]):
        self.img_dir = pathlib.Path(img_dir)
        if not self.img_dir.is_dir():
            raise ValueError(
                "img_dir must be a directory (with the saved kernel plots)."
            )
        self.cache = {}

    def get(self, rec_name, cluster_id):
        if (rec_name, cluster_id) in self.cache:
            return self.cache[(rec_name, cluster_id)]
        img = PIL.Image.open(
            self.img_dir
            / self.FILE_PATH_PATTERN.format(
                rec_name=rec_name, cluster_id=cluster_id
            )
        )
        self.cache[(rec_name, cluster_id)] = img
        return img

    @staticmethod
    def _generate_single(
        ds_rec, c_idx, snippet_len, snippet_pad, style, out_dir
    ):
        c_id = ds_rec.cluster_ids[c_idx]
        fig = kernel2(ds_rec, c_idx, snippet_len, snippet_pad, style)
        # Save figure.
        fig.write_image(
            out_dir
            / KernelPlots.FILE_PATH_PATTERN.format(
                rec_name=ds_rec.name, cluster_id=c_id
            )
        )

    @classmethod
    def generate(
        cls,
        recs: Iterable[mea.SpikeRecording],
        snippet_len: int,
        snippet_pad: int,
        out_dir: Union[str, pathlib.Path],
        style="normal",  # normal, mini, no-text
        include_text=True,
        num_workers=4,
    ) -> "KernelPlots":
        """
        Args:
            recs: The recordings to generate the kernel plots for.
            snippet_len: The length of the snippet to use for the kernel.
            snippet_pad: The number of padding bins to use after the spike.
            out_dir: The directory to save the plots to.
            style: if 'mini' make the plots smaller by removing axes and making
                the images smaller.
        """
        # Create outer directory.
        out_dir = pathlib.Path(out_dir) / cls.OUT_DIR
        exists_and_not_empty = out_dir.is_dir() and len(list(out_dir.iterdir()))
        if exists_and_not_empty:
            _logger.warning(
                f"out_dir already exists and has contents ({out_dir})"
            )
        out_dir.mkdir(parents=False, exist_ok=True)

        _logger.info(f"Creating kernel plots in {out_dir}")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            _func = functools.partial(
                cls._generate_single,
                snippet_len=snippet_len,
                snippet_pad=snippet_pad,
                style=style,
                out_dir=out_dir,
            )
            futures = {}
            for ds_rec in recs:
                for c_idx in range(len(ds_rec.cluster_ids)):
                    f = executor.submit(_func, ds_rec=ds_rec, c_idx=c_idx)
                    futures[f] = (ds_rec, c_idx)
            # Wait for all futures to complete.
            for future in concurrent.futures.as_completed(futures):
                ds_rec, c_idx = futures[future]
                try:
                    future.result()
                except Exception as err:
                    _logger.error(
                        f"Error creating kernel plot for cluster {c_idx} of "
                        f"recording {ds_rec.name}: {err}"
                    )
        _logger.info(f"Finished creating kernel plots")
        return KernelPlots(out_dir)


def stimulus_fig(
    stimulus: np.ndarray, start_ms: float = 0, bin_duration_ms: float = 1.0
):
    """
    A figure to display a stimulus.

    Args:
        stimulus: the 4 color stimulus, with shape (n_channels=4, n_timesteps)
    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_width=[0.2, 0.8],
        x_title=create_axis_title("time", "ms"),
    )
    xs = np.arange(stimulus.shape[1]) * bin_duration_ms + start_ms
    # RGBU
    for idx, stim in enumerate(mea.stimuli):
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=stimulus[idx, :],
                line_color=stim.display_hex,
                name=f"{stim.wavelength} nm",
                mode="lines",
            ),
            row=1,
            col=1,
        )
    # sum
    fig.append_trace(
        go.Scatter(
            x=xs,
            y=stimulus.sum(axis=0),
            line_color="black",
            name="sum",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        {"tickmode": "array", "tickvals": [-5, 0, 5]}, row=2, col=1
    )
    # Set a default layout.
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        {
            "title": {"text": "Stimulus"},
            # The title and xaxis label fit within the margin, so it needs
            # to be big enough to fit them.
            "yaxis": {
                "title": {"text": create_axis_title("stimulus", "no units")},
                "fixedrange": True,
                # The range is dependent on the agumentation.
                "range": [-2, 2],
                "tickmode": "array",
                "tickvals": [-1, 0, 1],
            },
        }
    )
    return fig


def distfield_fig(
    actual: np.ndarray,
    pred: np.ndarray,
    start_ms: float = 0,
    bin_duration_ms: float = 1.0,
    stride_bins: int = 0,
    log_space: bool = False,
):
    """
    A figure containing 1 or multiple line charts comparing distance fields.

    Args:
        stride: the jump in the x-axis origin label between each subplot. This
            is used to plot multiple times where each subplot starts say
            50 ms after the previous one.
    """
    if actual.shape != pred.shape:
        raise ValueError(
            "Both distfield inputs must have the same shape. Got "
            f"{actual.shape} and {pred.shape}."
        )
    # Expand dims if only one distfield is given (single or multiple distfields
    # are accepted).
    if len(actual.shape) == 1:
        actual = actual.reshape(1, -1)
        pred = pred.reshape(1, -1)
    num_rows = len(actual)
    # Initialize the figure, in its subplot glory.
    fig = plotly.subplots.make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=False,
        # Default vertical spacing is 0.3/num_rows
        vertical_spacing=0.5 / num_rows,
        # Make the plots have the same y-range.
        shared_yaxes="columns",
        x_title=create_axis_title("time", "ms"),
        y_title=(
            create_axis_title("log time to nearest spike", "log(ms)")
            if log_space
            else create_axis_title("time to nearest spike", "ms")
        ),
    )
    # Create the scatters.
    stride_ms = stride_bins * bin_duration_ms
    num_bins = actual.shape[1]
    for r in range(num_rows):
        start = start_ms + r * stride_ms
        xs = np.arange(num_bins) * bin_duration_ms + start
        showlegend = r == 0
        scatter_actual = go.Scatter(
            x=xs,
            y=actual[r, :],
            name="actual",
            mode="lines",
            line_color="tomato",
            showlegend=showlegend,
            legendgroup="actual",
        )
        scatter_pred = go.Scatter(
            x=xs,
            y=pred[r, :],
            name="pred",
            mode="lines",
            line_color="gray",
            showlegend=showlegend,
            legendgroup="pred",
        )
        fig.add_trace(scatter_actual, row=r + 1, col=1)
        fig.add_trace(scatter_pred, row=r + 1, col=1)
    # Determine the y-axis scale. These are defaults, free to be changed.
    if log_space:
        yaxis_range = [-4, 5]
    else:
        yaxis_range = [-1, 600]
    # Create a title.
    total_len = actual.shape[1] * bin_duration_ms
    title_start = "Log distance fields" if log_space else "Distance fields "
    stride_str = (
        f"strided by {(stride_bins)} bins ({stride_ms:.1f} ms)"
        if stride_bins > 0
        else "no stride specified"
    )

    title_str = create_title(
        f"{title_start}, actual and predicted",
        f"{num_rows} snippets, "
        f"{num_bins} bins x {bin_duration_ms:.1f} ms/bin "
        f"= {total_len:.1f} ms each, {stride_str} ",
    )

    height = 140 * num_rows
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        {
            "height": height,
            "yaxis": {
                "range": yaxis_range,
                "fixedrange": True,
            },
            "title": {"text": title_str},
            # Need left margin to fit the shared y-axis title, and some extra
            # on top for the subtitle.
            "margin": {"l": 70, "t": 70},
        }
    )
    return fig


def pixelcnn_model_in_out(
    stimulus: np.ndarray,
    dist_actual: np.ndarray,
    model_out: np.ndarray,
    max_dist: float,
    start_ms=0,
    bin_duration_ms=1.0,
    cluster_label=None,
):
    """TODO


    +--------------------+
    |      stimulus      |
    +--------------------+
    | dist               |
    +--------------------+

    """
    fig = plotly.subplots.make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        # row_width=[0.15, 0.15, 0.15, 0.15, 0.15, 0.45],
        x_title=create_axis_title("time", "ms"),
    )
    snip_len = stimulus.shape[1]
    dist_pred = model_out
    if len(dist_pred.shape) != 1:
        raise ValueError(
            f"Expected a 1D array for dist_pred; got ({dist_pred.shape})."
        )
    pred_len = dist_pred.shape[0]

    # 1. The 4-channel stimulus data.
    xs = np.arange(snip_len) * bin_duration_ms + start_ms
    # RGBU
    for idx, stim in enumerate(mea.stimuli):
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=stimulus[idx, :],
                line_color=stim.display_hex,
                name=f"{stim.wavelength} nm",
                mode="lines",
            ),
            row=idx + 1,
            col=1,
        )
        fig.update_yaxes(
            {
                "tickmode": "array",
                "tickvals": [-1, 0, 1],
                "fixedrange": True,
                "title_text": ["R", "G", "B", "UV"][idx],
                "title_standoff": 0,
            },
            row=idx + 1,
            col=1,
        )

    # 2. The summed stimulus data.
    fig.append_trace(
        go.Scatter(
            x=xs,
            y=stimulus.sum(axis=0),
            line_color="black",
            name="sum",
            mode="lines",
        ),
        row=5,
        col=1,
    )
    fig.update_yaxes(
        {
            "tickmode": "array",
            "tickvals": [-5, 0, 5],
            "fixedrange": True,
            "range": [-6, 6],
        },
        row=5,
        col=1,
    )

    pred_start_idx = snip_len - pred_len
    xs_dist = xs[pred_start_idx:]
    if len(xs_dist) != len(model_out):
        raise ValueError(
            "The spike mask isn't the same lengths as the model output. "
            f"Got ({len(xs_dist)}) and ({len(model_out)})"
        )
    if pred_start_idx + len(model_out) != snip_len:
        raise ValueError(
            "The stimulus length doesn't allign with the model " "output."
        )
    fig.append_trace(
        go.Scatter(
            x=xs,
            y=dist_actual,
            name="actual",
            mode="lines",
            line_color="tomato",
        ),
        col=1,
        row=6,
    )
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=model_out,
            name="pred",
            mode="lines",
            line_color="gray",
        ),
        col=1,
        row=6,
    )
    fig.update_yaxes({"range": [-2 * max_dist, max_dist + 0.5]}, row=6, col=1)
    # Cover the whole output region with a vrect to separate the regions.
    fig.add_vrect(
        x0=xs[pred_start_idx],
        x1=xs[-1],
        fillcolor="aqua",
        opacity=0.25,
        col=1,
        row=6,
        line_width=0,
        layer="below",
    )

    # Set a default layout.
    fig.update_layout(default_fig_layout())
    # Create title
    main_title_str = "Spike distance model input-output"
    if cluster_label is not None:
        main_title_str += f" ({cluster_label})"
    title = create_title(
        main_title_str,
        "Prediction region is light-blue. Dashed lines represent spikes.",
    )
    fig.update_layout(
        {
            "title": {"text": title},
            "height": 600,
            "margin": {"t": 70, "l": 100, "r": 100},
        }
    )
    fig.add_annotation(
        text=create_axis_title("RGB-UV stimulus (normalized)", "no units"),
        xref="paper",
        yref="paper",  # font_color="grey",
        x=-0.1,
        y=0.8,
        textangle=-90,
        showarrow=False,
    )
    fig.add_annotation(
        text="<span style=''>spike distance<br>(log, normalized)<br>"
        "<span style='font-size:90%;whitespace:pre;color:grey'>log(ms)</span>",
        xref="paper",
        yref="paper",  # font_color="grey",
        x=1.15,
        y=0.0,
        showarrow=False,
    )
    return fig


def dist_model_in_out(
    stimulus: np.ndarray,
    in_spikes: np.ndarray,
    target_dist: np.ndarray,
    model_out: np.ndarray,
    pos_enc: Optional[np.ndarray] = None,
    out_spikes: Optional[np.ndarray] = None,
    start_ms=0,
    bin_duration_ms=1.0,
    dist_prefix_len: int = 0,
    cluster_label=None,
):
    """A figure to view the inputs and outputs of a distfield model.

    Create a figure that has the stimulus on top, and the input spikes and
    the pair of distfields on the bottom (actual, pred). Roughly formatted
    like so:

    +--------------------+
    |      stimulus      | (includes the position encoding, if any)
    +--------------------+
    | spikes      | dist |
    +--------------------+

    """
    # The figure will be built in 4 steps:
    #   1. The 4-channel stimuli that spans the whole x-axis.
    #       1.1 Draw a 5th line for the position encoding, if any.
    #   2. The sum of the 4-channel stimuli (in black). This line is plotted
    #      below the 4-channel stimuli and also spans the whole x-axis.
    #   3. The model output distance field and the true (target) distance field.
    #      These lines occupy a small time-slice on the right, typically around
    #      50-200 ms depending on the model.
    #   4. The input spike data, shown as vertical lines on the same sub-plot
    #      as the distance fields.

    # It would be nice to reuse the functionality of the above two functions;
    # however, there are just enough differences for it to be not so easy.

    fig = plotly.subplots.make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        # row_width=[0.15, 0.15, 0.15, 0.15, 0.15, 0.45],
        x_title=create_axis_title("time", "ms"),
    )

    # 1. The 4-channel stimulus data.
    stim_len = stimulus.shape[1]
    x_max = max(stim_len, len(in_spikes) + len(target_dist) - dist_prefix_len)
    xs = np.arange(x_max) * bin_duration_ms + start_ms
    # RGBU
    for idx, stim in enumerate(mea.stimuli):
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=stimulus[idx, :],
                line_color=stim.display_hex,
                name=f"{stim.wavelength} nm",
                mode="lines",
            ),
            row=idx + 1,
            col=1,
        )
        fig.update_yaxes(
            {
                "tickmode": "array",
                "tickvals": [-1, 0, 1],
                "fixedrange": True,
                "title_text": ["R", "G", "B", "UV"][idx],
                "title_standoff": 0,
            },
            row=idx + 1,
            col=1,
        )
    if pos_enc is not None:
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=pos_enc,
                line_color="black",
                name="pos enc",
                mode="lines",
            ),
            row=5,
            col=1,
        )

    # 2. The summed stimulus data.
    fig.append_trace(
        go.Scatter(
            x=xs,
            y=stimulus.sum(axis=0),
            line_color="black",
            name="sum",
            mode="lines",
        ),
        row=5,
        col=1,
    )
    fig.update_yaxes(
        {
            "tickmode": "array",
            "tickvals": [-5, 0, 5],
            "fixedrange": True,
            "range": [-6, 6],
        },
        row=5,
        col=1,
    )

    # 3. The model output distance field and the target (actual).
    mask_start_idx = len(in_spikes) - dist_prefix_len
    xs_dist = xs[mask_start_idx:]
    if len(xs_dist) != len(model_out):
        raise ValueError(
            "The spike mask isn't the same lengths as the model output. "
            f"Got ({len(xs_dist)}) and ({len(model_out)})"
        )
    # It's okay for stimulus length and (spikes + model output) to not
    #  be the same: it could be a model without a masked area.
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=model_out,
            name="pred",
            mode="lines",
            line_color="gray",
            line_width=0.1,
        ),
        col=1,
        row=6,
    )
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=target_dist,
            name="actual",
            mode="lines",
            line_color="tomato",
        ),
        col=1,
        row=6,
    )
    fig.update_yaxes({"range": [-1.0, 6.0]}, row=6, col=1)
    # Cover the whole output region with a vrect to separate the regions.
    fig.add_vrect(
        x0=xs[mask_start_idx],
        x1=xs[-1],
        fillcolor="aqua",
        opacity=0.25,
        col=1,
        row=6,
        line_width=0,
        layer="below",
    )

    # 4.1. The vertical lines marking the input spikes.
    index_of_spikes = np.flatnonzero(in_spikes > 0)
    for idx in index_of_spikes:
        spike_loc = start_ms + idx * bin_duration_ms
        fig.add_vline(
            x=spike_loc, line_color="tomato", line_dash="dot", row=6, col=1
        )

    # 4.2. The vertical lines marking the output spikes.
    if out_spikes is not None:
        # Spikes at index zero correspond to t=0, so we don't need to subtract
        # the offset due to the distance prefix.
        spike_start_idx = len(in_spikes)
        index_of_spikes = np.flatnonzero(out_spikes > 0) + spike_start_idx
        for idx in index_of_spikes:
            spike_loc = start_ms + idx * bin_duration_ms
            fig.add_vline(
                x=spike_loc, line_color="gray", line_dash="dot", row=6, col=1
            )

    # Set a default layout.
    fig.update_layout(default_fig_layout())
    # Create title
    main_title_str = "Spike distance model input-output"
    if cluster_label is not None:
        main_title_str += f" ({cluster_label})"
    title = create_title(
        main_title_str,
        "Prediction region is light-blue. Dashed lines represent spikes.",
    )
    fig.update_layout(
        {
            "title": {"text": title},
            "height": 600,
            "margin": {"t": 70, "l": 100, "r": 100},
        }
    )
    fig.add_annotation(
        text=create_axis_title("RGB-UV stimulus (normalized)", "no units"),
        xref="paper",
        yref="paper",  # font_color="grey",
        x=-0.1,
        y=0.8,
        textangle=-90,
        showarrow=False,
    )
    fig.add_annotation(
        text="<span style=''>spike distance<br>(log, normalized)<br>"
        "<span style='font-size:90%;whitespace:pre;color:grey'>log(ms)</span>",
        xref="paper",
        yref="paper",  # font_color="grey",
        x=1.15,
        y=0.0,
        showarrow=False,
    )
    return fig


def dist_model_io_slim(
    stimulus: np.ndarray,
    in_spikes: np.ndarray,
    target_dist: np.ndarray,
    model_out: np.ndarray,
    out_spikes: Optional[np.ndarray] = None,
    start_ms=0,
    bin_duration_ms=1.0,
    dist_prefix_len: int = 0,
    cluster_label=None,
):
    """A figure to view the inputs and outputs of a distfield model.

    In this version, the stimulus is condensed to a single row.

    Create a figure that has the stimulus on top, and the input spikes and
    the pair of distfields on the bottom (actual, pred). Roughly formatted
    like so:

    +--------------------+
    |      stimulus      |
    +--------------------+
    | spikes      | dist |
    +--------------------+

    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_width=[0.5, 0.5],
        x_title=create_axis_title("time", "ms"),
    )

    # 1. The 4-channel stimulus data.
    stim_len = stimulus.shape[1]
    x_max = max(stim_len, len(in_spikes) + len(target_dist) - dist_prefix_len)
    xs = np.arange(x_max) * bin_duration_ms + start_ms
    # RGBU
    for idx, stim in enumerate(mea.stimuli):
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=stimulus[idx, :],
                line_color=stim.display_hex,
                name=f"{stim.wavelength} nm",
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(
            {
                "tickmode": "array",
                "tickvals": [-1, 0, 1],
                "fixedrange": True,
                "title_text": ["R", "G", "B", "UV"][idx],
                "title_standoff": 0,
            },
            row=1,
            col=1,
        )

    # 2. The model output distance field and the target (actual).
    mask_start_idx = len(in_spikes) - dist_prefix_len
    xs_dist = xs[mask_start_idx:]
    if len(xs_dist) != len(model_out):
        raise ValueError(
            "The spike mask isn't the same lengths as the model output. "
            f"Got ({len(xs_dist)}) and ({len(model_out)})"
        )
    # It's okay for stimulus length and (spikes + model output) to not
    #  be the same: it could be a model without a masked area.
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=model_out,
            name="pred",
            mode="lines",
            line_color="gray",
        ),
        col=1,
        row=2,
    )
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=target_dist,
            name="actual",
            mode="lines",
            line_color="tomato",
        ),
        col=1,
        row=2,
    )
    fig.update_yaxes({"range": [-1, 5]}, row=2, col=1)
    # Cover the whole output region with a vrect to separate the regions.
    fig.add_vrect(
        x0=xs[mask_start_idx],
        x1=xs[-1],
        fillcolor="aqua",
        opacity=0.25,
        col=1,
        row=2,
        line_width=0,
        layer="below",
    )

    # 4.1. The vertical lines marking the input spikes.
    index_of_spikes = np.flatnonzero(in_spikes > 0)
    for idx in index_of_spikes:
        spike_loc = start_ms + idx * bin_duration_ms
        fig.add_vline(
            x=spike_loc, line_color="tomato", line_dash="dot", row=2, col=1
        )

    # 4.2. The vertical lines marking the output spikes.
    if out_spikes is not None:
        # Spikes at index zero correspond to t=0, so we don't need to subtract
        # the offset due to the distance prefix.
        spike_start_idx = len(in_spikes)
        index_of_spikes = np.flatnonzero(out_spikes > 0) + spike_start_idx
        for idx in index_of_spikes:
            spike_loc = start_ms + idx * bin_duration_ms
            fig.add_vline(
                x=spike_loc, line_color="gray", line_dash="dot", row=2, col=1
            )

    # Set a default layout.
    fig.update_layout(default_fig_layout())
    # Create title
    main_title_str = "Spike distance model input-output"
    if cluster_label is not None:
        main_title_str += f" ({cluster_label})"
    title = create_title(
        main_title_str,
        "Prediction region is light-blue. Dashed lines represent spikes.",
    )
    fig.update_layout(
        {
            "title": {"text": title},
            "height": 600,
            "margin": {"t": 70, "l": 100, "r": 100},
        }
    )
    fig.add_annotation(
        text=create_axis_title("RGB-UV stimulus (normalized)", "no units"),
        xref="paper",
        yref="paper",  # font_color="grey",
        x=-0.1,
        y=0.8,
        textangle=-90,
        showarrow=False,
    )
    fig.add_annotation(
        text="<span style=''>spike distance<br>(log, normalized)<br>"
        "<span style='font-size:90%;whitespace:pre;color:grey'>log(ms)</span>",
        xref="paper",
        yref="paper",  # font_color="grey",
        x=1.15,
        y=0.0,
        showarrow=False,
    )
    return fig


def dist_model_out(
    in_spikes: np.ndarray,
    target_spikes: np.ndarray,
    target_dist: np.ndarray,
    model_out: np.ndarray,
    pred_spikes: np.ndarray,
    x_start_ms: float,
    dist_prefix_len: int,
    bin_duration_ms: float,
    title=None,
):
    """Plot comparing actual and output distance array.

      +--------------------+
      | spikes |  _ dist   |
      +--------------------+
      ^        ^  ^
    x_start    |  |
               | x0=t0
       (x0 - dist_prefix_len)

       Example:
           (x_start, dist_prefix_len) = (-200, 26)

    """
    # Settings
    line_width = 1.0
    # Calculate the shared x-axis range.
    if dist_prefix_len < 0:
        raise ValueError(
            "The distance prefix length should be positive. "
            f"Got ({dist_prefix_len})."
        )
    dist_prefix_ms = dist_prefix_len * bin_duration_ms
    if x_start_ms > -dist_prefix_ms:
        raise ValueError(
            "The x-axis should start at a bin before the "
            "beginning of the distance output. (x_start_ms: "
            f"{x_start_ms}, dist_prefix (ms): {dist_prefix_ms})"
        )
    x_end_ms = math.ceil((len(model_out) - dist_prefix_len) * bin_duration_ms)
    dist_xs = (np.arange(len(model_out)) - dist_prefix_len) * bin_duration_ms

    fig = go.Figure()

    def add_dist():
        fig.add_trace(
            go.Scatter(
                x=dist_xs,
                y=model_out,
                name="pred",
                mode="lines",
                line_color="gray",
                line_width=line_width,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=dist_xs,
                y=target_dist,
                name="actual",
                mode="lines",
                line_color="tomato",
                line_width=line_width,
            )
        )

    def add_actual_spikes():
        x_start_idx = math.ceil(x_start_ms / bin_duration_ms)
        actual_spikes = np.concatenate([in_spikes[x_start_idx:], target_spikes])
        index_of_spikes = np.flatnonzero(actual_spikes > 0)
        #
        #      |       |-----remainder------|
        # x_start_ms          (x_start_idx * bin_duration_ms)
        spike_start_ms = x_start_idx * bin_duration_ms
        spike_loc_ms = index_of_spikes * bin_duration_ms + spike_start_ms
        for loc_ms in spike_loc_ms:
            assert x_start_ms <= loc_ms <= x_end_ms, (
                f"Spike location must be within ({x_start_ms}, {x_end_ms})."
                f" Got {loc_ms}."
            )
            fig.add_vline(
                x=loc_ms,
                line_color="tomato",
                line_width=line_width,
                line_dash="dot",
            )

    def add_pred_spikes():
        index_of_spikes = np.flatnonzero(pred_spikes > 0)
        spike_loc_ms = index_of_spikes * bin_duration_ms
        for loc_ms in spike_loc_ms:
            assert x_start_ms <= loc_ms <= x_end_ms, (
                f"Spike location must be within ({x_start_ms}, {x_end_ms})."
                f" Got {loc_ms}."
            )
            fig.add_vline(
                x=loc_ms,
                line_color="gray",
                line_width=line_width,
                line_dash="dot",
            )

    def add_zone():
        fig.add_vrect(
            x0=0,
            x1=len(pred_spikes) * bin_duration_ms,
            fillcolor="aqua",
            opacity=0.1,
            line_width=0,
            layer="below",
        )

    def add_title():
        fig.update_layout(
            {
                "title": {
                    "text": (
                        f'<span style="font-size:75%">{title}</span><br>'
                    ),
                }
            }
        )

    def layout():
        fig.update_layout(default_fig_layout())
        fig.update_layout(
            {
                "showlegend": False,
                "height": 300,
                "width": 800,
                "xaxis": {"range": [x_start_ms, x_end_ms]},
                "yaxis": {"range": [-1.0, 5.5]},
            }
        )

    add_dist()
    add_actual_spikes()
    add_pred_spikes()
    add_zone()
    add_title()
    layout()

    return fig


def latent_tsne_fig(
    rec_ids: np.ndarray,
    cluster_ids: np.ndarray,
    zs: np.ndarray,
    use_label=False,
    perplexity=25,
):
    """Arbitrary dimensional latent points plotted in 2D using t-SNE.

    Args:
        perplexity: "perplexity" parameter for t-SNE. This option makes a big
            difference to the output. The default has been set based on some
            quick trial and error and guesswork, while playing with the latent
            spike of a 40D transformer VAE model.
    """

    # 1. Run t-SNE.
    z_2d = sklearn.manifold.TSNE(
        n_components=2,
        early_exaggeration=12,  # default = 12
        learning_rate="auto",
        n_iter=5000,  # default = 1000
        min_grad_norm=1e-7,  # default = 1e-7
        init="random",  # default = 'random', alternative 'pca'
        angle=0.6,  # default = 0.5. Higher is faster.
        n_jobs=20,  # default = 1
        perplexity=25,
    ).fit_transform(zs)

    # 1. Gather the data to plot. We will actually do that here, so this is
    # quite a proactive plotting function.
    fig = go.Figure()
    labels = [
        f"({r_idx}, {c_id})" for (r_idx, c_id) in zip(rec_ids, cluster_ids)
    ]
    mode = "markers+text" if use_label else "markers"
    scatter = go.Scatter(
        x=z_2d[:, 0],
        y=z_2d[:, 1],
        text=labels,
        textposition="bottom center",
        mode=mode,
    )
    fig.add_trace(scatter)
    fig.update_layout(
        {
            "title": {
                "text": f"Latent space, (t-SNE reduction from {zs.shape[1]} "
                "dimensions)"
            },
        }
    )
    return fig


def latent2d_fig(
    rec_names: Iterable[str],
    cluster_ids: Iterable[int],
    z_xs: np.ndarray,
    z_ys: np.ndarray,
    use_label=False,
):
    """Basic 2D plot of 2D latent variables."""
    # 1. Gather the data to plot. We will actually do that here, so this is
    # quite a proactive plotting function.
    fig = go.Figure()
    labels = [
        f"({r_name}, {c_id})" for (r_name, c_id) in zip(rec_names, cluster_ids)
    ]
    mode = "markers+text" if use_label else "markers"
    scatter = go.Scatter(
        x=z_xs,
        y=z_ys,
        text=labels,
        textposition="bottom center",
        mode=mode,
    )
    fig.add_trace(scatter)
    fig.update_layout(
        {
            "title": {"text": "Latent space, z"},
            "xaxis": {"range": [-3, 3]},
            "yaxis": {"range": [-3, 3]},
        }
    )
    return fig


def spike_sequence_as_bars(actual, pred, bin_ms: float):
    """A figure showing actual vs. predicted spike counts."""
    if len(actual) != len(pred):
        raise ValueError(
            "actual and pred must be the same length. Got"
            f"(actual: {len(actual)}, pred: {len(pred)})."
        )
    actual = actual.astype(np.float32)
    pred = pred.astype(np.float32)
    num_bins = len(actual)
    bins_per_row = 60
    num_rows = max(1, num_bins // bins_per_row)
    fig = plotly.subplots.make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        x_title=f"time bins ({bin_ms:.1f} ms)",
        y_title="spike count",
    )
    bar_width = 40
    bar_gap = 0.3
    center_bar_width = 0.7 * bar_width
    center_bar_offset = 0.5 * (
        bar_gap * bar_width + bar_width - center_bar_width
    )
    for i in range(num_rows):
        xs = np.arange(bins_per_row) * bin_ms
        start, end = np.array([i, i + 1]) * bins_per_row
        min_vals = np.minimum(actual[start:end], pred[start:end])
        err = pred[start:end] - actual[start:end]
        bar_actual = go.Bar(
            x=xs,
            y=actual[start:end],
            name="base",
            marker={
                "color": "rgba(140, 140, 140, 1)",
                "line": {"width": 0.6, "color": "rgb(10,10,10)"},
            },
            legendgroup="actual",
            showlegend=i == 0,
            width=bar_width,
            offset=0,
            # width=0.9
        )
        bar_pred = go.Bar(
            x=xs,
            y=pred[start:end],
            name="pred",
            # marker={"color": "rgb(220, 100, 80)", "line": {"width": 1}},
            marker={
                "color": "rgba(250, 170, 170, 1)",
                "line": {"width": 0.8, "color": "rgb(25,17,17)"},
            },
            # width=1,
            legendgroup="pred",
            showlegend=i == 0,
            width=center_bar_width,
            offset=center_bar_offset,
        )
        fig.add_trace(bar_actual, row=i + 1, col=1)
        fig.add_trace(bar_pred, row=i + 1, col=1)

    fig.update_layout(default_fig_layout())
    fig.update_layout(
        # barmode="stack",
        bargap=bar_gap,
        barmode="overlay",
    )
    fig.update_layout(
        {
            "title": {
                "text": create_title(
                    "Actual spikes (red) vs. predicted spikes (blue) in "
                    f"{bin_ms:.1f} ms bins"
                )
            },
            "height": max(200, 75 * num_rows),
        }
    )
    return fig


def spike_sequence_as_lines(
    actual, pred, bin_ms: float, smooth: Optional[int] = None
):
    """A figure showing actual vs. predicted spike counts."""
    if len(actual) != len(pred):
        raise ValueError(
            "actual and pred must be the same length. Got"
            f"(actual: {len(actual)}, pred: {len(pred)})."
        )
    actual = actual.astype(np.float32)
    pred = pred.astype(np.float32)
    num_bins = len(actual)
    bins_per_row = 100
    num_rows = max(1, num_bins // bins_per_row)
    fig = plotly.subplots.make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        x_title=f"time bins ({bin_ms:.1f} ms)",
        y_title="spike count",
    )
    for i in range(num_rows):
        xs = np.arange(bins_per_row) * bin_ms
        start, end = np.array([i, i + 1]) * bins_per_row
        dots_actual = go.Scatter(
            x=xs,
            y=actual[start:end],
            name="actual",
            line_color="rgb(10, 10, 10)",
            mode="markers",
            marker_size=4,
            legendgroup="actual",
            showlegend=i == 0,
        )
        dots_pred = go.Scatter(
            x=xs,
            y=pred[start:end],
            name="pred",
            line_color="rgb(250, 170, 170)",
            mode="markers",
            marker_size=4,
            legendgroup="pred",
            showlegend=i == 0,
        )
        # smooth_fn = scipy.ndimage.uniform_filter1d
        identity = lambda x, sigma, mode: x
        if smooth:
            smooth_fn = scipy.ndimage.gaussian_filter1d
        else:
            smooth_fn = identity

        smooth_actual = go.Scatter(
            x=xs,
            y=smooth_fn(actual[start:end], smooth, mode="reflect"),
            name="actual (window smooth)",
            mode=None,
            legendgroup="actual",
            showlegend=i == 0,
            fill=None,
            # fillcolor="rgb(255, 200, 200)")
            line={"color": "rgb(50, 50, 50)", "width": 2},
        )
        smooth_pred = go.Scatter(
            x=xs,
            y=smooth_fn(pred[start:end], smooth, mode="reflect"),
            name="pred (window smooth)",
            mode=None,  # no line
            legendgroup="pred",
            showlegend=i == 0,
            # fill="tozeroy",
            fill="tonexty",
            line_color="rgb(250, 170, 170)",
            fillcolor="rgb(170, 50, 50)",
            # fillcolor="rgba(0, 154, 255, 0.2)"
        )
        fig.add_trace(smooth_actual, row=i + 1, col=1)
        fig.add_trace(smooth_pred, row=i + 1, col=1)
        if smooth is None:
            fig.add_trace(dots_actual, row=i + 1, col=1)
            fig.add_trace(dots_pred, row=i + 1, col=1)

    fig.update_layout(default_fig_layout())
    fig.update_layout(
        {
            "title": {
                "text": create_title(
                    "Actual spikes (red) vs. predicted spikes (blue) in "
                    f"{bin_ms:.1f} ms bins"
                )
            },
            "height": max(200, 75 * num_rows),
        }
    )
    return fig
