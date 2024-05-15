"""
A lot of experimentation ends up here in models.py.
"""

import enum
from functools import partial
from itertools import pairwise
import logging
import math
import pathlib
from typing import Dict, Optional, Tuple, Union
import einops
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
import retinapy
import retinapy.mea as mea
import retinapy.nn

_logger = logging.getLogger(__name__)


def load_model(
    model, checkpoint_path: Union[str, pathlib.Path], map_location=None
):
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file/folder ({checkpoint_path}) not found."
        )
    if checkpoint_path.is_dir():
        checkpoint_path = list(checkpoint_path.glob("*.pth"))[-1]

    _logger.info(f"Loading model from ({checkpoint_path}).")
    checkpoint_state = torch.load(checkpoint_path, map_location)
    model_state = checkpoint_state["model"]
    model.load_state_dict(model_state)


def load_model_and_optimizer(
    model,
    checkpoint_path: Union[str, pathlib.Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file/folder ({checkpoint_path}) not found."
        )
    if checkpoint_path.is_dir():
        checkpoint_path = list(checkpoint_path.glob("*.pth"))[-1]

    _logger.info(f"Loading model from ({checkpoint_path}).")
    checkpoint_state = torch.load(checkpoint_path)
    model_state = checkpoint_state["model"]
    model.load_state_dict(model_state)
    if optimizer:
        optimizer.load_state_dict(checkpoint_state["optimizer"])
    if scheduler:
        scheduler.load_state_dict(checkpoint_state["scheduler"])


def save_model(model, path: pathlib.Path, optimizer=None, scheduler=None):
    _logger.info(f"Saving model to ({path})")
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    state = {
        "model": model.state_dict(),
    }
    if optimizer:
        state.update({"optimizer": optimizer.state_dict()})
    if scheduler:
        state.update({"scheduler": scheduler.state_dict()})
    torch.save(state, path)


def dist_loss(prediction, target):
    """
    MSE loss reduced along time dimension but not batch dimension.

    It is expected that this loss is just one part of a loss term, so reducing
    over the batch dimension is not done. And it's not even given as a option
    as it's too easy to mistake "mean" reduction to be over a single dimension,
    when it is a full reduction over all dimensions. This is a tricky aspect of
    PyTorch's MSE loss.
    """
    # Scale to get roughly in the ballpark of 0.1 to 10.
    DIST_LOSS_SCALE = 3
    loss = DIST_LOSS_SCALE * F.mse_loss(prediction, target, reduction="none")
    assert (
        len(prediction.shape) == 2
    ), f"Batch and time dim expected; got ({prediction.shape})."
    time_ave = torch.mean(loss, dim=1)
    batch_sum = torch.sum(time_ave)
    return batch_sum


def dist_loss_l1(prediction, target):
    """
    MSE loss reduced along time dimension but not batch dimension.

    It is expected that this loss is just one part of a loss term, so reducing
    over the batch dimension is not done. And it's not even given as a option
    as it's too easy to mistake "mean" reduction to be over a single dimension,
    when it is a full reduction over all dimensions. This is a tricky aspect of
    PyTorch's MSE loss.
    """
    # Scale to get roughly in the ballpark of 0.1 to 10.
    DIST_LOSS_SCALE = 1.7
    loss = DIST_LOSS_SCALE * F.l1_loss(prediction, target, reduction="none")
    assert (
        len(prediction.shape) == 2
    ), f"Batch and time dim expected; got ({prediction.shape})."
    time_ave = torch.mean(loss, dim=1)
    batch_sum = torch.sum(time_ave)
    return batch_sum


class TensorTag(torch.nn.Module):
    """Use this module to log tensors.

    The trainer can add hooks to this module to log the tensors.

    It's more general than logging, but so far only used for logging.

    The forward's label argument will be used as the key in the log. If there
    is no label, then the module's label will be used.
    """

    def __init__(self, label: Optional[str] = None):
        super().__init__()
        self.label = label

    def forward(self, x, label: Optional[str] = None):
        return x


class MonitoringBase(nn.Module):
    """Example model that enables monitoring."""

    def modules_to_inspect(self) -> Dict[str, nn.Module]:
        """If this method is present, and train() is run with activations or
        weight monitoring enabled, then the modules in this list will have
        hooks attached for monitoring purposed."""
        return {}


class InitConv(nn.Module):
    def __init__(self, in_nch, out_nch, kernel_len):
        super().__init__()
        self.leak = 0.1
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=out_nch,
                kernel_size=kernel_len,
                stride=2,
                padding=(kernel_len - 1) // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(out_nch),
            nn.LeakyReLU(self.leak, True),
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(
            self.layer[0].weight, nonlinearity="leaky_relu", a=self.leak
        )

    def forward(self, x):
        return self.layer(x)


class UpResBlock(nn.Module):
    def __init__(self, in_nch, out_nch, kernel_size, dropout=0.0):
        super().__init__()
        self.convt = nn.ConvTranspose1d(in_nch, in_nch, 2, stride=2)
        self.conv = retinapy.nn.ResBlock1d(
            in_nch,
            in_nch,
            out_nch,
            kernel_size,
            downsample=False,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.convt(x)
        x = self.conv(x)
        return x


class UpResBlock2(nn.Module):
    def __init__(self, in_nch, out_nch, kernel_size, dropout=0.0):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_nch, in_nch, 2, stride=2)
        self.conv = retinapy.nn.ConvNextv2(
            in_nch,
            out_nch,
            kernel_size,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class UpBlock3(nn.Module):
    def __init__(self, in_nch, out_nch, kernel_size, dropout=0.0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear")
        self.conv = retinapy.nn.ConvNextv2(
            in_nch,
            out_nch,
            kernel_size,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class SharedBase(nn.Module):
    NUM_IN_CHANNELS = 5
    NUM_L1_CHANNELS = 64
    NUM_DOWN_CHANNELS = 64
    NUM_MID_CHANNELS = 64
    L1_KERNEL_SIZE = 15
    DOWN_KERNEL_SIZE = 5
    MID_KERNEL_SIZE = 3
    DROPOUT_RATE = 0.2
    EXPANSION = 2

    def __init__(self, in_len: int, n_down: int, n_mid: int):
        super().__init__()

        self.layer1 = InitConv(
            in_nch=self.NUM_IN_CHANNELS,
            out_nch=self.NUM_L1_CHANNELS,
            kernel_len=self.L1_KERNEL_SIZE,
        )
        self.pos_embed = nn.Parameter(
            retinapy.nn.get_sinusoidal_embeddings(
                in_len // 2, self.NUM_L1_CHANNELS
            )
        )

        def down_fn(i_ch, o_ch):
            res = nn.Sequential(
                retinapy.nn.LayerNorm(
                    i_ch, eps=1e-6, data_format="channels_first"
                ),
                nn.Conv1d(i_ch, o_ch, kernel_size=3, stride=2, padding=1),
                retinapy.nn.ConvNextv2(
                    i_ch,
                    o_ch,
                    kernel_size=self.DOWN_KERNEL_SIZE,
                    dropout=self.DROPOUT_RATE,
                    expansion=self.EXPANSION,
                ),
            )
            return res

        down_nch = [self.NUM_L1_CHANNELS] + [self.NUM_DOWN_CHANNELS] * (
            n_down - 1
        )

        self.down = nn.Sequential(
            *[down_fn(i_ch, o_ch) for i_ch, o_ch in pairwise(down_nch)]
        )

        mid_nch = [down_nch[-1]] + [self.NUM_MID_CHANNELS] * n_mid
        mid_fn = partial(
            retinapy.nn.ConvNextv2,
            kernel_size=self.MID_KERNEL_SIZE,
            dropout=self.DROPOUT_RATE,
            expansion=self.EXPANSION,
        )
        self.mid = nn.Sequential(
            *[mid_fn(i_ch, o_ch) for i_ch, o_ch in pairwise(mid_nch)]
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x + self.pos_embed
        x = self.down(x)
        x = self.mid(x)
        return x


class PoissonNet2(nn.Module):
    """PoissonNet, but using the shared base."""

    def __init__(self, in_len: int, n_down: int, n_mid: int):
        super().__init__()
        self.linear_in_len = 1 + (in_len - 1) // 2**n_down
        assert self.linear_in_len == 8, "For now, assuming 8"

        self.shared_base = SharedBase(in_len, n_down, n_mid)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.shared_base.NUM_MID_CHANNELS
                * self.linear_in_len,
                out_features=1,
            ),
            nn.Softplus(),
            einops.layers.torch.Rearrange("b 1 -> b"),
        )

    def forward(self, x):
        x = self.shared_base(x)
        x = self.head(x)
        return x


class CnnUNet3(nn.Module):
    """
    CnnUNet2, but using the shared base.
    """

    NUM_UP_CHANNELS = 16
    UP_KERNEL_SIZE = 5

    def __init__(
        self,
        in_len: int,
        n_down: int,
        n_mid: int,
        n_up: int,
        prefix_len: int,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        if n_down < 1:
            raise ValueError("n_down must be at least 1")
        # This model has a normalization.
        self.register_buffer("input_mean", torch.zeros(size=(5,)))
        self.register_buffer("input_sd", torch.ones(size=(5,)))
        self.register_buffer("output_mean", torch.tensor(1.0))
        # Output has fixed scale to maintain dynamic range ~[0,5] even when
        # cells have very low spike rates.
        self.output_scale = 2.0

        self.shared_base = SharedBase(in_len, n_down, n_mid)

        up_fn = partial(
            UpBlock3,
            kernel_size=self.UP_KERNEL_SIZE,
            # Following the paper Simple Diffusion, dropout is
            # not kept on the upsampling half of the network.
            dropout=0,
        )
        up_nch = [self.shared_base.NUM_MID_CHANNELS] + [
            self.NUM_UP_CHANNELS
        ] * n_up
        self.up = nn.Sequential(
            *[up_fn(i_ch, o_ch) for i_ch, o_ch in pairwise(up_nch)]
        )

        self.past_future_embed = nn.Parameter(
            torch.normal(mean=0, std=0.25, size=(2, self.NUM_UP_CHANNELS, 1))
        )

        self.head = nn.Conv1d(
            in_channels=self.NUM_UP_CHANNELS,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )
        self.init_weights()

    def init_weights(self):
        # Last conv doesn't have relu activation, so use Xavier.
        nn.init.xavier_uniform_(self.head.weight)

    def set_input_mean_sd(self, m: torch.Tensor, sd: torch.Tensor):
        if m.shape != (5,):
            raise ValueError(
                "Input mean must have shape (5,). " f"Got ({m.shape})."
            )
        if sd.shape != (5,):
            raise ValueError(
                "Input sd must have shape (5,). " f"Got ({sd.shape})."
            )
        self.input_mean.copy_(m)
        self.input_sd.copy_(sd)

    def set_output_mean(self, m: float):
        """Set the value by which the log output dist is added to."""
        self.output_mean.fill_(m)

    def normalize_input(self, x):
        return (x - self.input_mean[None, :, None]) / self.input_sd[
            None, :, None
        ]

    def denormalize_output(self, x):
        return (x * self.output_scale) + self.output_mean

    def past_future_mult(self, x):
        """Modulate input to last layer based on past vs. future mask.

        This isn't a crutial component, but it seems reasonable to make this
        past-future distinction of the output distance array available to the
        model explicitly."""
        x = torch.cat(
            [
                x[:, :, 0 : self.prefix_len] * (self.past_future_embed[0] + 1),
                x[:, :, self.prefix_len :] * (self.past_future_embed[1] + 1),
            ],
            dim=-1,
        )
        return x

    def forward(self, x):
        x = self.normalize_input(x)
        x = self.shared_base(x)
        x = self.up(x)
        x = self.past_future_mult(x)
        x = self.head(x)
        x = self.denormalize_output(x)
        x = einops.rearrange(x, "b 1 w -> b w")
        return x


class FcModel(nn.Module):
    def __init__(self, in_n, out_n, clamp_max):
        super(FcModel, self).__init__()
        self.clamp_max = clamp_max
        self.fc_body = retinapy.nn.FcBlock(
            hidden_ch=1024 * 2,
            num_hidden_layers=5,
            in_features=in_n,
            out_features=out_n,
            outermost_linear=True,
        )
        self.act = torch.nn.Softplus()

    def forward(self, x):
        x = self.act(self.fc_body(x))
        x = torch.clamp(x, min=None, max=self.clamp_max)
        return x


class MiniVAE(nn.Module):
    def __init__(self, num_clusters, z_n=2):
        super(MiniVAE, self).__init__()
        self.embed_mu = nn.Embedding(num_clusters, z_n)
        self.embed_logvar = nn.Embedding(num_clusters, z_n)

    def encode(self, x):
        x = x.long()
        mu = self.embed_mu(x)
        logvar = self.embed_logvar(x)
        return mu, logvar

    def sampling(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new_empty(std.size()).normal_()
        return eps.mul_(std).add_(mu)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        if self.training:
            z = self.sampling(z_mu, z_logvar)
        else:
            z = z_mu
        return z, z_mu, z_logvar


class VAE(nn.Module):
    def __init__(self, num_clusters, z_n=2, h1_n=16, h2_n=32, out_n=5):
        super(VAE, self).__init__()
        self.embed_mu = nn.Embedding(num_clusters, z_n)
        self.embed_logvar = nn.Embedding(num_clusters, z_n)
        self.fc1 = nn.Linear(z_n, h1_n)
        self.fc2 = nn.Linear(h1_n, h2_n)
        self.fc3 = nn.Linear(h2_n + z_n, out_n)

    def encode(self, x):
        x = x.long()
        mu = self.embed_mu(x)
        logvar = self.embed_logvar(x)
        return mu, logvar

    def sampling(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new_empty(std.size()).normal_()
        return eps.mul_(std).add_(mu)

    def decode(self, z):
        """
        Trying out skip connections, due to positive results reported here:
             https://adjidieng.github.io/Papers/skipvae.pdf
        """
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))
        h2_skip = torch.cat([h2, z], dim=1)
        return self.fc3(h2_skip)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        if self.training:
            z = self.sampling(z_mu, z_logvar)
        else:
            z = z_mu
        return self.decode(z), z_mu, z_logvar


class QueryDecoder(nn.Module):
    def __init__(self, n_z, num_queries, key_len, n_hidden1=30, n_hidden2=30):
        super().__init__()
        self.out_shape = (num_queries, key_len)
        self.fc1 = nn.Linear(n_z, n_hidden1)
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, num_queries * key_len)

    def forward(self, z):
        x = torch.tanh(self.fc1(z))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, *self.out_shape)
        return x


class MultiClusterModel2(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(self, in_len, out_len, num_downsample, num_clusters):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.num_input_channels = self.LED_CHANNELS + self.NUM_CLUSTERS
        # Linear
        self.linear_in_len = 1 + (in_len - 1) // (2**num_downsample)

        self.num_channels = [20, 100, 100, 100]
        self.downsample = [False, False, True, False]
        self.num_repeats = [None, 2, num_downsample - 1, 3]
        self.mid_kernel_size = 7
        self.kernel_sizes = [
            51,
            self.mid_kernel_size,
            self.mid_kernel_size,
            self.mid_kernel_size,
        ]
        self.expansion = 1
        # VAE
        self.z_dim = 2
        self.num_clusters = num_clusters
        self.num_embed = num_clusters
        self.n_h1 = 20
        self.n_h2 = 20
        # HyperResnet
        warehouse_size = 1500
        self.key_len = 16  # 32 = ~ sqrt(1000)

        # Huge memory store.
        self.warehouse = retinapy.nn.Conv1dWarehouse(
            max_in_channels=1,  # depth-wise conv.
            warehouse_size=warehouse_size,
            kernel_size=self.mid_kernel_size,
            key_len=self.key_len,
        )

        # Traditional conv layers.
        self.layer0 = nn.Sequential(
            nn.Conv1d(
                self.num_input_channels,
                self.num_channels[0],
                kernel_size=self.kernel_sizes[0],
                stride=2,
                padding=self.kernel_sizes[0] // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                self.num_channels[0],
                self.num_channels[0],
                kernel_size=self.kernel_sizes[0],
                stride=1,
                padding=self.kernel_sizes[0] // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(self.num_channels[0]),
            nn.LeakyReLU(0.2, True),
        )

        mid = []
        total_num_channels = 0
        for layer_idx in range(1, len(self.num_channels)):
            blocks = []
            for i in range(self.num_repeats[layer_idx]):
                num_in = (
                    self.num_channels[layer_idx - 1]
                    if (i == 0)
                    else self.num_channels[layer_idx]
                )
                res_block_F = retinapy.nn.ResBlock1d_F(
                    num_in,
                    self.num_channels[layer_idx] * self.expansion,
                    self.num_channels[layer_idx],
                    kernel_size=self.kernel_sizes[layer_idx],
                    downsample=self.downsample[layer_idx],
                )
                blocks.append(res_block_F)
                total_num_channels += res_block_F.num_channels()
            mid.append(nn.ModuleList(blocks))
        self.mid_layers = nn.ModuleList(mid)

        # Back to traditional conv layers.
        self.layer4 = nn.Conv1d(
            in_channels=self.num_channels[-1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        linear_in_len = 1 + (in_len - 1) // (2**num_downsample)
        self.linear = nn.Linear(
            in_features=linear_in_len,
            out_features=self.out_len,
        )
        # VAE
        self.vae = MiniVAE(num_clusters, z_n=self.z_dim)
        # VAE decode to attention query
        # 1 query per mid channel.
        self.query_decoder = QueryDecoder(
            self.z_dim, total_num_channels, self.key_len
        )

    def encode_vae(self, cluster_id):
        id_ = cluster_id
        z, z_mu, z_logvar = self.vae(id_)
        return z, z_mu, z_logvar

    def forward(self, snippet, cluster_id):
        # Layer 0
        x = self.layer0(snippet)
        # VAE
        z, z_mu, z_logvar = self.encode_vae(cluster_id)
        # Queries
        queries = self.query_decoder(z)
        weights_W, weights_b = self.warehouse(queries)
        idx_start = 0
        for module_list in self.mid_layers:
            for layer in module_list:
                x, idx_start = layer.forward(x, weights_W, weights_b, idx_start)
        x = self.layer4(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        return x, z_mu, z_logvar


class CatMultiClusterModel(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(
        self,
        in_len,
        out_len,
        num_downsample,
        num_clusters,
        z_dim=2,
    ):
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        # led_channels, mean(led_channels), num_clusters, pos_encoding
        self.num_input_channels = self.LED_CHANNELS * 2 + self.NUM_CLUSTERS + 1
        # Linear
        self.linear_in_len = 1 + (in_len - 1) // (2**num_downsample)
        # L1
        self.l0_num_channels = 50
        # CNN parameters
        self.num_l1_blocks = num_downsample - 1
        self.num_l2_blocks = 3
        self.expansion = 2
        self.l1_num_channels = 100
        self.l2_num_channels = 200
        kernel_size = 21
        mid_kernel_size = 7
        # VAE
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        self.n_vae_out = 30

        self.layer0 = nn.Sequential(
            nn.Conv1d(
                self.num_input_channels,
                self.l0_num_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                self.l0_num_channels,
                self.l0_num_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(self.l0_num_channels),
            nn.LeakyReLU(0.2, True),
        )

        l1_blocks = []
        l1_num_in = self.l0_num_channels + self.n_vae_out
        for i in range(self.num_l1_blocks):
            num_in = self.l1_num_channels if i else l1_num_in
            l1_blocks.append(
                retinapy.nn.ResBlock1d(
                    num_in,
                    self.l1_num_channels * self.expansion,
                    self.l1_num_channels,
                    kernel_size=mid_kernel_size,
                    downsample=True,
                ),
            )
        self.layer1 = nn.Sequential(*l1_blocks)
        l2_blocks = []
        for i in range(self.num_l1_blocks):
            num_in = self.l2_num_channels if i else self.l1_num_channels
            l2_blocks.append(
                retinapy.nn.ResBlock1d(
                    num_in,
                    self.l2_num_channels * self.expansion,
                    self.l2_num_channels,
                    kernel_size=mid_kernel_size,
                    downsample=False,
                ),
            )
        self.layer2 = nn.Sequential(*l2_blocks)

        self.layer3 = nn.Conv1d(
            in_channels=self.l2_num_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        linear_in_len = 1 + (in_len - 1) // (2**num_downsample)
        self.linear = nn.Linear(
            in_features=linear_in_len,
            out_features=self.out_len,
        )
        # VAE
        self.vae = VAE(num_clusters, z_n=self.z_dim, out_n=self.n_vae_out)

    def encode_vae(self, cluster_id):
        id_ = cluster_id
        z_decode, z_mu, z_logvar = self.vae(id_)
        return z_decode, z_mu, z_logvar

    # def cat_downsample(self, x):
    #     ds = torchaudio.functional.lowpass_biquad(
    #             x[:,0:-1], sample_rate=992, cutoff_freq=10, Q=0.707
    #     )
    #     x = torch.cat([x, ds], dim=1)
    #     return x

    def cat_mean(self, snippet):
        m = (
            snippet[:, 0 : mea.NUM_STIMULUS_LEDS]
            .mean(dim=2, keepdim=True)
            .expand(-1, -1, snippet.shape[-1])
        )
        x = torch.cat([snippet, m], dim=1)
        return x

    def cat_z(self, x, z):
        z = z.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x = torch.cat([x, z], dim=1)
        return x

    def forward(self, snippet, cluster_id):
        # VAE
        x = self.cat_mean(snippet)
        x = self.layer0(x)
        z_decode, z_mu, z_logvar = self.encode_vae(cluster_id)
        x = self.cat_z(x, z_decode)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        return x, z_mu, z_logvar


class MultiClusterModel(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(
        self,
        in_len,
        out_len,
        num_downsample,
        num_clusters,
        z_dim=2,
    ):
        super(MultiClusterModel, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        # led_channels, mean(led_channels), num_clusters, pos_encoding
        self.num_input_channels = self.LED_CHANNELS * 2 + self.NUM_CLUSTERS + 1
        # Linear
        self.linear_in_len = 1 + (in_len - 1) // (2**num_downsample)
        # L1
        self.l0_num_channels = 50
        self.l1_num_channels = 50
        # CNN parameters
        self.num_l1_blocks = 1
        self.num_l2_blocks = num_downsample - 1
        self.num_l3_blocks = 1
        self.expansion = 1
        self.l2_num_channels = 100
        self.l3_num_channels = 200
        kernel_size = 21
        mid_kernel_size = 7
        # VAE
        self.z_dim = z_dim
        self.num_clusters = num_clusters
        # HyperResnet
        warehouse_factor = 8
        self.hyper_hidden1 = 16
        self.hyper_hidden2 = 16

        self.layer0 = nn.Sequential(
            nn.Conv1d(
                self.num_input_channels,
                self.l0_num_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                self.l0_num_channels,
                self.l0_num_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(self.l0_num_channels),
            nn.LeakyReLU(0.2, True),
        )
        l1_blocks = []
        for i in range(self.num_l1_blocks):
            num_in = self.l1_num_channels if i else self.l0_num_channels
            hyper_res_block = retinapy.nn.HyperResBlock1d(
                num_in,
                self.l1_num_channels * self.expansion,
                self.l1_num_channels,
                warehouse_factor=warehouse_factor,
                kernel_size=mid_kernel_size,
                downsample=False,
            )
            hyper_decoder = retinapy.nn.HyperResDecoder(
                hyper_res_block,
                in_n=self.z_dim,
                hidden1_n=self.hyper_hidden1,
                hidden2_n=self.hyper_hidden2,
            )
            l1_blocks.append(hyper_decoder)
        self.layer1 = nn.ModuleList(l1_blocks)

        l2_blocks = []
        for i in range(self.num_l2_blocks):
            num_in = self.l2_num_channels if i else self.l1_num_channels
            l2_blocks.append(
                retinapy.nn.ResBlock1d(
                    num_in,
                    self.l2_num_channels * self.expansion,
                    self.l2_num_channels,
                    kernel_size=mid_kernel_size,
                    downsample=True,
                ),
            )
        self.layer2 = nn.Sequential(*l2_blocks)

        l3_blocks = []
        for i in range(self.num_l3_blocks):
            num_in = self.l3_num_channels if i else self.l2_num_channels
            hyper_res_block = retinapy.nn.HyperResBlock1d(
                num_in,
                self.l3_num_channels * self.expansion,
                self.l3_num_channels,
                warehouse_factor=warehouse_factor,
                kernel_size=mid_kernel_size,
                downsample=False,
            )
            hyper_decoder = retinapy.nn.HyperResDecoder(
                hyper_res_block,
                in_n=self.z_dim,
                hidden1_n=self.hyper_hidden1,
                hidden2_n=self.hyper_hidden2,
            )
            l3_blocks.append(hyper_decoder)
        self.layer3 = nn.ModuleList(l3_blocks)
        self.layer4 = nn.Conv1d(
            in_channels=self.l3_num_channels,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        linear_in_len = 1 + (in_len - 1) // (2**num_downsample)
        self.linear = nn.Linear(
            in_features=linear_in_len,
            out_features=self.out_len,
        )
        # VAE
        self.vae = MiniVAE(num_clusters, z_n=self.z_dim)

    def encode_vae(self, cluster_id):
        id_ = self.num_clusters + cluster_id
        z, z_mu, z_logvar = self.vae(id_)
        return z, z_mu, z_logvar

    # def cat_downsample(self, x):
    #     ds = torchaudio.functional.lowpass_biquad(
    #             x[:,0:-1], sample_rate=992, cutoff_freq=10, Q=0.707
    #     )
    #     x = torch.cat([x, ds], dim=1)
    #     return x

    def cat_mean(self, snippet):
        m = (
            snippet[:, 0:-1]
            .mean(dim=2, keepdim=True)
            .expand(-1, -1, snippet.shape[-1])
        )
        x = torch.cat([snippet, m], dim=1)
        return x

    def forward(self, snippet, cluster_id):
        x = self.cat_mean(snippet)
        x = self.layer0(x)
        # VAE
        z, z_mu, z_logvar = self.encode_vae(cluster_id)
        # Hyper
        for l in self.layer1:
            x = l(x, z)
        x = self.layer2(x)
        for l in self.layer3:
            x = l(x, z)
        x = self.layer4(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        return x, z_mu, z_logvar


class ClusteringTransformer(nn.Module):
    """TransformerModel, but without using any spike data.

    The point being that the VAE is forced to learn a clustering that
    cannot offload any information to processing of the the spike data,
    which is otherwise sent in as a separate input to the model.
    """

    def __init__(
        self,
        in_len,
        out_len,
        stim_downsample,
        num_clusters,
        z_dim=2,
        num_heads=8,
        head_dim=64,
        num_tlayers=6,
    ):
        super().__init__()
        self.in_len = in_len
        self.num_clusters = num_clusters
        self.z_dim = z_dim
        # Stimulus CNN
        k0_size = 21
        k1_size = 7
        expansion = 1
        self.l0a_num_channels = 10
        self.l0b_num_channels = 20
        self.l1_num_channels = 100
        self.cnn = nn.Sequential(
            nn.Conv1d(
                mea.NUM_STIMULUS_LEDS * 2,  # stimulus + mean(stimulus)
                self.l0a_num_channels,
                kernel_size=k0_size,
                stride=2,
                padding=(k0_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                self.l0a_num_channels,
                self.l0b_num_channels,
                kernel_size=k0_size,
                stride=1,
                padding=(k0_size - 1) // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(self.l0b_num_channels),
            nn.LeakyReLU(0.2, True),
            *[
                retinapy.nn.ResBlock1d(
                    self.l1_num_channels if i else self.l0b_num_channels,
                    self.l1_num_channels * expansion,
                    self.l1_num_channels,
                    kernel_size=k1_size,
                    downsample=True,
                )
                for i in range(stim_downsample - 1)
            ],
        )

        self.embed_dim = 128
        # VAE
        self.vae = VAE(
            self.num_clusters,
            z_n=self.z_dim,
            h1_n=32,
            h2_n=32,
            out_n=self.embed_dim,
        )
        # Transformer
        self.stim_embed = nn.Conv1d(
            self.l1_num_channels, self.embed_dim, kernel_size=1
        )
        # Normally initialized nn.Parameter
        # 1092 = 992 + 100
        in_stim_len = self.in_len + out_len
        enc_stim_len = 1 + (in_stim_len - 1) // (2**stim_downsample)
        enc_len = enc_stim_len + 1  # 1 for VAE encoding.
        self.pos_embed = nn.Parameter(torch.randn(enc_len, self.embed_dim))

        mlp_expansion = 3
        self.transformer = retinapy.nn.Transformer(
            self.embed_dim,
            num_layers=num_tlayers,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=self.embed_dim * mlp_expansion,
        )
        self.decode = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            einops.layers.torch.Rearrange("b c 1 -> b c"),
            nn.Linear(enc_len, out_len),
        )

    def cat_mean(self, stim):
        m = (
            stim[:, 0 : mea.NUM_STIMULUS_LEDS]
            .mean(dim=2, keepdim=True)
            .expand(-1, -1, stim.shape[-1])
        )
        x = torch.cat([stim, m], dim=1)
        return x

    def encode_stimulus(self, stim):
        x = self.cat_mean(stim)
        x = self.cnn(x)
        x = self.stim_embed(x)
        x = einops.rearrange(x, "b c l -> b l c")
        return x

    def encode_vae(self, cluster_id):
        # For now, we just use the cluster_id directly.
        id_ = cluster_id
        z_decode, z_mu, z_logvar = self.vae(id_)
        return z_decode, z_mu, z_logvar

    def forward(self, snippet, cluster_id):
        z_decode, z_mu, z_logvar = self.encode_vae(cluster_id)
        x_stim = self.encode_stimulus(snippet[:, 0:-1])
        z = einops.rearrange(z_decode, "b c -> b () c")
        x = torch.cat([x_stim, z], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.decode(x)
        return x, z_mu, z_logvar


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_len,
        out_len,
        stim_downsample,
        num_clusters,
        z_dim=2,
        num_heads=8,
        head_dim=64,
        num_tlayers=6,
        spike_patch_len=8,
    ):
        super().__init__()
        self.in_len = in_len
        self.num_clusters = num_clusters
        self.z_dim = z_dim
        # Stimulus CNN
        k0_size = 21
        k1_size = 7
        expansion = 1
        self.l0a_num_channels = 10
        self.l0b_num_channels = 20
        self.l1_num_channels = 50
        self.stim_enc_num_channels = 3
        self.cnn = nn.Sequential(
            nn.Conv1d(
                mea.NUM_STIMULUS_LEDS,  # stimulus + mean(stimulus)
                self.l0a_num_channels,
                kernel_size=k0_size,
                stride=2,
                padding=(k0_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                self.l0a_num_channels,
                self.l0b_num_channels,
                kernel_size=k0_size,
                stride=1,
                padding=(k0_size - 1) // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(self.l0b_num_channels),
            nn.LeakyReLU(0.2, True),
            *[
                retinapy.nn.ResBlock1d(
                    self.l1_num_channels if i else self.l0b_num_channels,
                    self.l1_num_channels * expansion,
                    self.l1_num_channels,
                    kernel_size=k1_size,
                    downsample=True,
                )
                for i in range(stim_downsample - 1)
            ],
            retinapy.nn.ResBlock1d(
                self.l1_num_channels,
                self.l1_num_channels * expansion,
                self.stim_enc_num_channels,
                kernel_size=k1_size,
                downsample=False,
            ),
            # torch.nn.Dropout(p=0.2),
        )

        self.embed_dim = 64
        # VAE
        self.vae = VAE(
            self.num_clusters,
            z_n=self.z_dim,
            h1_n=32,
            h2_n=32,
            out_n=self.embed_dim,
        )
        # Transformer
        self.spike_patch_len = spike_patch_len
        self.stim_embed = nn.Conv1d(
            self.stim_enc_num_channels, self.embed_dim, kernel_size=1
        )
        # Normally initialized nn.Parameter
        # 1092 = 992 + 100
        in_stim_len = self.in_len + out_len
        enc_stim_len = 1 + (in_stim_len - 1) // (2**stim_downsample)
        enc_spikes_len = math.ceil(self.in_len // self.spike_patch_len)
        self.spike_pad = enc_spikes_len * self.spike_patch_len - self.in_len
        enc_len = enc_stim_len + enc_spikes_len + 1  # 1 for VAE encoding.
        self.pos_embed = nn.Parameter(torch.randn(enc_len, self.embed_dim))

        self.spikes_embed = nn.Linear(self.spike_patch_len, self.embed_dim)
        mlp_expansion = 3
        self.transformer = retinapy.nn.Transformer(
            self.embed_dim,
            num_layers=num_tlayers,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=self.embed_dim * mlp_expansion,
        )
        self.decode = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            einops.layers.torch.Rearrange("b c 1 -> b c"),
            nn.Linear(enc_len, out_len),
        )

    def cat_mean(self, stim):
        m = (
            stim[:, 0 : mea.NUM_STIMULUS_LEDS]
            .mean(dim=2, keepdim=True)
            .expand(-1, -1, stim.shape[-1])
        )
        x = torch.cat([stim, m], dim=1)
        return x

    def encode_stimulus(self, stim):
        # TODO: might this give away the specific time?
        # x = self.cat_mean(stim)
        x = self.cnn(stim)
        x = self.stim_embed(x)
        x = einops.rearrange(x, "b c l -> b l c")
        return x

    def encode_spikes(self, spikes):
        # Pad the spikes to be a multiple of the patch length.
        # Pad on the left, as the left has less of an effect on the output.
        x = F.pad(spikes, (self.spike_pad, 0))
        x = einops.rearrange(x, "b (t c) -> b t c", c=self.spike_patch_len)
        x = self.spikes_embed(x)
        return x

    def encode_vae(self, cluster_id):
        id_ = cluster_id
        z_decode, z_mu, z_logvar = self.vae(id_)
        return z_decode, z_mu, z_logvar

    def forward(self, snippet, cluster_id):
        z_decode, z_mu, z_logvar = self.encode_vae(cluster_id)
        x_stim = self.encode_stimulus(snippet[:, 0:-1])
        # x_stim[:,0:7,:] = 0.0
        x_spikes = self.encode_spikes(snippet[:, -1][:, : self.in_len])
        z = einops.rearrange(z_decode, "b c -> b () c")
        x = torch.cat([x_stim, x_spikes, z], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.decode(x)
        return x, z_mu, z_logvar


class DistanceFieldCnnModel(nn.Module):
    LED_CHANNELS = 4
    NUM_CLUSTERS = 1

    def __init__(self, in_len, out_len):
        super(DistanceFieldCnnModel, self).__init__()
        self.in_len = in_len
        self.out_len = out_len
        # led_channels, mean(led_channels), num_clusters
        self.num_input_channels = self.LED_CHANNELS * 2 + self.NUM_CLUSTERS
        self.l1_num_channels = 50
        self.l2_num_channels = 50
        self.l3_num_channels = 100
        kernel_size = 21
        mid_kernel_size = 7
        self.layer0 = nn.Sequential(
            nn.Conv1d(
                self.num_input_channels,
                self.l1_num_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=(kernel_size - 1) // 2,
                bias=True,
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(
                self.l1_num_channels,
                self.l1_num_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            retinapy.nn.create_batch_norm(self.l1_num_channels),
            nn.LeakyReLU(0.2, True),
        )
        self.layer1 = nn.Sequential(
            retinapy.nn.ResBlock1d(
                self.l1_num_channels,
                self.l2_num_channels,
                self.l2_num_channels,
                kernel_size=mid_kernel_size,
                downsample=False,
            ),
        )
        self.layer2_elements = []
        expansion = 1
        num_halves = 3  # There is 1 other downsamples other than the midlayers.
        for i in range(num_halves - 1):
            self.layer2_elements.append(
                retinapy.nn.ResBlock1d(
                    self.l2_num_channels,
                    self.l2_num_channels * expansion,
                    self.l2_num_channels,
                    kernel_size=mid_kernel_size,
                    downsample=True,
                )
            )
        self.layer2 = nn.Sequential(*self.layer2_elements)
        self.layer3 = retinapy.nn.ResBlock1d(
            self.l2_num_channels,
            self.l3_num_channels,
            self.l3_num_channels,
            kernel_size=mid_kernel_size,
            downsample=False,
        )
        self.layer4 = nn.Conv1d(
            in_channels=self.l3_num_channels,
            out_channels=1,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        linear_in_len = 1 + (in_len - 1) // 2**num_halves
        self.linear = nn.Linear(
            in_features=linear_in_len,
            out_features=self.out_len,
        )

    def cat_mean(self, snippet):
        m = (
            snippet[:, 0 : mea.NUM_STIMULUS_LEDS]
            .mean(dim=2, keepdim=True)
            .expand(-1, -1, snippet.shape[-1])
        )
        x = torch.cat([snippet, m], dim=1)
        return x

    def set_input_mean_sd(self, mean, sd):
        """Not used."""
        pass

    def set_output_mean_sd(self, mean, sd):
        """Not used."""
        pass

    def forward(self, x):
        x = self.cat_mean(x)
        x = self.layer0(x)
        # Keep or leave out? Interested to see a justification either way.
        # x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        # Alternative parameter free head:
        # x = torch.squeeze(F.adaptive_avg_pool1d(x, output_size=self.out_len), dim=1)
        return x


class CnnPyramid(nn.Module):

    NUM_IN_CHANNELS = 5
    NUM_MID_CHANNELS = 64
    NUM_HEAD_CHANNELS = 64
    L1_KERNEL_SIZE = 15
    MID_KERNEL_SIZE = 5
    RESBLOCK_EXPANSION = 2
    DROPOUT_RATE = 0.4

    Objective = enum.Enum("Objective", "poisson distfield distfieldw")

    def __init__(
        self, in_len: int, out_len: int, n_mid: int, objective: Objective
    ):
        super().__init__()
        if objective == self.Objective.distfieldw:
            if out_len != 120:
                raise ValueError("out_len must be 120 for distfieldw")
        # 1 downsamples in layer 1.
        num_downsample = 1 + n_mid
        self.linear_in_len = 1 + (in_len - 1) // 2**num_downsample
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.NUM_IN_CHANNELS,
                out_channels=self.NUM_MID_CHANNELS
                if n_mid
                else self.NUM_HEAD_CHANNELS,
                kernel_size=self.L1_KERNEL_SIZE,
                stride=2,
                padding=(self.L1_KERNEL_SIZE - 1) // 2,
                bias=True,
            ),
            retinapy.nn.create_batch_norm(self.NUM_MID_CHANNELS),
            nn.LeakyReLU(0.2, True),
        )
        self.layer2_elements = []
        for i in range(n_mid):
            is_last = i == n_mid - 1
            if objective in {self.Objective.poisson, self.Objective.distfield}:
                has_downsample = True
            else:
                assert objective == self.Objective.distfieldw
                # Allow two downsamples to get length 120 output.
                has_downsample = i <= 1
            self.layer2_elements.append(
                retinapy.nn.ResBlock1d(
                    self.NUM_MID_CHANNELS,
                    self.NUM_MID_CHANNELS,
                    self.NUM_HEAD_CHANNELS
                    if is_last
                    else self.NUM_MID_CHANNELS,
                    kernel_size=self.MID_KERNEL_SIZE,
                    downsample=has_downsample,
                    dropout=self.DROPOUT_RATE,
                )
            )
        self.layer2 = nn.Sequential(*self.layer2_elements)
        if objective == self.Objective.poisson:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=self.NUM_HEAD_CHANNELS * self.linear_in_len,
                    out_features=1,
                ),
                nn.Softplus(),
                einops.layers.torch.Rearrange("b 1 -> b"),
            )
        elif objective == self.Objective.distfield:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=self.NUM_HEAD_CHANNELS * self.linear_in_len,
                    out_features=out_len,
                ),
            )
        elif objective == self.Objective.distfieldw:
            self.head = nn.Sequential(
                nn.Conv1d(
                    in_channels=self.NUM_HEAD_CHANNELS,
                    out_channels=1,
                    kernel_size=5,
                    bias=True,
                ),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"Unknown objective ({objective}).")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x)
        return x


class PoissonNet(nn.Module):

    NUM_IN_CHANNELS = 5
    NUM_MID_CHANNELS = 64
    L1_KERNEL_SIZE = 15
    DOWN_KERNEL_SIZE = 5
    MID_KERNEL_SIZE = 3
    DROPOUT_RATE = 0.3

    def __init__(self, in_len: int, n_down: int, n_mid: int):
        super().__init__()
        self.linear_in_len = 1 + (in_len - 1) // 2**n_down
        assert self.linear_in_len == 8, "For now, assuming 8"
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.NUM_IN_CHANNELS,
                out_channels=self.NUM_MID_CHANNELS,
                kernel_size=self.L1_KERNEL_SIZE,
                stride=2,
                padding=(self.L1_KERNEL_SIZE - 1) // 2,
                bias=True,
            ),
            retinapy.nn.create_batch_norm(self.NUM_MID_CHANNELS),
            nn.LeakyReLU(0.2, True),
        )
        downs = []
        for i in range(n_down - 1):
            downs.append(
                retinapy.nn.ResBlock1d(
                    self.NUM_MID_CHANNELS,
                    self.NUM_MID_CHANNELS,
                    self.NUM_MID_CHANNELS,
                    kernel_size=self.MID_KERNEL_SIZE,
                    downsample=True,
                    dropout=self.DROPOUT_RATE,
                )
            )
        self.down = nn.Sequential(*downs)
        mids = []
        for i in range(n_mid):
            mids.append(
                retinapy.nn.ResBlock1d(
                    self.NUM_MID_CHANNELS,
                    self.NUM_MID_CHANNELS,
                    self.NUM_MID_CHANNELS,
                    kernel_size=self.MID_KERNEL_SIZE,
                    downsample=False,
                    dropout=self.DROPOUT_RATE,
                )
            )
        self.mid = nn.Sequential(*mids)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.NUM_MID_CHANNELS * self.linear_in_len,
                out_features=1,
            ),
            nn.Softplus(),
            einops.layers.torch.Rearrange("b 1 -> b"),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.down(x)
        x = self.mid(x)
        x = self.head(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim_head = dim / heads
        if int(self.dim_head) != self.dim_head:
            raise ValueError(
                f"dim ({dim}) must be divisible by heads ({heads})"
            )
        self.dim_head = int(self.dim_head)
        self.scale = self.dim_head**-0.5
        self.heads = heads
        hidden_dim = self.dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.norm = retinapy.nn.create_batch_norm(dim)

    def forward(self, x):
        b, c, n = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) n -> b h c n", h=self.heads),
            qkv,
        )

        q = q * self.scale

        raw_attn = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = raw_attn.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        out = einops.rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


class ResAtt(nn.Module):
    def __init__(
        self,
        in_nch: int,
        out_nch: int,
        kernel_len: int,
        downsample: bool,
        dropout_rate: float,
        heads: int = 4,
    ):
        super().__init__()

        self.res = retinapy.nn.ConvNextv2(
            in_nch,
            out_nch,
            kernel_size=kernel_len,
            dropout=dropout_rate,
        )
        self.att = Attention(out_nch, heads)

    def forward(self, x):
        x = self.res(x)
        r = x.clone()
        x = self.att(x)
        x = x + r
        return x


class ResAttWithEmb(nn.Module):
    def __init__(
        self,
        in_nch: int,
        out_nch: int,
        kernel_len: int,
        dropout_rate: float,
        heads: int = 4,
    ):
        super().__init__()

        self.res = retinapy.nn.ConvNextv2(
            in_nch,
            out_nch,
            kernel_size=kernel_len,
            dropout=dropout_rate,
        )
        self.att = Attention(out_nch, heads)

    def forward(self, x, emb):
        x = self.res(x)
        # Concatenate embedding on time axis, bct.
        x = torch.cat((emb, x), dim=2)
        r = x.clone()
        x = self.att(x)
        x = x + r
        # Remove last element, as we have added an extra one.
        x = x[:, :, :-1]
        return x


class UpAtt(nn.Module):
    def __init__(
        self, in_nch, out_nch, kernel_len, heads: int = 4, dropout_rate=0.0
    ):
        super().__init__()
        self.convt = nn.ConvTranspose1d(in_nch, in_nch, 2, stride=2)
        self.att = ResAtt(
            in_nch,
            out_nch,
            kernel_len,
            downsample=False,
            dropout_rate=dropout_rate,
            heads=heads,
        )

    def forward(self, x):
        x = self.convt(x)
        x = self.att(x)
        return x


class AttCnnUNet(nn.Module):
    """
    dim   992 496 248 124  62  31  16    8
    down       32  32  64  64 128 128 128

    dim     8  16  32  64 128 128
    up        128 128  64  64   1
    """

    NUM_IN_CHANNELS = 5
    NUM_DOWN_CHANNELS = [64, 64, 64, 64, 64, 128, 128]
    NUM_UP_CHANNELS = [128, 64, 32, 16]
    NUM_MID_CHANNELS = 128
    L1_KERNEL_SIZE = 15
    DOWN_KERNEL_SIZE = 7
    MID_KERNEL_SIZE = 3
    UP_KERNEL_SIZE = 5
    DROPOUT_RATE = 0.3
    UP_DROPOUT_RATE = 0.10
    N_HEADS = 4

    def __init__(self, in_len, num_clusters, n_mid=5):
        super().__init__()
        self.num_clusters = num_clusters
        # This model has a normalization.
        self.register_buffer("output_mean", torch.tensor(1.0))
        self.register_buffer("output_sd", torch.tensor(1.0))
        self.register_buffer("input_mean", torch.zeros(size=(5,)))
        self.register_buffer("input_sd", torch.ones(size=(5,)))
        self.tag = TensorTag()

        # (in,out) pairs will be made from the following.
        down_nch = self.NUM_DOWN_CHANNELS  # including layer 1
        last_down_nch = down_nch[-1]
        enc_ch = last_down_nch
        mid_nch = [last_down_nch] + [self.NUM_MID_CHANNELS] * n_mid
        last_mid_nch = mid_nch[-1]
        up_nch = [last_mid_nch] + self.NUM_UP_CHANNELS

        self.layer1 = InitConv(
            in_nch=5, out_nch=down_nch[0], kernel_len=self.L1_KERNEL_SIZE
        )
        self.pos_embed = nn.Parameter(torch.randn(down_nch[0], in_len // 2))
        self.cid_embed = nn.Embedding(self.num_clusters, enc_ch)

        conv_fn = partial(
            retinapy.nn.ConvNextv2,
            kernel_size=self.DOWN_KERNEL_SIZE,
            dropout=self.DROPOUT_RATE,
        )
        self.down = nn.Sequential(
            *[
                nn.Sequential(
                    conv_fn(i_ch, o_ch),
                    # Rearrange("b c (w ds) -> b (c ds) w", ds=2),
                    # nn.Conv1d(o_ch * 2, o_ch, 1),
                    nn.Conv1d(o_ch, o_ch, kernel_size=3, stride=2, padding=1),
                )
                for i_ch, o_ch in pairwise(down_nch)
            ]
        )

        self.first_mid = ResAttWithEmb(
            mid_nch[0],
            mid_nch[1],
            kernel_len=self.DOWN_KERNEL_SIZE,
            heads=self.N_HEADS,
            dropout_rate=self.DROPOUT_RATE,
        )

        mids_fn = partial(
            ResAtt,
            downsample=False,
            kernel_len=self.MID_KERNEL_SIZE,
            dropout_rate=self.DROPOUT_RATE,
            heads=self.N_HEADS,
        )
        self.mid = nn.Sequential(
            *[mids_fn(i_ch, o_ch) for i_ch, o_ch in pairwise(mid_nch[1:])]
        )
        up_fn = partial(
            UpBlock3,
            kernel_size=self.UP_KERNEL_SIZE,
            dropout=self.UP_DROPOUT_RATE,
        )

        self.up = nn.Sequential(
            *[up_fn(i_ch, o_ch) for i_ch, o_ch in pairwise(up_nch)]
        )

        self.last = retinapy.nn.ConvNextv2(
            up_nch[-1],
            1,
            kernel_size=self.UP_KERNEL_SIZE,
            use_norm=False,
            layer_scale_init_value=0.0,
        )
        self.init_weights()

    def init_weights(self):
        # Last conv doesn't have relu activation, so use Xavier.
        nn.init.xavier_uniform_(self.last.pwconv2.weight)

    def set_input_mean_sd(self, m: torch.Tensor, sd: torch.Tensor):
        if m.shape != (5,):
            raise ValueError(
                "Input mean must have shape (5,). " f"Got ({m.shape})."
            )
        if sd.shape != (5,):
            raise ValueError(
                "Input sd must have shape (5,). " f"Got ({sd.shape})."
            )
        self.input_mean.copy_(m)
        self.input_sd.copy_(sd)

    def set_output_mean_sd(self, m: float, sd: float):
        """Set the value by which the log output dist is added to."""
        self.output_mean.fill_(m)
        self.output_sd.fill_(sd)

    def normalize_input(self, x):
        return (x - self.input_mean[None, :, None]) / self.input_sd[
            None, :, None
        ]

    def denormalize_output(self, x):
        return (x * self.output_sd) + self.output_mean

    def encode_cid(self, cid):
        res = einops.rearrange(self.cid_embed(cid.long()), "b c -> b c 1")
        return res

    def forward(self, x, cid):
        cid_enc = self.encode_cid(cid)
        x = self.tag(x, "input")
        x = self.normalize_input(x)
        x = self.tag(x, "input_normalized")
        x = self.layer1(x)
        x = self.tag(x, "layer1")
        x = x + self.pos_embed
        x = self.tag(x, "input_pos_embedded")
        x = self.down(x)
        x = self.tag(x, "down")
        x = self.first_mid(x, cid_enc)
        x = self.tag(x, "first_mid")
        x = self.mid(x)
        x = self.tag(x, "mid")
        x = self.up(x)
        x = self.tag(x, "up")
        x = self.last(x)
        x = self.tag(x, "last")
        x = self.denormalize_output(x)
        x = self.tag(x, "out")
        x = einops.rearrange(x, "b 1 w -> b w")
        return x


class FilmLayer(nn.Module):
    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def forward(self, x):
        gamma, beta = self.factory.get_weights(self)
        gamma = einops.rearrange(gamma, "b c -> b c 1")
        beta = einops.rearrange(beta, "b c -> b c 1")
        return gamma * x + beta
        return x


class Filmer(nn.Module):
    def __init__(self, embed_nch, n_out):
        super().__init__()
        self.fc1 = nn.Linear(embed_nch, n_out)
        # torch.nn.init.zeros_(self.fc1.weight)
        torch.nn.init.normal_(self.fc1.weight, mean=0.0, std=0.001)
        assert self.fc1.bias.shape[0] % 2 == 0
        # 1, 0, 1, 0, 1, 0
        bias_init = [1 - (i % 2) for i in range(self.fc1.bias.shape[0])]
        self.fc1.bias.data.copy_(torch.tensor(bias_init))

    def forward(self, x):
        x = self.fc1(x)
        return x


class FilmFactory:
    n_layers: int

    def __init__(self, embed_nch):
        self.n_params = 0
        self.embed_nch = embed_nch
        self.layer_to_slice = dict()
        self.activations = None

    def film_layer(self, out_nch) -> FilmLayer:
        res = FilmLayer(self)
        start = self.n_params
        n_params = 2 * out_nch
        end = start + n_params
        self.layer_to_slice[res] = slice(start, end)
        self.n_params += n_params
        return res

    def get_weights(
        self, layer: FilmLayer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.activations is not None
        sl = self.layer_to_slice[layer]
        # gamma, beta = self.activations[:, sl].chunk(2, dim=1)
        weights = self.activations[:, sl]
        gamma, beta = weights[:, ::2], weights[:, 1::2]
        return gamma, beta

    def construct(self) -> nn.Module:
        self.filmer = Filmer(self.embed_nch, self.n_params)
        return self.filmer

    def forward(self, embed):
        self.activations = self.filmer(embed)
        return self.activations


class FilmUNet(nn.Module):
    """
    dim   992 496 248 124  62  31  16    8
    down       32  32  64  64 128 128 128

    dim     8  16  32  64 128 128
    up        128 128  64  64   1
    """

    NUM_IN_CHANNELS = 5
    NUM_DOWN_CHANNELS = [64, 64, 128, 128, 128, 256, 256]
    NUM_UP_CHANNELS = [128, 64, 32, 16]
    NUM_MID_CHANNELS = 256
    L1_KERNEL_SIZE = 15
    DOWN_KERNEL_SIZE = 7
    MID_KERNEL_SIZE = 3
    UP_KERNEL_SIZE = 5
    DROPOUT_RATE = 0.3
    UP_DROPOUT_RATE = 0.05

    def __init__(self, in_len, num_clusters, n_mid=5):
        super().__init__()
        self.num_clusters = num_clusters
        # This model has a normalization.
        self.register_buffer("output_mean", torch.tensor(1.0))
        self.register_buffer("output_sd", torch.tensor(1.0))
        self.register_buffer("input_mean", torch.zeros(size=(5,)))
        self.register_buffer("input_sd", torch.ones(size=(5,)))
        self.tag = TensorTag()

        # (in,out) pairs will be made from the following.
        down_nch = self.NUM_DOWN_CHANNELS  # including layer 1
        last_down_nch = down_nch[-1]
        enc_ch = last_down_nch
        mid_nch = [last_down_nch] + [self.NUM_MID_CHANNELS] * n_mid
        last_mid_nch = mid_nch[-1]
        up_nch = [last_mid_nch] + self.NUM_UP_CHANNELS

        # FiLM
        self.film_factory = FilmFactory(enc_ch)
        self.layer1 = nn.Sequential(
            InitConv(
                in_nch=5, out_nch=down_nch[0], kernel_len=self.L1_KERNEL_SIZE
            ),
            self.film_factory.film_layer(down_nch[0]),
        )
        self.pos_embed = nn.Parameter(torch.randn(down_nch[0], in_len // 2))
        self.cid_embed = nn.Embedding(self.num_clusters, enc_ch)

        conv_fn = partial(
            retinapy.nn.ConvNextv2,
            kernel_size=self.DOWN_KERNEL_SIZE,
            dropout=self.DROPOUT_RATE,
        )
        self.down = nn.Sequential(
            *[
                nn.Sequential(
                    conv_fn(i_ch, o_ch),
                    self.film_factory.film_layer(o_ch),
                    TensorTag(f"FiLM_down_{i}"),
                    nn.Conv1d(o_ch, o_ch, kernel_size=3, stride=2, padding=1),
                )
                for i, (i_ch, o_ch) in enumerate(pairwise(down_nch))
            ]
        )

        mids_fn = partial(
            retinapy.nn.ConvNextv2,
            kernel_size=self.MID_KERNEL_SIZE,
            dropout=self.DROPOUT_RATE,
        )
        self.mid = nn.Sequential(
            *[
                nn.Sequential(
                    mids_fn(i_ch, o_ch),
                    self.film_factory.film_layer(o_ch),
                    TensorTag(f"FiLM_mid_{i}"),
                )
                for i, (i_ch, o_ch) in enumerate(pairwise(mid_nch))
            ]
        )
        up_fn = partial(
            UpBlock3,
            kernel_size=self.UP_KERNEL_SIZE,
            dropout=self.UP_DROPOUT_RATE,
        )

        self.up = nn.Sequential(
            *[
                nn.Sequential(
                    up_fn(i_ch, o_ch),
                    self.film_factory.film_layer(o_ch),
                    TensorTag(f"FiLM_up_{i}"),
                )
                for i, (i_ch, o_ch) in enumerate(pairwise(up_nch))
            ]
        )

        self.last = retinapy.nn.ConvNextv2(
            up_nch[-1],
            1,
            kernel_size=self.UP_KERNEL_SIZE,
            use_norm=False,
            layer_scale_init_value=0.0,
        )
        self.filmer = self.film_factory.construct()
        self.init_weights()

    def init_weights(self):
        # Last conv doesn't have relu activation, so use Xavier.
        nn.init.xavier_uniform_(self.last.pwconv2.weight)

    def set_input_mean_sd(self, m: torch.Tensor, sd: torch.Tensor):
        if m.shape != (5,):
            raise ValueError(
                "Input mean must have shape (5,). " f"Got ({m.shape})."
            )
        if sd.shape != (5,):
            raise ValueError(
                "Input sd must have shape (5,). " f"Got ({sd.shape})."
            )
        self.input_mean.copy_(m)
        self.input_sd.copy_(sd)

    def set_output_mean_sd(self, m: float, sd: float):
        """Set the value by which the log output dist is added to."""
        self.output_mean.fill_(m)
        self.output_sd.fill_(sd)

    def normalize_input(self, x):
        return (x - self.input_mean[None, :, None]) / self.input_sd[
            None, :, None
        ]

    def denormalize_output(self, x):
        return (x * self.output_sd) + self.output_mean

    def encode_cid(self, cid):
        res = self.cid_embed(cid.long())
        return res

    def calc_film_values(self, embed):
        return self.film_factory.forward(embed)

    def forward(self, x, cid):
        cid_enc = self.encode_cid(cid)
        film_vals = self.calc_film_values(cid_enc)
        self.tag(film_vals, "film_vals")
        x = self.tag(x, "input")
        x = self.normalize_input(x)
        x = self.tag(x, "input_normalized")
        x = self.layer1(x)
        x = self.tag(x, "layer1")
        x = x + self.pos_embed
        x = self.tag(x, "input_pos_embedded")
        x = self.down(x)
        x = self.tag(x, "down")
        x = self.mid(x)
        x = self.tag(x, "mid")
        x = self.up(x)
        x = self.tag(x, "up")
        x = self.last(x)
        x = self.tag(x, "last")
        x = self.denormalize_output(x)
        x = self.tag(x, "out")
        x = einops.rearrange(x, "b 1 w -> b w")
        return x


class CnnUNet(nn.Module):

    NUM_IN_CHANNELS = 5
    NUM_L1_CHANNELS = 64
    NUM_MID_CHANNELS = 64
    NUM_UP_CHANNELS = 16
    NUM_HEAD_CHANNELS = 16
    L1_KERNEL_SIZE = 15
    DOWN_KERNEL_SIZE = 5
    MID_KERNEL_SIZE = 3
    UP_KERNEL_SIZE = 5
    DROPOUT_RATE = 0.3

    def __init__(
        self,
        in_len: int,
        n_down: int,
        n_mid: int,
        n_up: int,
    ):
        super().__init__()
        if n_down < 1:
            raise ValueError("n_down must be at least 1")
        # This model has a normalization.
        self.register_buffer("mean_offset", torch.tensor(1.0))
        # This older model didn't have these:
        # self.register_buffer("input_mean", torch.zeros(size=(5,)))
        # self.register_buffer("input_sd", torch.ones(size=(5,)))

        self.pos_embed = nn.Parameter(torch.randn(self.NUM_IN_CHANNELS, in_len))
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.NUM_IN_CHANNELS,
                out_channels=self.NUM_L1_CHANNELS
                if n_down > 1
                else self.NUM_HEAD_CHANNELS,
                kernel_size=self.L1_KERNEL_SIZE,
                stride=2,
                padding=(self.L1_KERNEL_SIZE - 1) // 2,
                bias=True,
            ),
            retinapy.nn.create_batch_norm(self.NUM_L1_CHANNELS),
            nn.LeakyReLU(0.2, True),
        )
        downs = []
        for i in range(1, n_down):
            downs.append(
                retinapy.nn.ResBlock1d(
                    self.NUM_L1_CHANNELS,
                    self.NUM_L1_CHANNELS,
                    self.NUM_L1_CHANNELS,
                    kernel_size=self.DOWN_KERNEL_SIZE,
                    downsample=True,
                    dropout=self.DROPOUT_RATE,
                )
            )
        n_mid_in = self.NUM_L1_CHANNELS
        self.down = nn.Sequential(*downs)
        mids = []
        for i in range(n_mid):
            mids.append(
                retinapy.nn.ResBlock1d(
                    n_mid_in,
                    n_mid_in,
                    n_mid_in if i < n_mid - 1 else self.NUM_UP_CHANNELS,
                    kernel_size=self.MID_KERNEL_SIZE,
                    downsample=False,
                    dropout=self.DROPOUT_RATE,
                )
            )
        self.mid = nn.Sequential(*mids)

        ups = []
        for i in range(n_up):
            ups.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        self.NUM_UP_CHANNELS,
                        self.NUM_UP_CHANNELS,
                        2,
                        stride=2,
                    ),
                    retinapy.nn.ResBlock1d(
                        self.NUM_UP_CHANNELS,
                        self.NUM_UP_CHANNELS,
                        self.NUM_UP_CHANNELS
                        if i < n_up - 1
                        else self.NUM_HEAD_CHANNELS,
                        kernel_size=self.UP_KERNEL_SIZE,
                        downsample=False,
                        dropout=self.DROPOUT_RATE,
                    ),
                )
            )
        self.up = nn.Sequential(*ups)

        self.head = nn.Sequential(
            # retinapy.nn.ResBlock1d(
            #     self.NUM_UP_CHANNELS,
            #     self.NUM_UP_CHANNELS,
            #     self.NUM_HEAD_CHANNELS,
            #     kernel_size=3,
            #     downsample=False,
            #     dropout=self.DROPOUT_RATE / 2,
            # ),
            nn.Conv1d(
                in_channels=self.NUM_HEAD_CHANNELS,
                out_channels=1,
                kernel_size=3,
                bias=True,
            ),
        )

    def set_input_mean_sd(self, m: torch.Tensor, sd: torch.Tensor):
        pass

    def set_output_mean(self, m: float):
        """Set the value by which the output dist is multiplied by."""

        def inv_softplus(x):
            """
            From:
            https://github.com/pytorch/pytorch/issues/72759#issuecomment-1236496693
            """
            return x + torch.log(-torch.expm1(-x))

        # m = inv_softplus(m)
        # m = self.mean_offset.new_tensor([m])
        self.mean_offset.fill_(m)

    def forward(self, x):
        x += self.pos_embed
        x = self.layer1(x)
        x = self.down(x)
        x = self.mid(x)
        x = self.up(x)
        x = self.head(x)
        # Flatten
        x = einops.rearrange(x, "b 1 w -> b w")
        x += self.mean_offset
        return x


class CnnUNet2(nn.Module):
    """
    CnnUNet using ConvNext blocks.
    """

    NUM_IN_CHANNELS = 5
    NUM_L1_CHANNELS = 64
    NUM_DOWN_CHANNELS = 64
    NUM_MID_CHANNELS = 64
    NUM_UP_CHANNELS = 16
    NUM_HEAD_CHANNELS = 16
    L1_KERNEL_SIZE = 15
    DOWN_KERNEL_SIZE = 5
    MID_KERNEL_SIZE = 3
    UP_KERNEL_SIZE = 5
    DROPOUT_RATE = 0.3

    def __init__(
        self,
        in_len: int,
        n_down: int,
        n_mid: int,
        n_up: int,
    ):
        super().__init__()
        if n_down < 1:
            raise ValueError("n_down must be at least 1")
        # This model has a normalization.
        self.register_buffer("output_mean", torch.tensor(1.0))
        self.register_buffer("output_sd", torch.tensor(1.0))
        self.register_buffer("input_mean", torch.zeros(size=(5,)))
        self.register_buffer("input_sd", torch.ones(size=(5,)))
        self.tag = TensorTag()

        self.layer1 = InitConv(
            in_nch=self.NUM_IN_CHANNELS,
            out_nch=self.NUM_L1_CHANNELS,
            kernel_len=self.L1_KERNEL_SIZE,
        )

        self.pos_embed = nn.Parameter(
            retinapy.nn.get_sinusoidal_embeddings(
                in_len // 2, self.NUM_L1_CHANNELS
            )
        )

        downs_fn = partial(
            retinapy.nn.ResBlock1d,
            kernel_size=self.DOWN_KERNEL_SIZE,
            downsample=True,
            dropout=self.DROPOUT_RATE,
        )
        down_nch = [self.NUM_L1_CHANNELS] + [self.NUM_DOWN_CHANNELS] * (
            n_down - 1
        )
        self.down = nn.Sequential(
            *[downs_fn(i_ch, o_ch, o_ch) for i_ch, o_ch in pairwise(down_nch)]
        )

        # Slightly uncanonical to have the last layer take the number of
        # channels of the up block.
        mid_nch = (
            [down_nch[-1]]
            + [self.NUM_MID_CHANNELS] * (n_mid - 1)
            + [self.NUM_UP_CHANNELS]
        )
        mids_fn = partial(
            retinapy.nn.ResBlock1d,
            kernel_size=self.MID_KERNEL_SIZE,
            downsample=False,
            dropout=self.DROPOUT_RATE,
        )
        self.mid = nn.Sequential(
            *[mids_fn(i_ch, o_ch, o_ch) for i_ch, o_ch in pairwise(mid_nch)]
        )

        up_fn = partial(
            UpResBlock,
            kernel_size=self.UP_KERNEL_SIZE,
            dropout=self.DROPOUT_RATE,
        )
        up_nch = (
            [mid_nch[-1]]
            + [self.NUM_UP_CHANNELS] * (n_up - 1)
            + [self.NUM_HEAD_CHANNELS]
        )
        self.up = nn.Sequential(
            *[up_fn(i_ch, o_ch) for i_ch, o_ch in pairwise(up_nch)]
        )

        self.head = nn.Conv1d(
            in_channels=self.NUM_HEAD_CHANNELS,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        # Last conv doesn't have relu activation, so use Xavier.
        nn.init.xavier_uniform_(self.head.weight)

    def set_input_mean_sd(self, m: torch.Tensor, sd: torch.Tensor):
        if m.shape != (5,):
            raise ValueError(
                "Input mean must have shape (5,). " f"Got ({m.shape})."
            )
        if sd.shape != (5,):
            raise ValueError(
                "Input sd must have shape (5,). " f"Got ({sd.shape})."
            )
        self.input_mean.copy_(m)
        self.input_sd.copy_(sd)

    def set_output_mean_sd(self, m: float, sd: float):
        """Set the value by which the log output dist is added to."""
        self.output_mean.fill_(m)
        self.output_sd.fill_(sd)

    def normalize_input(self, x):
        return (x - self.input_mean[None, :, None]) / self.input_sd[
            None, :, None
        ]

    def denormalize_output(self, x):
        return (x * self.output_sd) + self.output_mean

    def forward(self, x):
        x = self.tag(x, "1. input")
        x = self.normalize_input(x)
        x = self.tag(x, "2. input_normalized")
        x = self.layer1(x)
        x = self.tag(x, "3. layer1")
        x = x + self.pos_embed
        x = self.tag(x, "4. input_pos_embedded")
        x = self.down(x)
        x = self.tag(x, "5. down")
        x = self.mid(x)
        x = self.tag(x, "6. mid")
        x = self.up(x)
        x = self.tag(x, "7. up")
        x = self.head(x)
        x = self.tag(x, "8. last")
        x = self.denormalize_output(x)
        x = self.tag(x, "9. out")
        x = einops.rearrange(x, "b 1 w -> b w")
        return x


class LinearNonlinear(nn.Module):
    def __init__(self, in_n, out_n):
        super(LinearNonlinear, self).__init__()
        self.linear = nn.Linear(in_features=in_n, out_features=out_n)
        self.act = nn.Softplus()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.act(x)
        # Remove the last dimension, as it is always 1.
        assert x.shape[1] == 1
        x = torch.squeeze(x, dim=1)
        return x


class DeepRetina2016(nn.Module):
    """
    From:
        https://github.com/baccuslab/deep-retina/blob/master/deepretina/models.py
    """

    def __init__(self, in_len, in_n, out_n):
        super(DeepRetina2016, self).__init__()
        self.in_n = in_n
        self.out_n = out_n
        self.in_len = in_len

        self.conv1 = nn.Conv2d(
            in_channels=self.in_n, out_channels=16, kernel_size=15
        )
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=9)
        linear_in_len = in_len - 15 + 1 - 9 + 1
        self.linear = nn.Linear(linear_in_len, self.out_n)
        self.act = nn.Softplus()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = self.act(x)


class PixelCnn(nn.Module):
    """Porting the kernel masking idea from PixelCNN."""

    IN_N = 5  # 4 stimulus + distfield
    MID_N = 64
    EXPANSION = 3
    HEAD_N = 256
    NUM_MID_LAYERS = 14

    def __init__(self):
        super().__init__()
        self.layer1 = retinapy.nn.LeftRightCausalConv1d(
            in_n=self.IN_N,
            out_n=self.MID_N,
            kernel_size=7,
            bias=True,
            groups=1,
            block_center=True,
        )
        self.mid_layers = nn.Sequential(
            *[
                retinapy.nn.CausalResBlock1d(
                    in_n=self.MID_N,
                    mid_n=self.MID_N * self.EXPANSION,
                    out_n=self.MID_N,
                    kernel_size=5,
                    use_depthwise_conv=True,
                )
                for _ in range(self.NUM_MID_LAYERS - 1)
            ]
        )
        self.head = nn.Sequential(
            retinapy.nn.CausalResBlock1d(
                in_n=self.MID_N,
                mid_n=self.MID_N * self.EXPANSION,
                out_n=self.HEAD_N,
                kernel_size=5,
                use_depthwise_conv=True,
            ),
            torch.nn.Conv1d(
                in_channels=self.HEAD_N, out_channels=1, kernel_size=1
            ),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.mid_layers(x)
        x = self.head(x)
        x = einops.rearrange(x, "b 1 t -> b t")
        return x
