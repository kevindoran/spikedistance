"""
A lot of experimentation ends up here in nn.py. At some point, the lesser used
functions and modules should be moved elsewhere and leave nn.py to only contain
the frequently used and well tested functionality that can be used in multiple
projects.
"""

import logging
import math
from typing import Callable, Optional
import einops
import torch
import torch.nn as nn
from torch.nn import functional as F

_logger = logging.getLogger(__name__)


def identity_fn(x, *args, **kwargs):
    return x


def create_identity(*args, **kwargs):
    return nn.Identity()


def create_batch_norm(n: int) -> nn.Module:
    """
    Following Davis and Frank's recommendation in "Revisiting Batch
    Normalization", we would initialize batch norm weights to less than 1
    (they suggested to use 0.1). They also recommended using a lower learning
    rate for the γ parameter.

    For comparison, fastai initialize β to 0.001 and γ to 1.

    I tried both, and found better results with fastai's defaults.
    """
    bn = nn.BatchNorm1d(n)
    # fastai
    # bn.weight.data.fill_(1.0)
    # bn.bias.data.fill_(1e-3)
    # Davis and Frank
    bn.weight.data.fill_(0.1)
    bn.bias.data.fill_(0)
    return bn


def create_shortcut(in_n, out_n, stride, num_dim=1):
    """Residual connection.

    The identify path is one of those finicky bits of ResNet type networks.

    Depending on whether the input and output match in terms of channel
    count and dimension, we have the following behaviour:

    Match?
    ------

        | Channel count | Dimensions | Bevaviour                  |
        |---------------|------------|----------------------------|
        |      ✓        |     ✓      | identity                   |
        |      ✓        |     ✘      | pool or conv               |
        |      ✘        |     ✓      | 1x1 conv                   |
        |      ✘        |     ✘      | pool or conv and 1x1 conv  |


    The most interesting choice is whether to use a strided pool or a
    strided convolution to achieve the downsampling effect. It's
    interesting as implementations are split on which one to use. There
    are further choices too, such as whether to use dilation in addition
    to strides, and whether to downsample before or after the 1x1 conv.

    Some implementations for reference:
        - fastai: https://github.com/fastai/fastai/blob/aa58b1316ad8e7a5fa2e434e15e5fe6df4f4db56/nbs/01_layers.ipynb
        - lightning: https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/self_supervised/resnets.py
        - pytorch image models: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py
             - there are two functions, downsample_conv() and
               downsample_avg() that are used to create the downsampling for
               the shortcut connection.
             - uses ceil_mode=True, so a dimension n=7 would be reduced to
               n=4, not n=3.

    My gut instinct is that swapping out pooling for convolution to achieve
    the downsample will naively achieve better results; however, the
    convolution is intrinsically more powerful (actual has parameters) and
    so, if you care about parameter counts, then a fair comparison would
    involve reducing parameters elsewhere. Given that the whole point of
    the residual layer is to short circuit a layer and allow gradients to
    flow easily, I think that pooling gets more theory points for staying
    true to this idea. Ironically, I think the original ResNet
    implementation might have used convolution."""
    # 1. Identity
    # In the simplest case, we can just return the input. For this to work,
    # both the channel count and channel dimensions of the input and output
    # must match. The output channel dimension is determined by the stride.
    if num_dim == 1:
        poolXdClass = nn.AvgPool1d
        convXdClass = nn.Conv1d
    elif num_dim == 2:
        poolXdClass = nn.AvgPool2d
        convXdClass = nn.Conv2d
    else:
        raise ValueError(f"Only 1d and 2d supported; got ({num_dim}).")
    channels_match = in_n == out_n
    downsample = stride > 1
    if channels_match and not downsample:
        return nn.Identity()

    skip_layers = []
    # 2. Downsample
    if downsample:
        # The following (1, 7) input:
        # |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
        # when pooled with stride=kernel=2 becomes (1, 4):
        # |    1.5    |    3.5    |    5.5   | 7 |
        pool = poolXdClass(
            kernel_size=stride,
            stride=stride,
            count_include_pad=False,
            ceil_mode=True,
        )
        skip_layers.append(pool)
    # 3. 1x1 conv
    if not channels_match:
        # There isn't a consensus on whether:
        #   - to use batch norm or a bias or neither.
        conv = convXdClass(in_n, out_n, kernel_size=1, bias=False)
        # TODO: how is best to initialize this connection?
        skip_layers.append(conv)
    res = nn.Sequential(*skip_layers)
    return res


@torch.no_grad()
def get_sinusoidal_embeddings(length: int, n_ch: int) -> torch.Tensor:
    """
    Create a tensor of shape [time_steps, n_channels//2] and fill all rows with
    the sequence: [0, 1, 2, ..., length-1]. Then, scale each row by a different
    value. The list of scaling values are chosen to span from 1 to 1/10000.
    The 0th channel is scaled by 1, then gradually the scale is reduced
    until the last channel, (n_ch//2 -1), is scaled by 1/10000. The steps are
    exponentially spaced, so the scaling will initially rapidly decrease, then
    slow down as the minimum is approached.
    This resulting array is used as input to two functions: sine and cos, and
    the two results are concatenated in order to get a tensor of shape
    [time_steps, n_channels].
    """
    half_nch = n_ch // 2
    # From 1 to 1/10000, exponentially in half_nch steps.
    slope = (
        torch.arange(half_nch, dtype=torch.float)
        * -math.log(10000)
        / (half_nch - 1)
    ).exp()
    t = slope[:, None] * torch.arange(length, dtype=torch.float)[None, :]
    res = torch.cat([t.sin(), t.cos()], dim=0).to(dtype=torch.float32)
    return res


class Decoder1dBlock(nn.Module):
    """
    I referred to decoder architecture here:
    https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/decoders
    """

    def __init__(self, in_channels, out_channels, act=True):
        super(Decoder1dBlock, self).__init__()
        self.act = act
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            # No need for bias here, since we're using batch norm.
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            # No need for bias here, since we're using batch norm.
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.relu(self.bn1(self.conv1(x)))
        if self.act:
            x = F.relu(self.bn2(self.conv2(x)))
        else:
            x = self.conv2(x)
        return x


class ResBlock1d(nn.Module):
    """A residual block with 1d convolutions.

    This is a pretty inflexible implementation. No need to make it any
    more general yet.
    """

    def __init__(
        self,
        in_n,
        mid_n,
        out_n,
        kernel_size=3,
        downsample=False,
        dropout: float = 0,
        act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        norm_factory: Optional[
            Callable[[int], torch.nn.Module]
        ] = create_batch_norm,
    ):
        """
        Args:
            act_fn: A function like F.relu.
            norm_factory: a function that returns a Pytorch module. Other
                libraries take a class, and call Class(...) making the __init__
                effectively a factory. However, if you want to do any parameter
                initialization or anything, the interface must be a standard
                function.
        """
        super(ResBlock1d, self).__init__()
        self.downsample = downsample
        self.dropout_rate = dropout
        self.act_fn = identity_fn if act_fn is None else act_fn
        self.norm_factory = (
            create_identity if norm_factory is None else norm_factory
        )
        stride = 2 if self.downsample else 1
        self.shortcut = create_shortcut(in_n, out_n, stride=stride)
        # Note: bias is False for the conv layers, as they will be followed
        # by batch norm.
        self.conv1 = nn.Conv1d(
            in_n,
            mid_n,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        padding = (kernel_size - 1) // 2
        self.conv2 = nn.Conv1d(
            mid_n,
            mid_n,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            bias=False,
        )
        self.conv3 = nn.Conv1d(
            mid_n,
            out_n,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        self.dropout = (
            nn.Dropout(p=self.dropout_rate)
            if self.dropout_rate > 0
            else nn.Identity()
        )
        # To use batch norm or group norm?
        self.bn1 = self.norm_factory(mid_n)
        self.bn2 = self.norm_factory(mid_n)
        self.bn3 = self.norm_factory(out_n)

        self.se = SEModule(out_n)

    def init_weights(self):
        nn.init.kaiming_normal_(
            self.conv1.weight, mode="fan_in", nonlinearity="relu"
        )
        nn.init.kaiming_normal_(
            self.conv2.weight, mode="fan_in", nonlinearity="relu"
        )
        if self.act_fn == F.relu:
            nn.init.kaiming_normal_(
                self.conv3.weight, mode="fan_in", nonlinearity="relu"
            )
        else:
            _logger.info(
                "conv3 has no activation function; no attempt is made "
                "to guess a good normalization."
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.bn3(self.conv3(x))
        x = self.se(x) + shortcut
        x = self.act_fn(x)
        return x


def drop_path(
    x,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class LayerNorm(nn.Module):
    """LayerNorm that supports channels_last (default) or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None] * x + self.bias[:, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNextv2(nn.Module):
    def __init__(
        self,
        in_n,
        out_n,
        kernel_size,
        expansion=4,
        dropout=0.0,
        layer_scale_init_value=1e-6,
        use_norm=True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        super().__init__()
        self.out_n = out_n
        self.dwconv = nn.Conv1d(
            in_n,
            in_n,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=in_n,
        )
        self.norm = LayerNorm(in_n, eps=1e-6) if use_norm else nn.Identity()
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(in_n, expansion * in_n)
        self.act = nn.GELU()
        self.grn = GRN(expansion * in_n) if use_norm else nn.Identity()
        self.pwconv2 = nn.Linear(expansion * in_n, out_n)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_n)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        b, c, t = x.shape
        input = x
        if c > self.out_n:
            # A rudimentary shortcut connection. This is only used once
            # at the junction between the shared base and the distance head.
            # A more typical 1x1 conv could have been used; however, I wanted
            # to simplify the parameter count calculation and reporting.
            input = input[:, : self.out_n, :]
        x = self.dwconv(x)
        x = einops.rearrange(x, "n c t -> n t c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = einops.rearrange(x, "n t c -> n c t")
        x = input + self.dropout(x)
        return x


class LeftRightCausalConv1d(nn.Conv1d):
    """

    For length 5, uses the following kernel passthrough mask, broadcasted along
    batch and channel:

        1 1 X 0 0

    With X=0 if block_center is True, and X=1 otherwise.
    """

    def __init__(
        self,
        in_n: int,
        out_n: int,
        kernel_size: int,
        bias: bool,
        groups: int,
        block_center: bool,
    ):
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got ({kernel_size}).")
        super().__init__(
            in_n,
            out_n,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            groups=groups,
            bias=bias,
        )
        self.block_center = block_center
        _in, _out, l = self.weight.shape
        mask = torch.zeros((_in, _out, l))
        mask[:, :, : l // 2 + int(not self.block_center)] = 1
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class CausalResBlock1d(nn.Module):
    def __init__(
        self,
        in_n,
        mid_n,
        out_n,
        kernel_size=3,
        mask_center=False,
        use_depthwise_conv=False,
    ):
        super().__init__()
        self.shortcut = create_shortcut(in_n, out_n, stride=1)
        self.conv1 = nn.Conv1d(
            in_n,
            mid_n,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        self.conv2 = LeftRightCausalConv1d(
            mid_n,
            mid_n,
            kernel_size=kernel_size,
            bias=False,
            groups=mid_n if use_depthwise_conv else 1,
            block_center=mask_center,
        )
        self.conv3 = nn.Conv1d(
            mid_n,
            out_n,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
        )
        self.bn1 = create_batch_norm(mid_n)
        self.bn2 = create_batch_norm(mid_n)
        self.bn3 = create_batch_norm(out_n)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + shortcut
        x = F.relu(x)
        return x


class FcLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.net(input)


class FcBlock(nn.Module):
    def __init__(
        self,
        hidden_ch,
        num_hidden_layers,
        in_features,
        out_features,
        outermost_linear=False,
    ):
        super().__init__()

        self.net_elements = []
        self.net_elements.append(nn.Flatten())
        self.net_elements.append(
            FcLayer(in_features=in_features, out_features=hidden_ch)
        )

        for i in range(num_hidden_layers):
            self.net_elements.append(
                FcLayer(in_features=hidden_ch, out_features=hidden_ch)
            )

        if outermost_linear:
            self.net_elements.append(
                nn.Linear(in_features=hidden_ch, out_features=out_features)
            )
        else:
            self.net_elements.append(
                FcLayer(in_features=hidden_ch, out_features=out_features)
            )

        self.net = nn.Sequential(*self.net_elements)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net_elements[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(
                m.weight, a=0.0, nonlinearity="relu", mode="fan_in"
            )

    def forward(self, input):
        ans = self.net(input)
        return ans


class Conv1dSame(torch.nn.Module):
    """1D convolution that pads to keep spatial dimensions equal.
    Cannot deal with stride. Only quadratic kernels (=scalar kernel_size).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        padding_layer=nn.ReflectionPad1d,
    ):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Scalar. Spatial dimensions of kernel (only quadratic kernels supported).
        :param bias: Whether or not to use bias.
        :param padding_layer: Which padding to use. Default is reflection padding.
        """
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            nn.Conv1d(
                in_channels, out_channels, kernel_size, bias=bias, stride=1
            ),
        )

        self.weight = self.net[1].weight
        self.bias = self.net[1].bias

    def forward(self, x):
        return self.net(x)


class Conv1dWarehouse(nn.Module):
    def __init__(
        self,
        max_in_channels,
        warehouse_size,
        kernel_size=7,
        key_len=16,
    ):
        super().__init__()
        assert kernel_size == 7, "Currently, only kernel_size=7 is supported."
        self.max_in_channels = max_in_channels
        self.warehouse_size = warehouse_size
        self.key_len = key_len
        self.weights_per_kernel = kernel_size * self.max_in_channels
        self.warehouse_W = nn.Linear(
            self.warehouse_size, self.weights_per_kernel
        )
        self.warehouse_b = nn.Linear(self.warehouse_size, 1)
        self.keys = nn.Linear(self.key_len, self.warehouse_size)

    def forward(self, q):
        """
        q.shape = (batch, num_channels, key_len)
        """
        batch_size = q.shape[0]
        num_kernels = q.shape[1]
        att = F.softmax(self.keys(q), dim=2)
        weights_W = self.warehouse_W(att)
        weights_b = self.warehouse_b(att).unsqueeze(-1)
        return weights_W, weights_b

    def forward_old(self, q, num_in_channels):
        """
        q.shape = (batch, num_channels, key_len)
        """
        batch_size = q.shape[0]
        num_kernels = q.shape[1]
        # soft
        # att = F.softmax(self.keys(q), dim=2)
        # hard
        att = F.argmax(self.keys(q), dim=2, keepdim=True)
        weights_W = self.warehouse_W(att)
        weights_W = weights_W.view(
            batch_size, num_kernels, self.max_in_channels, self.kernel_size
        )
        weights_W = weights_W[:, :, :num_in_channels, :]
        weights_b = self.warehouse_b(att).unsqueeze(-1)
        return weights_W, weights_b


class LayerNorm1d(nn.LayerNorm):
    """LayerNorm for channels of '2D' spatial NCHW tensors"""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x.permute(0, 2, 1),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        ).permute(0, 2, 1)


class ResBlock1d_F(nn.Module):
    """Almost functional residual block.

    Need to get rid of bath norm to be fully functional.
    """

    def __init__(
        self,
        in_n,
        mid_n,
        out_n,
        kernel_size=3,
        downsample=False,
    ):
        super().__init__()
        self.downsample = downsample
        self.in_n = in_n
        self.mid_n = mid_n
        self.out_n = out_n
        self.kernel_size = kernel_size
        self.stride = 2 if self.downsample else 1

        self.shortcut = create_shortcut(in_n, out_n, stride=self.stride)
        self.ln2 = create_batch_norm(mid_n)  # LayerNorm1d(mid_n)
        self.ln3 = create_batch_norm(out_n)  # LayerNorm1d(out_n)

    def num_channels(self):
        return self.out_n + self.mid_n * 2

    def forward(self, x, all_W, all_b, from_idx):
        batch_size = x.shape[0]
        W1 = (
            all_W[:, from_idx : from_idx + self.mid_n]
            .view(batch_size, self.mid_n, self.kernel_size)
            .contiguous()
        )
        # Only use a subset of the full available layer.
        # W1 = W1[:, :, : self.in_n, :].contiguous()
        b1 = all_b[:, from_idx : from_idx + self.mid_n].view(batch_size, -1, 1)
        from_idx += self.mid_n
        W2 = (
            all_W[:, from_idx : from_idx + self.mid_n]
            .view(batch_size, self.mid_n, self.kernel_size)
            .contiguous()
        )
        # W2 = W2[:, :, : self.mid_n, :]
        from_idx += self.mid_n
        W3 = (
            all_W[:, from_idx : from_idx + self.out_n]
            .view(batch_size, self.out_n, self.kernel_size)
            .contiguous()
        )
        # W3 = W3[:, :, : self.mid_n, :]
        from_idx += self.out_n
        x = self._forward(x, W1, b1, W2, W3)
        return x, from_idx

    def _forward(self, x, W1, b1, W2, W3):
        shortcut = self.shortcut(x)
        x = F.relu(self.batch_unique_conv(x, W1) + b1)
        x = self.batch_unique_conv(x, W2, self.stride)
        x = F.relu(self.ln2(x))
        x = F.relu(self.ln3(self.batch_unique_conv(x, W3)))
        x = F.relu(x + shortcut)
        return x

    def batch_unique_conv(self, x, W, stride=1):
        batch_size = x.shape[0]
        W = W.view(-1, 1, self.kernel_size)
        num_out_channels = W.shape[0] // batch_size
        num_in_channels = x.shape[1]
        assert W.shape[0] % batch_size == 0
        # W = torch.reshape(W, shape=(-1, x.shape[1], self.kernel_size))
        x_flat = x.view(1, -1, x.shape[-1])
        x_flat = F.pad(x_flat, (self.kernel_size // 2, self.kernel_size // 2))
        x = F.conv1d(
            x_flat,
            W,
            bias=None,
            padding="valid",
            stride=stride,
            # Not sure about this!
            groups=batch_size * num_in_channels,
        )
        x = x.view(batch_size, num_out_channels, x.shape[-1])
        return x


class HyperResDecoder(nn.Module):
    def __init__(self, hyper_res_block, in_n, hidden1_n, hidden2_n):
        super().__init__()
        self.hyper_res_block = hyper_res_block
        self.fc1 = nn.Linear(in_n, hidden1_n)
        self.fc2 = nn.Linear(hidden1_n, hidden2_n)
        self.att1_shape = (
            self.hyper_res_block.mid_n,
            self.hyper_res_block.mid_warehouse_n,
        )
        self.att2_shape = (
            self.hyper_res_block.mid_n,
            self.hyper_res_block.mid_warehouse_n,
        )
        self.att3_shape = (
            self.hyper_res_block.out_n,
            self.hyper_res_block.out_warehouse_n,
        )
        self.fcA1 = nn.Linear(
            hidden2_n, self.att1_shape[0] * self.att1_shape[1]
        )
        self.fcA2 = nn.Linear(
            hidden2_n, self.att2_shape[0] * self.att2_shape[1]
        )
        self.fcA3 = nn.Linear(
            hidden2_n, self.att3_shape[0] * self.att3_shape[1]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fcA1.weight)
        nn.init.xavier_uniform_(self.fcA2.weight)
        nn.init.xavier_uniform_(self.fcA3.weight)
        self.fcA1.bias.data.fill_(0.0)
        self.fcA2.bias.data.fill_(0.0)
        self.fcA3.bias.data.fill_(0.0)

    def forward(self, x, z):
        batch_size = x.shape[0]
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        key_scale = math.sqrt(h.shape[-1])
        att1 = F.softmax(
            self.fcA1(h).view(
                batch_size,
                self.hyper_res_block.mid_n,
                self.hyper_res_block.mid_warehouse_n,
            )
            / key_scale,
            dim=2,
        )
        att2 = F.softmax(
            self.fcA2(h).view(
                batch_size,
                self.hyper_res_block.mid_n,
                self.hyper_res_block.mid_warehouse_n,
            )
            / key_scale,
            dim=2,
        )
        att3 = F.softmax(
            self.fcA3(h).view(
                batch_size,
                self.hyper_res_block.out_n,
                self.hyper_res_block.out_warehouse_n,
            )
            / key_scale,
            dim=2,
        )
        x = self.hyper_res_block(x, att1, att2, att3)
        return x


class SEModule(nn.Module):
    def __init__(self, n_channels, reduction=1):
        super().__init__()
        n_mid = n_channels // reduction
        self.fc1 = nn.Conv1d(n_channels, n_mid, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(n_mid, n_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in")
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        module_input = x
        x = x.mean((2,), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class HyperResBlock1d(nn.Module):
    def __init__(
        self,
        in_n,
        mid_n,
        out_n,
        warehouse_factor,
        kernel_size=3,
        downsample=False,
        squeeze=True,
    ):
        super().__init__()
        self.downsample = downsample
        self.in_n = in_n
        self.mid_n = mid_n
        self.out_n = out_n
        self.mid_warehouse_n = mid_n * warehouse_factor
        self.out_warehouse_n = out_n * warehouse_factor
        self.kernel_size = kernel_size
        stride = 2 if self.downsample else 1

        self.num_W1_weights = in_n * kernel_size
        self.num_W2_weights = mid_n * kernel_size
        self.num_W3_weights = mid_n * kernel_size

        self.shortcut = self.create_shortcut(in_n, out_n, stride=stride)
        self.W1_weights = nn.Linear(self.mid_warehouse_n, self.num_W1_weights)
        self.W2_weights = nn.Linear(self.mid_warehouse_n, self.num_W2_weights)
        self.W3_weights = nn.Linear(self.out_warehouse_n, self.num_W3_weights)

        # Bias is a way to inject channel information via attention, so
        # leave at least one layer using bias rather than batch norm.
        # Batch norm
        # self.bn1 = create_batch_norm(mid_n)
        self.bn2 = create_batch_norm(mid_n)
        self.bn3 = create_batch_norm(out_n)
        # Bias instead.
        self.b1_weights = nn.Linear(self.mid_warehouse_n, 1)
        # self.b2_weights = nn.Linear(mid_warehouse_n, 1)
        # self.b3_weights = nn.Linear(mid_warehouse_n, 1)
        self.se = SEModule(out_n)

    @staticmethod
    def create_shortcut(in_n, out_n, stride):
        # 1. Identity
        # In the simplest case, we can just return the input. For this to work,
        # both the channel count and channel dimensions of the input and ouput
        # must match. The output channel dimension is determined by the stride.
        channels_match = in_n == out_n
        downsample = stride > 1
        if channels_match and not downsample:
            return nn.Identity()

        skip_layers = []
        # 2. Downsample
        if downsample:
            # The following (1, 7) input:
            # |  1  |  2  |  3  |  4  |  5  |  6  |  7  |
            # when pooled with stride=kernel=2 becomes (1, 4):
            # |    1.5    |    3.5    |    5.5   | 7 |
            pool = nn.AvgPool1d(
                kernel_size=stride,
                stride=stride,
                count_include_pad=False,
                ceil_mode=True,
            )
            skip_layers.append(pool)
        # 3. 1x1 conv
        if not channels_match:
            # There isn't a consensus on whether:
            #   - to use batch norm or a bias or neither.
            conv = nn.Conv1d(in_n, out_n, kernel_size=1, bias=False)
            skip_layers.append(conv)
        res = nn.Sequential(*skip_layers)
        return res

    def forward(self, x, att1, att2, att3):
        batch_size = x.shape[0]
        shortcut = self.shortcut(x)
        # Attention
        W1 = self.W1_weights(att1).view(
            batch_size * self.mid_n, self.in_n, self.kernel_size
        )
        W2 = self.W2_weights(att2).view(
            batch_size * self.mid_n, self.mid_n, self.kernel_size
        )
        W3 = self.W3_weights(att3).view(
            batch_size * self.out_n, self.mid_n, self.kernel_size
        )
        b1 = self.b1_weights(att1).view(batch_size, self.mid_n)

        # b2 = self.b2_weights(att2).view(batch_size, self.mid_n)
        # b3 = self.b3_weights(att3).view(batch_size, self.out_n)
        # Conv
        # Use bias?
        x = F.relu(self.att_conv(x, W1, b1))
        x = F.relu(self.bn2(self.att_conv(x, W2)))
        x = F.relu(self.bn3(self.att_conv(x, W3)))
        x = self.se(x) + shortcut
        x = F.relu(x)
        return x

    def att_conv(self, x, W, b=None):
        batch_size = x.shape[0]
        x_flat = x.view(1, -1, x.shape[-1])
        x = F.conv1d(x_flat, W, bias=None, padding="same", groups=batch_size)
        num_out_channels = W.shape[0] // batch_size
        x = x.view(batch_size, num_out_channels, x.shape[-1])
        if b is not None:
            x = x + b.unsqueeze(-1)
        return x

        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = self.bn3(self.conv3(x))
        # x = F.relu(x + shortcut)


class Attention(nn.Module):
    """
    Partly copied from:
        https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
    """

    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        inner_dim = self.num_heads * head_dim
        self.norm = nn.LayerNorm(self.embed_dim)
        self.to_qkv = nn.Linear(self.embed_dim, 3 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, self.embed_dim, bias=False)

    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: einops.rearrange(
                t, "b n (h d) -> b h n d", h=self.num_heads
            ),
            [q, k, v],
        )
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        attn = F.softmax(attn, dim=-1)
        y = attn @ v
        y = einops.rearrange(y, "b h n d -> b n (h d)")
        y = self.to_out(y)
        return y


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, head_dim, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": Attention(embed_dim, num_heads, head_dim),
                        "ff": nn.Sequential(
                            nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(0.2),
                            nn.Linear(mlp_dim, embed_dim),
                        ),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer["attn"](x) + x
            x = layer["ff"](x) + x
        return x


# TODO: make upsampling modules that can be used interchangably:
#   - interpolation
#   - transposed convolution
#   - pixel shuffle / subpixel convolution
