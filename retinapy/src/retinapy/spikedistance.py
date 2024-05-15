"""
Calculate distance arrays from spikes, and infer spikes from distance arrays.

At this point, the file is a bit on an anachronysm when we called everything
spike distance fields. Rename to spikedistance.py at some point.
"""

import logging
import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


_logger = logging.getLogger(__name__)

MIN_DIST = 0.25


class Vrange:
    """
    V-shape cache (to concatenated aranges, one flipped).
    Example, len(target_dist) = 5
              M
    0 1 2 3 4 5 6 7 8 9 10
    5 4 3 2 1 0 1 2 3 4 5
    """

    def __init__(self, half_len: int, min_val: float, device):
        # Example: half_len = 5
        # [1, 2, 3, 4] -> [4, 3, 2, 1]
        #                            [0, 1, 2, 3, 4]
        # =>
        #  0  1  2  3  4  5  6  7  8]
        # [4, 3, 2, 1, 0, 1, 2, 3, 4]
        # mid = 4
        #
        self._vcache = torch.cat(
            [
                torch.flip(
                    torch.arange(1, half_len, dtype=torch.float, device=device),
                    dims=[0],
                ),
                torch.arange(half_len, dtype=torch.float, device=device),
            ]
        )
        self._invcache_odd = self._vcache.clone() * -1
        self._invcache_even = (
            torch.cat(
                [
                    torch.flip(
                        torch.arange(
                            half_len, dtype=torch.float, device=device
                        ),
                        dims=[0],
                    ),
                    # * -1,
                    torch.arange(
                        half_len - 1, dtype=torch.float, device=device
                    ),
                ]
            )
            * -1
        )
        self._mid = half_len - 1
        self._vcache[self._mid] = min_val

    def v(self, l_inc: int, mid: int, r_exc: int):
        """Slice the V-shaped range."""
        start = self._mid - (mid - l_inc)
        end = self._mid + (r_exc - mid)
        return self._vcache[start:end]

    def inv_odd(self, l_inc: int, mid: int, r_exc: int):
        start = self._mid - (mid - l_inc)
        end = self._mid + (r_exc - mid)
        res = self._invcache_odd[start:end]
        return res

    def inv_even(self, l_inc: int, mid: int, r_exc: int):
        start = self._mid - (mid - l_inc)
        end = self._mid + (r_exc - mid)
        res = self._invcache_even[start:end]
        return res


def distance_arr(spikes: np.ndarray, default_distance: float):
    """
    Calculates the distance array of a spike train.

    Args:
        spikes: a 1D array where a 1 represents a timestep where a spike
            occurred, and 1 where a spike did not occur.
        default_distance: the value to initialize each element of the
            distance array. This functions as a maximum distance. Notably, if
            there are no spikes in the spikes array, then all elements of the
            distance array will be set to this value.
    """
    dist_arr = np.full_like(spikes, default_distance, float)
    spike_indicies = (spikes == 1).nonzero()[0]
    all_indicies = np.arange(len(spikes))
    for s in spike_indicies:
        dist_arr = np.minimum(dist_arr, np.abs(all_indicies - s))
    # If a spike occurs somewhere in a bin, then the distance assigned to that
    # bin is the expected distance from two points drawn from a uniform
    # distribution in [0, 1], which is 0.5.
    # Or, we can take a fixed midpoint, and only calculate the expectation over
    # the spike position, which would give 0.25 (MIN_DIST).
    dist_arr[spikes == 1] = MIN_DIST
    return dist_arr


def distance_arr_torch(spikes: torch.Tensor, default_distance: float):
    dist_arr = torch.full_like(spikes, default_distance, dtype=torch.float)
    spike_indicies = (spikes == 1).nonzero(as_tuple=True)[0]
    all_indicies = torch.arange(len(spikes))
    for s in spike_indicies:
        dist_arr = torch.minimum(dist_arr, torch.abs(all_indicies - s))
    dist_arr[spikes == 1] = MIN_DIST
    return dist_arr


def spike_interval(spikes: np.ndarray, default_count: int):
    """Count the timesteps between spikes"""
    count_arr = np.full_like(spikes, default_count, int)
    spike_indicies = (spikes == 1).nonzero()[0]
    count_arr[spike_indicies] = 0
    counts = np.diff(spike_indicies) - 1
    for idx in range(len(counts)):
        count_arr[spike_indicies[idx] + 1 : spike_indicies[idx + 1]] = counts[
            idx
        ]
    return count_arr


def distance_arr2(spikes, default_dist):
    """An alternative (non-vector) distance array implementation.

    Not used at the moment. Leaving it here for reference.
    """
    dist = [
        default_dist,
    ] * len(spikes)
    spike_indicies = [idx for idx, v in enumerate(spikes) if v == 1]

    def _dfs(idx, cur_dist):
        if dist[idx] <= cur_dist:
            return
        dist[idx] = cur_dist
        if idx > 0:
            _dfs(idx - 1, cur_dist + 1)
        if idx < len(spikes) - 1:
            _dfs(idx + 1, cur_dist + 1)

    for s in spike_indicies:
        _dfs(s, cur_dist=0)
    dist = np.clip(dist, a_min=MIN_DIST, a_max=None)
    return dist


# Below are various attempts at inference.


def quick_inference(dist, target_interval, threshold=0.1):
    dist = dist[:, target_interval[0] : target_interval[1]]
    num_spikes = torch.sum(dist < threshold, dim=1)
    return num_spikes


def quick_inference2(dist, target_interval, threshold=0.1):
    dist = dist[:, target_interval[0] : target_interval[1]]
    kernel = torch.FloatTensor(
        [[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]
    ).to(dist.device)
    smoothed = torch.squeeze(F.conv1d(torch.unsqueeze(dist, 1), kernel))
    below_threshold = (smoothed < threshold).float()
    # below_threshold = below_threshold[:,::3]
    transitions = torch.sum((torch.diff(below_threshold, 1, dim=1) > 0), dim=1)
    return transitions


def mle_inference_via_dp(
    dist: torch.Tensor,
    lhs_spike,
    rhs_spike,
    spike_pad,
    max_clamp=None,
    max_num_spikes=100,
    resolution=1,
):
    if len(dist.shape) != 1:
        raise ValueError("Batching isn't supported (yet)")
    init_a = max(0, lhs_spike + spike_pad + 1)
    init_b = min(len(dist) - 1, rhs_spike - spike_pad - 1)
    max_n = int(math.ceil((init_b - init_a) / (spike_pad + 1)))
    max_n = min(max_n, max_num_spikes)
    _len = len(dist)
    device = dist.device

    # If a-1 and b+1 are the indicies of two spikes, what is the energy
    # contributed by the elements in (a,b)?
    memo = {}  # (a,b, num_allowed) -> ('energy')

    zero_spike_memo = torch.zeros(_len, _len, device=device)
    for i in range(_len):
        for j in range(i, _len):
            d_after = torch.arange(
                j - i + 1, dtype=torch.float32, device=device
            )
            d_before = torch.flip(
                torch.arange(j - i + 1, dtype=torch.float32, device=device),
                dims=(0,),
            )
            # The endpoints need special treatment.
            if i == 0:
                d_after += -lhs_spike
            if j == 0:
                d_before += rhs_spike
            d_min = torch.clamp(torch.minimum(d_after, d_before), max=max_clamp)
            energy = torch.sum(torch.abs(dist[i : j + 1] - d_min))
            zero_spike_memo[i, j] = energy

    global_best_energy = math.inf
    low_energy_positions = dist < 20

    def _dfs(
        a, b, energy_so_far, num_allowed_spikes
    ) -> Tuple[float, Tuple[int, ...]]:
        nonlocal global_best_energy
        if a >= b:
            return 0, ()
        if (a, b, num_allowed_spikes) in memo:
            return memo[(a, b, num_allowed_spikes)]
        if energy_so_far > global_best_energy:
            return math.inf, ()
        no_spike_energy = zero_spike_memo[a, b]
        best_energy = no_spike_energy
        best_seq = ()
        if not num_allowed_spikes:
            return best_energy, best_seq
        for candidate_pos in range(a, b + 1, resolution):
            if not low_energy_positions[candidate_pos]:
                continue
            for num_l_spikes in range(num_allowed_spikes):
                for num_r_spikes in range(num_allowed_spikes - num_l_spikes):
                    lhs_energy, lhs_seq = _dfs(
                        a, candidate_pos - 1, energy_so_far, num_l_spikes
                    )
                    spike_pos_energy = min(dist[candidate_pos], max_clamp)
                    energy = energy_so_far + lhs_energy + spike_pos_energy
                    rhs_energy, rhs_seq = _dfs(
                        candidate_pos + 1, b, energy, num_r_spikes
                    )
                    energy += rhs_energy
                    if energy < best_energy:
                        best_energy = energy
                        best_seq = lhs_seq + (candidate_pos,) + rhs_seq
                        if (best_energy + energy_so_far) < global_best_energy:
                            global_best_energy = best_energy
        memo[(a, b)] = (best_energy, best_seq, num_allowed_spikes)
        return best_energy, best_seq

    e, seq = _dfs(0, _len - 1, energy_so_far=0, num_allowed_spikes=max_n)
    return e, seq


def lhs_spike(in_spikes: torch.Tensor, max_dist: int):
    """
    Args:
        in_spikes: (batch_size, input_len)
    """
    ramp = torch.arange(in_spikes.shape[1]).unsqueeze(0)
    tmp = (in_spikes > 0).int() * ramp
    indices = torch.argmax(tmp, dim=1)
    res = torch.clamp(
        indices - in_spikes.shape[1],
        min=-max_dist,
    )
    return res


class SpikeLinkedList:
    class Node:
        def __init__(self, pos: int, total_len: int, device):
            self.pos = pos
            self.lnode = None
            self.rnode = None
            self.total_len = total_len
            self.device = device
            self.is_removed = False
            self.n_skip = 0
            # A tag to indicate whether this node represents the known
            # historical lhs_spike that is recent enough to have found
            # itself inside the target distance region.
            self.is_lhs_spike = False

        def __str__(self):
            res = (
                f"Node(pos={self.pos}, has lnode? {self.lnode is not None}, "
                f"has rnode? {self.rnode is not None}, "
                f"is_removed={self.is_removed})"
            )
            return res

        def left_resp(self):
            if self.lnode is None:
                left_r = 0
            else:
                # Rounded up integer division by 2.
                # Example:
                #   pos = 10, self.lnode.pos = 5
                #
                #  5 6 7 8 9 10
                #  ^         ^
                #        ^
                left_r = (self.pos + self.lnode.pos + 1) // 2
            assert left_r <= self.pos, (
                "Left responsibility is inclusive, but must be less "
                "than or equal to the position of the node. "
                f"(left_r={left_r}, self.pos={self.pos})"
            )
            left_r = min(left_r, self.total_len)
            return left_r

        def right_resp(self):
            if self.rnode is None:
                right_r = self.total_len
            else:
                # Rounded down integer division by 2.
                # Example:
                #   pos = 5, self.rnode.pos = 10
                #
                #  5 6 7 8 9 10
                #  ^         ^
                #      ^
                # But, we return as exclusive, so +1.
                right_r = 1 + (self.pos + self.rnode.pos) // 2
            right_r = min(right_r, self.total_len)
            return right_r

        def curr_dist(self, vrange: Vrange):
            return vrange.v(self.left_resp(), self.pos, self.right_resp())

        def curr_dist2(self, l_resp, r_resp, vrange: Vrange):
            return vrange.v(l_resp, self.pos, r_resp)

        def dist_on_removal(self, vrange: Vrange):
            l_resp = self.left_resp()
            r_resp = self.right_resp()
            return self.dist_on_removal2(l_resp, r_resp, vrange)

        def dist_on_removal2(self, l_resp, r_resp, vrange: Vrange):
            # If only one node left, return inf.
            if self.lnode is None and self.rnode is None:
                res = torch.full(
                    [self.total_len],
                    fill_value=float("inf"),
                    device=self.device,
                )
            elif self.lnode is None:
                # n: node, r: right node
                #
                #    0 1 2 3 4 5 6 7 8 9
                #          n   ^   r
                #    7 6 5 4 3 2
                #
                start = self.rnode.pos
                # Careful: r_idx is exclusive, but so is `end` param for arange
                # making things cancel to be what we want.
                end = self.rnode.pos - r_resp
                res = torch.arange(start, end, -1, device=self.device)
            elif self.rnode is None:
                # n: node, l: left node
                #
                #    0 1 2 3 4 5 6 7 8 9
                #      l   ^ n
                #          2 3 4 5 6 7 8
                #
                start = l_resp - self.lnode.pos
                end = self.total_len - self.lnode.pos
                res = torch.arange(start, end, device=self.device)
            else:
                new_mid, flat_peak = divmod(self.lnode.pos + self.rnode.pos, 2)
                #    0 1 2 3 4 5 6 7 8 9
                #      l   ^ n   ^     r
                #      l   ^   N ^     r
                #          2 3 4 3
                #
                # flat_peak = True
                #    0 1 2 3 4 5 6 7 8
                #      l   ^ n   ^   r
                #      l   ^   N ^   r
                #          2 3 4 3
                #
                raise_by = new_mid - self.lnode.pos
                if bool(flat_peak):
                    res = vrange.inv_even(l_resp, new_mid, r_resp) + raise_by
                else:
                    res = vrange.inv_odd(l_resp, new_mid, r_resp) + raise_by
            return res

        def set_lnode(self, lnode):
            assert not self.is_removed
            self.lnode = lnode

        def set_rnode(self, rnode):
            assert not self.is_removed
            self.rnode = rnode

    def jitter_scores(self, s, jitter):
        if len(s) == 0:
            return s
        s = s.clone().detach()
        res = s + torch.rand_like(s) * torch.min(s) * jitter
        return res

    def __init__(
        self,
        target_dist: torch.Tensor,
        lhs_spike: int,
        max_dist: int,
        dist_prefix_len: int,
        num_refactory_bins: int,
        ave_dist: Optional[float],
    ):
        if lhs_spike > 0:
            raise ValueError(f"lhs is expected to be negative. ({lhs_spike})")
        lhs_spike = lhs_spike + dist_prefix_len
        self.lhs_spike = lhs_spike
        self.device = target_dist.device
        self.max_dist = max_dist
        self.dist_prefix_len = dist_prefix_len
        self.target_dist = target_dist
        self.output_len = len(target_dist) - dist_prefix_len
        # If a bin contains a spike, and it's location relative to the center
        # of the bin is uniformly distributed over [-0.5, 0.5], then the
        # expected value of the distance is 0.25.
        self.min_dist = MIN_DIST
        self.vrange_len = len(target_dist) * 3

        self.vrange = Vrange(
            self.vrange_len, min_val=self.min_dist, device=self.device
        )
        # The initial distance array is initialized in two ways, depending on
        # whether the lhs_spike is positive or negative. Positive means it's
        # within the time period covered by the target distance array. This
        # occurs due to the offset of the target distance (offset back in time).
        if self.lhs_spike >= 0:
            self.empty_dist = torch.clamp(
                self.vrange.v(0, self.lhs_spike, len(self.target_dist)),
                max=self.max_dist,
            )
        else:
            self.empty_dist = torch.clamp(
                torch.arange(len(target_dist), device=self.device)
                - self.lhs_spike
                + self.min_dist,
                max=self.max_dist,
            )
        assert self.empty_dist.min() >= self.min_dist
        self.num_refactory_bins = num_refactory_bins
        self.ave_dist = ave_dist

        # Create linked list
        init_scores = target_dist.clone()
        # When the dist arr is very flat, don't always put a spike at
        # the beginning. Achieve this easily by adding slight jitter to
        # the starting inspection order.
        jitter = 0.05
        init_scores = self.jitter_scores(init_scores, jitter).cpu().tolist()
        i_pos = self.dist_prefix_len
        self.scores = []
        if lhs_spike >= 0:
            self.first_node = self.Node(
                lhs_spike, len(target_dist), self.device
            )
            self.first_node.is_lhs_spike = True
        else:
            self.first_node = self.Node(i_pos, len(target_dist), self.device)
            self.scores.append((init_scores[i_pos], self.first_node))
            i_pos += 1
        self.last_node = self.first_node
        # Note here that num nodes is 1, even if the 1 node is the fixed
        # lhs_spike node. Therefore, don't use num_nodes to determine if there
        # are no spikes predicted.
        self.num_nodes = 1
        for i_pos in range(i_pos, len(target_dist)):
            node = self.Node(i_pos, len(target_dist), self.device)
            self.last_node.set_rnode(node)
            node.set_lnode(self.last_node)
            self.last_node = node
            self.num_nodes += 1
            self.scores.append((init_scores[i_pos], node))

        # Add on some sparse end nodes that allow the inference to place
        # spikes further into the future. The purpose of this is to not
        # force the inference to place sub-optimal spikes in the target inference
        # window if later spikes would better fit the distance array.
        # The larger END_NODE_START, the less of an effect, as the MAX_DIST
        # takes effect. At 100, there is very little effect. It is worth
        # investigating whether adding denser and closer end nodes can improve
        # inference.
        END_NODE_START = 100
        END_NODE_INTERVAL = 20
        end_node_pos = range(
            len(target_dist) + END_NODE_START,
            self.vrange_len,
            END_NODE_INTERVAL,
        )
        for idx, pos in enumerate(end_node_pos):
            node = self.Node(pos, len(target_dist), self.device)
            self.last_node.set_rnode(node)
            node.set_lnode(self.last_node)
            self.last_node = node
            self.num_nodes += 1
            score = -idx  # Avoid considering these nodes for removal early.
            self.scores.append((score, node))

    def remove(self, node):
        if node is None:
            raise ValueError("Called remove(None)")
        if node.lnode is None:
            # Removing the first node
            self.first_node = node.rnode
        else:
            node.lnode.rnode = node.rnode
        if node.rnode is None:
            # Removing the last node
            self.last_node = node.lnode
        else:
            node.rnode.lnode = node.lnode
        self.num_nodes -= 1
        node.is_removed = True
        return node.rnode

    def should_remove(self, node):
        assert not node.is_removed
        l_resp = node.left_resp()
        r_resp = node.right_resp()
        curr_dist = node.curr_dist2(l_resp, r_resp, self.vrange)
        alternative_dist = node.dist_on_removal2(l_resp, r_resp, self.vrange)
        assert (
            curr_dist.shape == alternative_dist.shape
        ), f"{curr_dist.shape} != {alternative_dist.shape}"
        # Need to clamp the alternative every time.
        alternative_dist = torch.minimum(
            alternative_dist, self.empty_dist[l_resp:r_resp]
        )
        CONFIDENCE_MAX = 1.0
        CONFIDENCE_MIN = 0.3
        confidence = torch.linspace(
            CONFIDENCE_MAX,
            CONFIDENCE_MIN,
            len(self.target_dist) + 1,
            device=self.device,
        )[l_resp:r_resp]
        curr_dist = torch.minimum(curr_dist, self.empty_dist[l_resp:r_resp])
        curr_loss = self.loss(
            curr_dist, self.target_dist[l_resp:r_resp], confidence
        )
        alternative_loss = self.loss(
            alternative_dist,
            self.target_dist[l_resp:r_resp],
            confidence,
        )
        res = alternative_loss < curr_loss
        loss_dec_on_removal = curr_loss - alternative_loss
        return res, loss_dec_on_removal

    def loss(self, pdist, tdist, confidence):
        """
        The quantity to reduce to by inserting/removing spikes.

        The confidence scaling is optional. If you are predicting distance
        arrays that extend into the future, you may be more confident about
        the near future than the far future. In that case, you can use the
        confidence scaling to weight the loss. For short predictions, this
        doesn't matter much, but if predicting quite far into the future, say
        >100 ms, then it is worth scaling the effect of the distance array so
        that our stronger predictions have more sway. These are hyperparameters
        that are worth investigating further, although preliminary tests
        suggest they are not very sensitive.
        """
        res = F.mse_loss(pdist, tdist, reduction="none")
        res = torch.mean(res * confidence)
        return res

    def reduce(self) -> int:
        new_scores = []
        nodes = []
        # The score is the potential loss decrease on removal.
        # Higher means more profitable to be removed.
        for s, n in sorted(self.scores, key=lambda s: -s[0]):
            should_remove, loss_dec_on_removal = self.should_remove(n)
            if should_remove:
                self.remove(n)
            else:
                new_scores.append((loss_dec_on_removal, n))
                nodes.append(n)
            n = n.rnode
        self.scores = new_scores
        return self.num_nodes

    def refactory_remove(self):
        prev_num_nodes = self.num_nodes
        # Iterate nodes
        node = self.first_node
        while node is not None and node.rnode is not None:
            violates_refactor = (
                node.rnode.pos - node.pos <= self.num_refactory_bins
            )
            if violates_refactor:
                self.remove(node.rnode)
                # Don't do the following:
                #   next_node = node.rnode.rnode
                # As we want to stay on the current node until it's refactory
                # period is clear of any subsequent spikes.
            else:
                node = node.rnode
        _logger.debug(
            f"Refactory removed {prev_num_nodes - self.num_nodes} nodes"
        )

    def spike_indicies(self):
        node = self.first_node
        res = []
        while node is not None:
            res.append(node.pos)
            node = node.rnode
        return res

    def is_everywhere_high(self, threshold=80):
        """
        Returns true if the distance array is everywhere high.
        """
        res = torch.all(self.target_dist > threshold)
        return res

    def _infer_spikes(self):
        use_shortcut = False
        if use_shortcut and self.is_everywhere_high():
            # This speeds up inference, but it _does_ effect results, so it is
            # only used while experimenting and not for final results.
            return torch.zeros_like(self.target_dist)[self.dist_prefix_len :]
        prev_num_spikes = self.num_nodes
        halt = False
        n_nodes_processed = []
        while not halt:
            num_spikes = self.reduce()
            if num_spikes == prev_num_spikes:
                halt = True
            n_nodes_processed.append(prev_num_spikes)
            prev_num_spikes = num_spikes
        spike_indicies = self.spike_indicies()
        assert len(spike_indicies) == prev_num_spikes
        if self.num_refactory_bins > 0:
            self.refactory_remove()
        spike_indicies = [
            s for s in self.spike_indicies() if s < len(self.target_dist)
        ]
        spikes = torch.zeros_like(self.target_dist)
        spikes[spike_indicies] = 1
        # Cut off the prefix, for which we won't make predictions.
        # There could be a spike in the prefix representing the lhs spike.
        assert (
            torch.sum(spikes[0 : self.dist_prefix_len]) <= 1
        ), "Prefix should at most 1 spike, representing the lhs spike."
        spikes = spikes[self.dist_prefix_len :]
        return spikes, n_nodes_processed

    def infer_spikes(self):
        spikes, _ = self._infer_spikes()
        return spikes


@torch.no_grad()
def predict(
    dist,
    lhs_spike: int,
    max_dist: int,
    dist_prefix_len: int,
    refactory: int = 0,
    ave_dist: Optional[float] = None,
):
    pred_len = len(dist) - dist_prefix_len
    if pred_len <= 0:
        raise ValueError(
            f"dist_prefix_len ({dist_prefix_len}) too large for distance "
            f"array length ({len(dist)})."
        )
    if len(dist.shape) > 1:
        raise ValueError("Batches not supported (no batch dim accepted)")
    ll = SpikeLinkedList(
        dist, lhs_spike, max_dist, dist_prefix_len, refactory, ave_dist
    )
    res = ll.infer_spikes()
    return res


def predict_sum(
    dist,
    lhs_spike: int,
    max_dist,
    dist_prefix_len: int,
    refactory: int = 0,
    eval_start: int = 0,
    eval_len: Optional[int] = None,
) -> int:
    eval_len = eval_len if eval_len is not None else len(dist)
    if eval_len + eval_start > len(dist):
        raise ValueError(
            f"eval_len + eval_start ({eval_len + eval_start}) "
            f"> len(dist) ({len(dist)})"
        )
    mle_spikes = predict(
        dist,
        lhs_spike,
        max_dist,
        dist_prefix_len,
        refactory=refactory,
    )
    res = mle_spikes[eval_start : eval_start + eval_len].sum()
    return round(res.item())
