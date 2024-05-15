import pytest
import retinapy.spikedistance as sdf
import numpy as np
import torch
import itertools


@pytest.fixture
def spike_data1():
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    MAX_DIST = 6
    M = MAX_DIST  # Used to make the below array literal tidier.

    after_field = np.array(
        [
            [0, 1, 2, 3, 4, 5, M, M, M, 0],
            [M, M, 0, 1, 2, 3, 4, 5, M, M],
            [M, M, M, M, M, M, M, 0, 1, 2],
            [M, 0, 1, 0, 1, 2, 3, 0, 1, 2],
            [M, 0, 0, 0, 1, 2, 3, 4, 5, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    before_field = np.array(
        [
            [0, M, M, M, 5, 4, 3, 2, 1, 0],
            [2, 1, 0, M, M, M, M, M, M, M],
            [M, M, 5, 4, 3, 2, 1, 0, M, M],
            [1, 0, 1, 0, 3, 2, 1, 0, M, M],
            [1, 0, 0, 0, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    return (spike_batch, MAX_DIST, before_field, after_field)


@pytest.fixture
def spike_data2():
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )  # just for reference

    MAX_DIST = 100
    M = MAX_DIST  # Used to make the below array literal tidier.

    after_field = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
            [M, M, 0, 1, 2, 3, 4, 5, 6, 7],
            [M, M, M, M, M, M, M, 0, 1, 2],
            [M, 0, 1, 0, 1, 2, 3, 0, 1, 2],
            [M, 0, 0, 0, 1, 2, 3, 4, 5, 6],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    before_field = np.array(
        [
            [0, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [2, 1, 0, M, M, M, M, M, M, M],
            [7, 6, 5, 4, 3, 2, 1, 0, M, M],
            [1, 0, 1, 0, 3, 2, 1, 0, M, M],
            [1, 0, 0, 0, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )

    spike_counts = np.array([2, 1, 1, 3, 3, 0])

    return (spike_counts, MAX_DIST, before_field, after_field)


@pytest.fixture
def distance_field_data():
    MAX_DIST = 100
    M = MAX_DIST  # Used to make the below array literal tidier.
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    distance_field = np.clip(
        np.array(
            [
                [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
                [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
                [7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
                [1, 0, 1, 0, 1, 2, 1, 0, 1, 2],
                [1, 0, 0, 0, 1, 2, 3, 4, 5, 6],
                [M, M, M, M, M, M, M, M, M, M],
            ],
            dtype=float,
        ),
        a_min=0.25,
        a_max=None,
    )
    return MAX_DIST, spike_batch, distance_field


def test_distance_field(distance_field_data):
    M, spikes, dist_fields = distance_field_data
    for spike, known_df in zip(spikes, dist_fields):
        dist_field = sdf.distance_arr(spike, M)
        dist_field_cpu = sdf.distance_arr2(spike, M)
        assert np.array_equal(known_df, dist_field)
        assert np.array_equal(dist_field_cpu, dist_field)


@pytest.fixture
def spike_interval_data():
    MAX_COUNT = 100
    M = MAX_COUNT  # Used to make the below array literal tidier.
    spike_batch = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    spike_intervals = np.array(
        [
            [0, 8, 8, 8, 8, 8, 8, 8, 8, 0],
            [M, M, 0, M, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, 0, M, M],
            [M, 0, 1, 0, 3, 3, 3, 0, M, M],
            [M, 0, 0, 0, M, M, M, M, M, M],
            [M, M, M, M, M, M, M, M, M, M],
        ]
    )
    return MAX_COUNT, spike_batch, spike_intervals


def test_spike_interval(spike_interval_data):
    M, spikes, spike_intervals = spike_interval_data
    for spike, known_si in zip(spikes, spike_intervals):
        si = sdf.spike_interval(spike, M)
        assert np.array_equal(known_si, si)


def test_mle_inference_from_df(distance_field_data):
    M, spikes, dist_fields = distance_field_data
    for spike, dist_field in zip(spikes, dist_fields):
        num_spikes = sdf.mle_inference_via_dp(
            torch.Tensor(dist_field),
            lhs_spike=-M,
            rhs_spike=len(spike) + M - 1,
            spike_pad=1,
            max_clamp=M * 2,
            max_num_spikes=5,
            resolution=1,
        )
    # TODO


def test_predict():
    dist_len = 128
    lhs_spikes = -np.arange(1, 451, 90)
    max_dists = [100, 150, 300, 500]
    prefix_lens = [0, 1, 2, 3, 10, 17, 50]
    spike_idxs = np.arange(0, dist_len)

    for lhs_spike, max_dist, prefix_len, spike_idx in itertools.product(
        lhs_spikes, max_dists, prefix_lens, spike_idxs
    ):
        spikes = torch.zeros(dist_len)
        if spike_idx + prefix_len < dist_len:
            spikes[spike_idx+prefix_len] = 1
        if lhs_spike + prefix_len >= 0:
            spikes[lhs_spike + prefix_len] = 1
        actual_dist = sdf.distance_arr_torch(spikes, max_dist)

        # Test
        pred_spikes = sdf.predict(
            actual_dist, lhs_spike, max_dist, prefix_len
        )
        correct_spikes = spikes[prefix_len:]
        np.testing.assert_array_equal(
            correct_spikes,
            pred_spikes,
            err_msg=f"lhs_spike={lhs_spike}, max_dist={max_dist}, "
            f"dist_prefix_len={prefix_len}, spike_idx={spike_idx}",
        )


def test_predict_ghostspike1_case():
    """
    Tests the predict() function in the case where a past spike results in
    a distance field with a dip. This dip should not cause the prediction of
    a future spike.

    This test was created in response to a real encountered issue.

    Refer to notebook/pytest_companion/test_spikedistancefield.ipynb for
    a visualization and further details.
    """
    # Setup
    model_out = torch.tensor(
        [5.64485312,  5.64290333,  5.6094594 ,  5.55425596,  5.48242569,
        5.39028597,  5.27889872,  5.15175056,  5.01009417,  4.85483742,
        4.68749237,  4.50965595,  4.32297373,  4.12901211,  3.92946959,
        3.72606754,  3.5204196 ,  3.3617847 ,  3.24470615,  3.16525722,
        3.12083793,  3.10999751,  3.13238764,  3.18873   ,  3.28085375,
        3.40291262,  3.55803418,  3.75028539,  3.9848597 ,  4.26830482,
        4.60883188,  5.01666117,  8.19895363,  8.86996269,  9.56994057,
       10.2973423 , 11.05019665, 11.82625294, 12.6228447 , 13.4369669 ,
       14.26522923, 15.1179142 , 15.99336338, 16.88964653, 17.80462837,
       18.73593521, 19.68094063, 20.63680267, 21.60055351, 22.56772614,
       23.53541756, 24.50022507, 25.45860672, 26.40710831, 27.34199905,
       28.25960159, 29.15610886, 29.99212074, 30.7609787 , 31.45633125,
       32.07235718, 32.60315704, 33.04401398, 33.39071655, 33.63959122,
       33.89370728, 34.15212631, 34.41485977, 34.68213272, 34.95448685,
       35.23212433, 35.51521301, 35.8038826 , 36.0922966 , 36.38035583,
       36.66743851, 36.95330811, 37.23734665, 37.51920319, 37.79896927,
       38.07702255, 38.35443497, 38.63191605, 38.91067505, 39.19100571,
       39.47293854, 39.75658417, 40.04223251, 40.32990265, 40.63016891,
       40.94324493, 41.26902008, 41.60762405, 41.95845032, 42.32147217,
       42.69570923, 43.08057785, 43.48801804, 43.9185257 , 44.37303162,
       44.85240936, 45.35910416, 45.89411545, 46.4590416 , 47.05488205,
       47.73072433, 48.48965454, 49.33524704, 50.2714119 , 51.30131149,
       52.42959595, 53.6602211 , 55.00008011, 56.24311066, 57.38219833,
       58.40987015, 59.3198204 , 60.10715866, 60.76706696, 61.29632187,
       61.6914711 , 62.03909302, 62.33841705, 62.58457947, 62.77700424,
       62.92211533, 63.07161713, 63.13805771])
    lhs_spike = -8
    dist_prefix_len = 32
    max_dist = 200


    # Test
    pred_spikes = sdf.predict(model_out, lhs_spike, max_dist=max_dist,
                              dist_prefix_len=dist_prefix_len, refactory=0)

    
    # This test is currently failing. One approach is to take the log of the
    # distances before calculating the error. However, it seems like a better
    # approach would be to create a better estimate of the distance array in
    # the absense of a spike in the prediction area. For example, by better 
    # placing future spikes that reflect the average spike rate of a cell.
    assert torch.all(pred_spikes == 0), "Known failing test case."


@pytest.mark.parametrize("total_len", [1, 2, 3, 4, 5, 977, 1000])
def test_SpikeLinkedList_Node_resp(total_len):
    """
    Test left_resp() and right_resp() methods of the SpikeLinkedList.Node class.

    Visualization to help understand the context:

        0 1 2 3 4 5 6 7 8 9
          ^           ^
          a a a a
                b b b b

        0 1 2 3 4 5 6 7 8 9
          ^         ^
          a a a
                b b b

    Test that:
        1. If the node is the head of the list, then the left_resp() method
            returns 0.
        2. If the node is the tail of the list, then the right_resp() method
            returns the total length of the list.
        3. Even distances between nodes are handled correctly.
            - Invariant
              There will be an odd number of positions between the nodes,
              so the they will share a position. As right responsibility uses
              exclusive bounds, the left node's right_resp() will be
              one more than the right_node's left_resp().
        4. Odd distances between nodes are handled correctly.
            - Invariant
              The left node's right_resp() will be the same as the right_node's
              left_resp().
    """
    # Setup
    seed = 123
    rng = np.random.default_rng(seed)
    num_variants = min(total_len * 5, 500)
    device = torch.device("cpu")

    # Test 1 & 2
    p1p2 = np.sort(rng.integers(0, total_len, size=(num_variants, 2)), axis=1)
    for left_p, right_p in p1p2:
        head = sdf.SpikeLinkedList.Node(
            pos=left_p, total_len=total_len, device=device
        )
        tail = sdf.SpikeLinkedList.Node(
            pos=right_p, total_len=total_len, device=device
        )
        mid_pos = (left_p + right_p) // 2
        mid = sdf.SpikeLinkedList.Node(
            pos=mid_pos, total_len=total_len, device=device
        )
        head.rnode = mid
        mid.lnode = head
        mid.rnode = tail
        tail.lnode = mid
        # Test 1
        assert head.left_resp() == 0
        # Test 2
        assert tail.right_resp() == total_len

    # Test 3 & 4
    p1p2 = np.sort(rng.integers(0, total_len, size=(num_variants, 2)), axis=1)
    for left_p, right_p in p1p2:
        lnode = sdf.SpikeLinkedList.Node(
            pos=left_p, total_len=total_len, device=device
        )
        rnode = sdf.SpikeLinkedList.Node(
            pos=right_p, total_len=total_len, device=device
        )
        lnode.rnode = rnode
        rnode.lnode = lnode
        if (right_p - left_p) % 2 == 1:
            # Test 3
            # E.g. 5 6 7 8 9 10
            # 10 - 5 = 5, which is odd, and there is even middle positions.
            # 5's right_resp() should be 8 exclusive, and 10's left_resp()
            # should be 8 inclusive.
            assert lnode.right_resp() == rnode.left_resp()
        else:
            # Test 4
            # E.g. 5 6 7 8 9
            # 9 - 5 = 4, which is even, and there is odd middle positions.
            # 5's right_resp() should be 8 exclusive, and 9's left_resp()
            # should be 7 inclusive.
            assert lnode.right_resp() == rnode.left_resp() + 1


def test_SpikeLinkedList_Node_curr_dist():
    """Tests Node's curr_dist() method.

    This test runs through some hard-coded examples.

    Tests that:
        1. result is [min_val] when a node has only 1 position in its
            responsibility.
    """
    # Setup
    device = torch.device("cpu")
    total_len = 1000
    min_val = 0.25
    vrange = sdf.Vrange(total_len, min_val, device)

    def create_nodes(points):
        nodes = []
        for p in points:
            nodes.append(sdf.SpikeLinkedList.Node(p, total_len, device))

        def connect(nA, nB):
            nA.rnode = nB
            nB.lnode = nA

        for i in range(len(nodes) - 1):
            connect(nodes[i], nodes[i + 1])

        return nodes

    # Test 1
    #
    # n0, n1 and n2 should have a curr_dist() of [min_val]
    #
    #  0  1  2  3
    #  ^  ^  ^  ^
    # n0 n1 n2 n3
    #
    def test1():
        n0, n1, n2, n3 = create_nodes([0, 1, 2, 3])
        torch.testing.assert_close(
            n0.curr_dist(vrange), torch.tensor([min_val])
        )
        torch.testing.assert_close(
            n1.curr_dist(vrange), torch.tensor([min_val])
        )
        torch.testing.assert_close(
            n2.curr_dist(vrange), torch.tensor([min_val])
        )

    test1()


def test_SpikeLinkedList_Node_dist_on_removal():
    """Tests Node's dist_on_removal() method.

    This test runs through some hard-coded examples.

    Tests that:
        1.  Removing a middle node. The easiest case.
    """
    # Setup
    device = torch.device("cpu")
    total_len = 1000
    min_val = 0.25
    vrange = sdf.Vrange(total_len, min_val, device)

    def create_nodes(points):
        nodes = []
        for p in points:
            nodes.append(sdf.SpikeLinkedList.Node(p, total_len, device))

        def connect(nA, nB):
            nA.rnode = nB
            nB.lnode = nA

        for i in range(len(nodes) - 1):
            connect(nodes[i], nodes[i + 1])

        return nodes

    # Test 1.1
    # Remove a node with only 1 position in its responsibility.
    #
    # Initial state:
    #
    #  0  1  2  3
    #     ^
    # n0 n1 n2 n3
    #
    # Remove n1 (or n2):
    #
    #  0  1  2  3
    #     ^
    # n0    n2 n3
    #
    def test1():
        n0, n1, n2, n3 = create_nodes([0, 1, 2, 3])
        n1_dist = n1.dist_on_removal(vrange)
        n2_dist = n2.dist_on_removal(vrange)
        expected_dist = torch.tensor([1.0])
        torch.testing.assert_close(n1_dist, expected_dist)
        torch.testing.assert_close(n2_dist, expected_dist)

    test1()

    # Test 1.2
    # Remove a node with 2 positions in its responsibility.
    #
    # Initial state:
    #
    #  0  1  2  3
    #     ^  ^
    # n0 n1    n2
    #
    # Remove n1:
    #
    #  0  1  2  3
    #     ^  ^
    # n0       n2
    #
    def test2():
        n0, n1, n2 = create_nodes([0, 1, 3])
        n1_dist = n1.dist_on_removal(vrange)
        expected_dist = torch.tensor([1.0, 1.0])
        print(n1_dist)
        torch.testing.assert_close(n1_dist, expected_dist)

    test2()

    # Test 1.3
    # Remove a node with 3 positions in its responsibility.

