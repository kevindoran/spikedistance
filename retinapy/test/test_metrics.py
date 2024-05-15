import pytest
import numpy as np

import retinapy.metrics as metrics

def test_van_rossum():
    """
    Tests the van rossum distance.
    """
    signal_a = np.zeros(1000)
    signal_b = np.zeros(1000)

    a_idx = 500
    b_idx = 510
    signal_a[a_idx] = 1
    signal_b[b_idx] = 1

    res = metrics.van_rossum(signal_a, signal_b, bin_ms=1, tau_ms=5)
    print(res)





