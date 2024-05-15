"""
A few figures try to map as faithfully as possible between the LED values and 
the colors that would be seen by a human. This is done for visualization
purposes only. This module does the mapping.
"""

import importlib.resources
import json
import numpy as np


# Load color matching data.
# Reference:
# https://github.com/wimglenn/resources-example/blob/master/myapp/example4.py
with importlib.resources.open_text("retinapy.pkdata", "judd_vos_cmf.json") as f:
    judd_vos_cmf = json.load(f)


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """Converts XYZ to sRGB.

    Args:
        xyz (np.ndarray): A sequence of XYZ values, range [0, 1]. The shape of
            the array is (N, 3), where N is the number of frames in the
            stimulus.

    Returns:
        np.ndarray: sRGB values, range [0, 1].
    """
    if len(xyz.shape) != 2 and xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    mat = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    ).T
    rgb = xyz @ mat

    # Gamma correction
    def gamma_fn(x):
        if x <= 0.0031308:
            return 12.92 * x
        else:
            return 1.055 * np.power(x, 1 / 2.4) - 0.055
    rgb = np.vectorize(gamma_fn)(rgb)
    # Get errors in power due to >=0 values ( I think).
    # rgb = np.where(
    #     rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055
    # )

    # Clamp values to [0, 1]
    rgb = np.clip(rgb, 0, 1)
    return rgb


def to_xyz(rgbv: np.ndarray) -> np.ndarray:
    """Converts a sequence of LED on/off values to XYZ.

    Args:
        rgbv (np.ndarray): A sequence of LED on/off values. The shape is
            (N, 4), where N is the number of frames in the stimulus.

    This needs to be able to handle different LED wavelengths and powers
    at some point.

    Returns:
        np.ndarray: XYZ values.
    """
    if len(rgbv.shape) != 2 and rgbv.shape[1] != 4:
        raise ValueError("rgbv must have shape (N, 4).")
    # A 4x3 matrix. stimulus of shape (N, 4) multiplies like so:
    # stimulus @ mat to get XYZ.
    mat = np.array([
            # Red
            np.array(judd_vos_cmf["635"]) * 100 / 4 +
            np.array(judd_vos_cmf["630"]) * 100 / 2 +
            np.array(judd_vos_cmf["625"]) * 100 / 4,
            # Green
            np.array(judd_vos_cmf["510"]) * 65 / 4 +
            np.array(judd_vos_cmf["505"]) * 65 / 2 +
            np.array(judd_vos_cmf["500"]) * 65 / 4,
            # Blue
            np.array(judd_vos_cmf["485"]) * 60 / 4 +
            np.array(judd_vos_cmf["480"]) * 60 / 2 +
            np.array(judd_vos_cmf["475"]) * 60 / 4,
            # UV
            np.array(judd_vos_cmf["425"]) * 65 / 4 +
            np.array(judd_vos_cmf["420"]) * 65 / 2 +
            np.array(judd_vos_cmf["415"]) * 65 / 4,
        ])
    res = rgbv @ mat
    return res


def stim_to_srgb(rgbv: np.ndarray) -> np.ndarray:
    xyz = to_xyz(rgbv)
    xyzWhite = to_xyz(np.array([[1, 1, 1, 1]]))[0];
    xyz_norm = xyz / xyzWhite[1]
    rgb = xyz_to_srgb(xyz_norm)
    return rgb
