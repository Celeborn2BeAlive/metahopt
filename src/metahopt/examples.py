from typing import Union

import numpy as np

from metahopt.typing import ArrayLike, Scalar


def square_norm(x: Union[ArrayLike, Scalar]) -> np.ndarray:
    """Squared distance from 0.

    Args:
        x (np.ndarray): Array of dimension at least 2. Axis 0 holds solutions, other
            axes are for the solution dimensions.

    Returns:
        np.ndarray: 1-dimensional array.
    """
    sum_axes = tuple(range(1, np.ndim(x)))  # All axes but the first
    return np.square(x).sum(axis=sum_axes)


def spherical_sinc(x: Union[ArrayLike, Scalar]) -> np.ndarray:
    sum_axes = tuple(range(1, np.ndim(x)))  # All axes but the first
    return np.sinc(np.sqrt(np.square(x).sum(axis=sum_axes)))


def nsinc(x: Union[ArrayLike, Scalar]) -> np.ndarray:
    sum_axes = tuple(range(1, np.ndim(x)))  # All axes but the first
    return np.sinc(x).sum(axis=sum_axes)
