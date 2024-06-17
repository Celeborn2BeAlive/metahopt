"""Examples of objective functions.
"""

import numpy as np

from metahopt.typing import ArrayLike


# Numerical problems


def square_norm(x: ArrayLike) -> np.ndarray:
    """Squared distance from 0, vectorized on axis 0.

    Args:
        x (ArrayLike): Array of dimension at least 1, of shape (n_sol, *). Axis 0 holds
            solutions, other axes are for the solution dimensions.

    Returns:
        np.ndarray: 1-dimensional array of shape (n_sol,).
    """
    sum_axes = tuple(range(1, np.ndim(x)))  # All axes but the first
    return np.square(x).sum(axis=sum_axes)


def spherical_sinc(x: ArrayLike) -> np.ndarray:
    """Sine cardinal of distance to 0, vectorized on axis 0.

    Args:
        x (ArrayLike): Array of dimension at least 1, of shape (n_sol, *). Axis 0 holds
            solutions, other axes are for the solution dimensions.

    Returns:
        np.ndarray: 1-dimensional array of shape (n_sol,).
    """
    sum_axes = tuple(range(1, np.ndim(x)))  # All axes but the first
    return np.sinc(np.sqrt(np.square(x).sum(axis=sum_axes)))


def nsinc(x: ArrayLike) -> np.ndarray:
    """N-dimensional sine cardinal, vectorized on axis 0, summed over all other axes.

    Args:
        x (ArrayLike): Array of dimension at least 1, of shape (n_sol, *). Axis 0 holds
            solutions, other axes are for the solution dimensions.

    Returns:
        np.ndarray: 1-dimensional array of shape (n_sol,).
    """
    sum_axes = tuple(range(1, np.ndim(x)))  # All axes but the first
    return np.sinc(x).sum(axis=sum_axes)
