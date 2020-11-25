import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from metahopt.examples import nsinc, spherical_sinc, square_norm


def test_square_norm():
    # 1d
    assert_array_equal(square_norm([2]), [4])
    assert_array_equal(square_norm([1, 2, 3]), [1, 4, 9])
    # 2d
    assert_array_equal(square_norm([[1, 2], [3, 4]]), [5, 25])
    # 3d
    assert_array_equal(square_norm([[[1, 2], [3, 4]], [[0, 3], [1, 4]]]), [30, 26])


def test_spherical_sinc():
    s = np.sinc(1)
    # 1d
    assert_allclose(spherical_sinc([2]), -s)
    assert_allclose(spherical_sinc([1, 2, 3]), [s, -s, s])
    # 2d
    assert_allclose(spherical_sinc([[1, 2], [3, 4]]), [np.sinc(np.sqrt(5)), s])
    # 3d
    assert_allclose(
        spherical_sinc([[[1, 2], [3, 4]], [[0, 3], [1, 4]]]),
        [np.sinc(np.sqrt(30)), np.sinc(np.sqrt(26))],
    )


def test_nsinc():
    s = np.sinc(1)
    # 1d
    assert_allclose(nsinc([2]), -s)
    assert_allclose(nsinc([1, 2, 3]), [s, -s, s])
    # 2d
    assert_allclose(nsinc([[1, 2], [3, 5]]), [0, 2 * s])
    # 3d
    assert_allclose(nsinc([[[1, 2], [3, 4]], [[1, 3], [5, 4]]]), [0, 2 * s])
