from typing import TypeVar, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence


# TODO (alexandre.marty, 20201107): Turn this TypeVar into numpy.typing.ArrayLike when
#  it is released.
ArrayLike = TypeVar("ArrayLike", np.ndarray, list, tuple)
Scalar = TypeVar("Scalar", int, float)
SolutionType = TypeVar("SolutionType")

RngSeedType = Union[None, int, ArrayLike, SeedSequence, BitGenerator, Generator]
