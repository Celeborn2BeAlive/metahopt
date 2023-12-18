from abc import ABCMeta
from collections.abc import Callable, Iterable, Sequence, Sized
from typing import Generic, TypeVar

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence


# TODO (alexandre.marty, 20201107): Turn this TypeVar into numpy.typing.ArrayLike when
#  it is released.
ArrayLike = TypeVar("ArrayLike", np.ndarray, list, tuple)
Scalar = TypeVar("Scalar", int, float)

Solution = TypeVar("Solution", bound=object)
ObjectiveFunc = Callable[[Solution], float]
VectorizedObjectiveFunc = Callable[[Iterable[Solution]], Sequence[float]]
RngSeed = None | int | ArrayLike | SeedSequence | BitGenerator | Generator


_T = TypeVar("_T")


class SizedIterable(Iterable[_T], Sized, Generic[_T], metaclass=ABCMeta):
    ...
