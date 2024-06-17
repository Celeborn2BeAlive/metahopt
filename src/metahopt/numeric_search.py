from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from metahopt.local_search import LocalSearch, LocalSearchResults, LocalSearchState
from metahopt.typing import Solution


BoundT = float | int | None
BoundsT = np.ndarray | tuple[BoundT, BoundT] | tuple[list[BoundT], list[BoundT]]


def make_bounds(bounds: BoundsT) -> NDArray[np.float64]:
    bounds = np.array(bounds)
    if bounds.shape[0] != 2:  # noqa: PLR2004
        msg = "bounds' first dimension must have length 2"
        raise ValueError(msg)

    bounds[0] = np.where(np.equal(bounds[0], None), -np.inf, bounds[0])
    bounds[1] = np.where(np.equal(bounds[1], None), np.inf, bounds[1])
    bounds = bounds.astype(np.float64)
    if not np.all(bounds[0] <= bounds[1]):
        msg = "bounds must be ordered"
        raise ValueError(msg)
    return bounds


@dataclass(kw_only=True)
class NumericLocalSearch(LocalSearch[np.ndarray]):
    bounds: BoundsT = (None, None)

    def __post_init__(self):
        super().__post_init__()
        self.bounds = make_bounds(self.bounds)

    def solve(self, starting_point: ArrayLike) -> LocalSearchResults[np.ndarray]:
        starting_point = np.array(starting_point)
        if np.any((starting_point < self.bounds[0]) | (starting_point > self.bounds[1])):
            msg = "starting_point must be within bounds"
            raise ValueError(msg)
        return super().solve(starting_point)


@dataclass(kw_only=True)
class IntCoordinateSearch(NumericLocalSearch[NDArray[np.int_]]):
    max_stalled_iter: int = field(default=1, init=False)

    def poll_set_vectorized(self, state: LocalSearchState) -> NDArray[np.int_]:
        gen_mat = np.eye(state.best_solution.shape[0], dtype=np.int_)
        poll_set = np.concatenate(
            [state.best_solution + gen_mat, state.best_solution - gen_mat]
        )
        in_bounds = (self.bounds[0] <= poll_set) & (poll_set <= self.bounds[1])
        return poll_set[np.all(in_bounds, axis=-1)]

    def solve(self, starting_point: ArrayLike) -> LocalSearchResults[NDArray[np.int_]]:
        starting_point_int = np.array(starting_point, dtype=np.int_)
        if not np.equal(starting_point, starting_point_int).all():
            msg = "starting_point values must be integers"
            raise ValueError(msg)
        return super().solve(starting_point)
