from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from metahopt.local_search import LocalSearch, LocalSearchResults, LocalSearchState
from metahopt.typing import Solution


BoundT = float | int | None
BoundsT = np.ndarray | tuple[BoundT, BoundT] | tuple[list[BoundT], list[BoundT]]


def make_bounds(bounds: BoundsT) -> NDArray[np.float_]:
    bounds = np.array(bounds)
    if bounds.shape[0] != 2:  # noqa: PLR2004
        msg = "bounds' first dimension must have length 2"
        raise ValueError(msg)

    bounds[0] = np.where(np.equal(bounds[0], None), -np.inf, bounds[0])
    bounds[1] = np.where(np.equal(bounds[1], None), np.inf, bounds[1])
    bounds = bounds.astype(np.float_)
    if not np.all(bounds[0] <= bounds[1]):
        msg = "bounds must be ordered"
        raise ValueError(msg)
    return bounds


@dataclass(kw_only=True)
class NumericLocalSearch(LocalSearch):
    bounds: BoundsT = (None, None)

    def __post_init__(self):
        super().__post_init__()
        self.bounds = make_bounds(self.bounds)

    def solve(self, starting_point: Solution) -> LocalSearchResults[Solution]:
        starting_point = np.array(starting_point)
        return super().solve(starting_point)


@dataclass(kw_only=True)
class IntCoordinateSearch(NumericLocalSearch):
    max_stalled_iter: int = field(default=1, init=False)

    def poll_set_vectorized(self, state: LocalSearchState) -> np.ndarray:
        gen_mat = np.eye(state.best_solution.shape[0], dtype=np.int_)
        poll_set = np.concatenate(
            [state.best_solution + gen_mat, state.best_solution - gen_mat]
        )
        in_bounds = (self.bounds[0] <= poll_set) & (poll_set <= self.bounds[1])
        return poll_set[np.all(in_bounds, axis=-1)]
