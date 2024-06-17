"""Primitives for local search.
"""

from __future__ import annotations

import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from time import process_time
from typing import Generic, cast

from metahopt.scoring import ScoringResults, score_solutions, score_vectorized
from metahopt.typing import (
    ObjectiveFunc,
    RngSeed,
    SizedIterable,
    Solution,
    VectorizedObjectiveFunc,
)
from metahopt.utils import format_time


class PollOrder(Enum):
    """
    Matlab:
    'Consecutive' (default) — The algorithm polls the mesh points in consecutive order,
    that is, the order of the pattern vectors as described in Poll Method.

    'Random' — The polling order is random.

    'Success' — The first search direction at each iteration is the direction in which
    the algorithm found the best point at the previous iteration. After the first point,
    the algorithm polls the mesh points in the same order as 'Consecutive'.
    """

    Consecutive = "consecutive"
    Success = "success"
    Random = "random"


class TerminationReason(Enum):
    MaxTime = "max_time"
    MaxIter = "max_iter"
    MaxCalls = "max_calls"
    MinScore = "min_score"
    MaxStalledTime = "max_stalled_time"
    MaxStalledIter = "max_stalled_iter"
    MaxStalledCalls = "max_stalled_calls"
    EmptyPollSet = "empty_poll_set"


def reached_max_threshold(value: int | float, threshold: int | float | None) -> bool:
    return threshold is not None and value >= threshold


@dataclass
class LocalSearchState(Generic[Solution]):
    best_score: float
    best_solution: Solution
    time: float
    # stalled_time: float  # TODO: Implement
    n_iter: int
    n_stalled_iter: int
    n_calls: int
    n_stalled_calls: int
    success_direction: int | None

    def update(
        self, scoring_res: ScoringResults, start_time: float
    ) -> LocalSearchState[Solution]:
        # Score improved
        if scoring_res.score < self.best_score:
            best_score = scoring_res.score
            best_solution = scoring_res.solution
            success_direction = scoring_res.solution_index
            n_stalled_iter = 0
            n_stalled_calls = 0
        # No improvement
        else:
            best_score = self.best_score
            best_solution = self.best_solution
            success_direction = None
            n_stalled_iter = self.n_stalled_iter + 1
            n_stalled_calls = self.n_stalled_calls + scoring_res.n_calls

        return LocalSearchState[Solution](
            best_score=best_score,
            best_solution=best_solution,
            time=process_time() - start_time,
            n_iter=self.n_iter + 1,
            n_stalled_iter=n_stalled_iter,
            n_calls=self.n_calls + scoring_res.n_calls,
            n_stalled_calls=n_stalled_calls,
            success_direction=success_direction,
        )


@dataclass
class LocalSearchResults(Generic[Solution]):
    termination_reason: TerminationReason
    state: LocalSearchState[Solution]
    stats: list[ScoringResults[Solution]]


@dataclass(kw_only=True)  # type: ignore  # https://github.com/python/mypy/issues/5374
class LocalSearch(Generic[Solution], metaclass=ABCMeta):
    # TODO: Add:
    #  * cache?
    #  * parallelization
    #  * termination function tolerance: average change of score over max_stall_iter is
    #    less than func_tolerance
    #  * display, output callback

    objective: ObjectiveFunc | VectorizedObjectiveFunc
    vectorized: bool = False
    max_time: float | None = None
    max_iter: int | None = None
    max_calls: int | None = None
    min_score: float | None = None
    poll_order: PollOrder = PollOrder.Consecutive
    complete_poll: bool = True
    rng_seed: RngSeed = None
    max_stalled_iter: int | None = None
    max_stalled_calls: int | None = None
    # stalled_score_tolerance: float = 1e-6

    def __post_init__(self):
        self.poll_order = PollOrder(self.poll_order)
        self._logger = logging.getLogger("metahopt.solver")

        # Type hints
        self._objective_iter = cast(ObjectiveFunc, self.objective)
        self._objective_vec = cast(VectorizedObjectiveFunc, self.objective)

    def init_state(self, starting_point: Solution) -> LocalSearchState[Solution]:
        if self.vectorized:
            init_score = self._objective_vec([starting_point])[0]
        else:
            init_score = self._objective_iter(starting_point)
        return LocalSearchState[Solution](
            best_score=init_score,
            best_solution=starting_point,
            time=0,
            n_iter=0,
            n_stalled_iter=0,
            n_calls=1,
            n_stalled_calls=0,
            success_direction=None,
        )

    def poll_set_iter(self, state: LocalSearchState[Solution]) -> SizedIterable[Solution]:
        """Generate neighborhood."""
        return self.poll_set_vectorized(state)

    @abstractmethod
    def poll_set_vectorized(
        self, state: LocalSearchState[Solution]
    ) -> Sequence[Solution]:
        """Generate neighborhood."""

    def score_iter(
        self, state: LocalSearchState[Solution], polling_set: SizedIterable[Solution]
    ) -> ScoringResults:
        return score_solutions(
            self._objective_iter,
            polling_set,
            max_time=None if self.max_time is None else self.max_time - state.time,
            max_eval=None if self.max_calls is None else self.max_calls - state.n_calls,
            stop_score=None if self.complete_poll else state.best_score,
            random_order=self.poll_order is PollOrder.Random,
            rng_seed=self.rng_seed,
        )

    def score_vectorized(
        self,
        state: LocalSearchState[Solution],  # noqa: ARG002
        polling_set: Sequence[Solution],
    ) -> ScoringResults:
        random_order = self.poll_order is PollOrder.Random
        return score_vectorized(
            self._objective_vec,
            polling_set,
            random_order=random_order,
            rng_seed=self.rng_seed,
        )

    def check_termination(
        self, state: LocalSearchState[Solution]
    ) -> TerminationReason | None:
        if self.min_score is not None and state.best_score < self.min_score:
            self._logger.debug("Stopping: reached score limit (%s)", self.min_score)
            return TerminationReason.MinScore

        if reached_max_threshold(state.time, self.max_time):
            self._logger.debug("Stopping: reached time limit (%s)", self.max_time)
            return TerminationReason.MaxTime
        if reached_max_threshold(state.n_iter, self.max_iter):
            self._logger.debug("Stopping: reached max iterations (%s)", self.max_iter)
            return TerminationReason.MaxIter
        if reached_max_threshold(state.n_calls, self.max_calls):
            self._logger.debug(
                "Stopping: reached max objective function calls (%s)", self.max_calls
            )
            return TerminationReason.MaxCalls
        # TODO: Implement stalled_time
        if reached_max_threshold(state.n_stalled_iter, self.max_stalled_iter):
            self._logger.debug(
                "Stopping: reached max stalled iterations (%s)", self.max_stalled_iter
            )
            return TerminationReason.MaxStalledIter
        if reached_max_threshold(state.n_stalled_calls, self.max_stalled_calls):
            self._logger.debug(
                "Stopping: reached max stalled objective function calls (%s)",
                self.max_stalled_calls,
            )
            return TerminationReason.MaxStalledCalls
        return None

    def solve(self, starting_point: Solution) -> LocalSearchResults[Solution]:
        self._logger.info(
            "Minimizing %r with %s", self.objective, self.__class__.__name__
        )
        stats = []
        start_time = process_time()

        state = self.init_state(starting_point)
        termination_reason = self.check_termination(state)
        while termination_reason is None:
            if self.vectorized:
                poll_set = self.poll_set_vectorized(state)
                if len(poll_set) == 0:
                    termination_reason = TerminationReason.EmptyPollSet
                    break
                scoring_res = self.score_vectorized(state, poll_set)
            else:
                poll_set = self.poll_set_iter(state)
                if len(poll_set) == 0:
                    termination_reason = TerminationReason.EmptyPollSet
                    break
                scoring_res = self.score_iter(state, poll_set)
            stats.append(scoring_res)
            state = state.update(scoring_res, start_time)
            termination_reason = self.check_termination(state)

        n_step = 0 if state.n_iter is None else state.n_iter + 1
        self._logger.info(
            "Finished solving in %s steps and %s (best_score=%s, best_solution=%s)",
            n_step,
            format_time(state.time),
            state.best_score,
            state.best_solution,
        )
        return LocalSearchResults(termination_reason, state, stats)
