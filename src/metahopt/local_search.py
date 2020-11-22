from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
import logging
from time import process_time
from typing import Callable, cast, Sequence, Tuple, Generic, List, Optional, Union

from metahopt.scoring import ScoringResults, score_solutions, score_vectorized
from metahopt.typing import (
    RngSeed,
    ScoreFunc,
    SizedIterable,
    SolutionType,
    VectorizedScoreFunc,
)


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

    consecutive = "consecutive"
    success = "success"
    random = "random"


class TerminationReason(Enum):
    max_time = 0
    max_iter = 1
    max_calls = 2
    min_score = 3


@dataclass
class LocalSearchState(Generic[SolutionType]):
    params: LocalSearch
    best_score: float
    best_solution: SolutionType
    time: float
    n_iter: int
    # n_stall_iter: int  # TODO
    n_calls: int
    success_direction: Optional[int]

    @classmethod
    def from_base_state(cls, state: LocalSearchState, *args, **kwargs):
        # TODO: Metaclass to assign child's class signature?
        unpacked = tuple(getattr(state, field.name) for field in fields(state))
        return cls(*unpacked, *args, **kwargs)


@dataclass  # type: ignore  # https://github.com/python/mypy/issues/5374
class LocalSearch(metaclass=ABCMeta):
    # TODO: Add:
    #  * cache?
    #  * parallelization
    #  * termination function tolerance: average change of score over max_stall_iter is
    #    less than func_tolerance
    #  * display, output callback

    objective_func: Union[ScoreFunc, VectorizedScoreFunc]
    vectorized: bool = False
    max_time: Optional[float] = None
    max_iter: Optional[int] = None
    max_calls: Optional[int] = None
    min_score: Optional[float] = None
    poll_order: PollOrder = PollOrder.consecutive
    complete_poll: bool = True
    rng_seed: RngSeed = None
    # max_stall_iter: Optional[int] = None  # TODO
    # stall_score_tolerance: float = 1e-3  # TODO

    def __post_init__(self):
        self.poll_order = PollOrder(self.poll_order)
        self._logger = logging.getLogger("metahopt.solver")

        # Type hints
        self._score_func_iter = cast(ScoreFunc, self.objective_func)
        self._score_func_vec = cast(VectorizedScoreFunc, self.objective_func)
        self._neighborhood_func_vec = cast(
            Callable[[LocalSearchState], Sequence[SolutionType]],
            self.neighborhood,
        )
        self._neighborhood_func_iter = cast(
            Callable[[LocalSearchState], SizedIterable[SolutionType]],
            self.neighborhood,
        )

    def init_state(self, starting_point: SolutionType) -> LocalSearchState:
        if self.vectorized:
            init_score = self._score_func_vec([starting_point])[0]
        else:
            init_score = self._score_func_iter(starting_point)
        return LocalSearchState(
            params=self,
            best_score=init_score,
            best_solution=starting_point,
            time=0,
            n_iter=0,
            n_calls=1,
            success_direction=None,
        )

    @abstractmethod
    def neighborhood(
        self, state: LocalSearchState
    ) -> Union[SizedIterable[SolutionType], Sequence[SolutionType]]:
        """Generate neighborhood."""

    def score_iter(
        self, state: LocalSearchState, polling_set: SizedIterable[SolutionType]
    ) -> ScoringResults:
        random_order = self.poll_order is PollOrder.random
        max_time = None if self.max_time is None else self.max_time - state.time
        max_eval = None if self.max_calls is None else self.max_calls - state.n_calls
        return score_solutions(
            self._score_func_iter,
            polling_set,
            max_time,
            max_eval,
            stop_score=None if self.complete_poll else state.best_score,
            random_order=random_order,
            rng_seed=self.rng_seed,
        )

    def score_vectorized(
        self, state: LocalSearchState, polling_set: Sequence[SolutionType]
    ) -> ScoringResults:
        random_order = self.poll_order is PollOrder.random
        return score_vectorized(
            self._score_func_vec, polling_set, random_order, self.rng_seed
        )

    def update(
        self, state: LocalSearchState, scoring_res: ScoringResults, start_time: float
    ) -> LocalSearchState:
        if scoring_res.score >= state.best_score:
            best_score = state.best_score
            best_solution = state.best_solution
            success_direction = None
        else:
            best_score = scoring_res.score
            best_solution = scoring_res.solution
            success_direction = scoring_res.solution_index
        return LocalSearchState(
            params=self,
            best_score=best_score,
            best_solution=best_solution,
            time=process_time() - start_time,
            n_iter=state.n_iter + 1,
            n_calls=state.n_calls + scoring_res.n_calls,
            success_direction=success_direction,
        )

    def check_termination(self, state: LocalSearchState) -> Optional[TerminationReason]:
        if self.min_score is not None and state.best_score < self.min_score:
            self._logger.debug("Stopping: reached score limit (%s)", self.min_score)
            return TerminationReason.min_score
        if self.max_time is not None and state.time > self.max_time:
            self._logger.debug("Stopping: reached time limit (%s)", self.max_time)
            return TerminationReason.max_time
        if self.max_iter is not None and state.n_iter >= self.max_iter:
            self._logger.debug("Stopping: reached max steps (%s)", self.max_iter)
            return TerminationReason.max_iter
        if self.max_calls is not None and state.n_calls >= self.max_calls:
            self._logger.debug(
                "Stopping: reached max score function calls (%s)", self.max_calls
            )
            return TerminationReason.max_calls
        return None

    def solve(
        self, starting_point: SolutionType
    ) -> Tuple[LocalSearchState, TerminationReason, List[ScoringResults]]:
        self._logger.info(
            "Minimizing %r with %s", self.objective_func, self.__class__.__name__
        )
        stats = []
        start_time = process_time()

        state = self.init_state(starting_point)
        termination_reason = self.check_termination(state)
        while termination_reason is None:
            if self.vectorized:
                neighborhood_vec = self._neighborhood_func_vec(state)
                scoring_res = self.score_vectorized(state, neighborhood_vec)
            else:
                neighborhood_iter = self._neighborhood_func_iter(state)
                scoring_res = self.score_iter(state, neighborhood_iter)
            stats.append(scoring_res)
            state = self.update(state, scoring_res, start_time)
            termination_reason = self.check_termination(state)

        n_step = 0 if state.n_iter is None else state.n_iter + 1
        self._logger.info(
            "Finished solving in %s steps (best_score=%s, best_solution=%s)",
            *(n_step, state.best_score, state.best_solution),
        )
        return state, termination_reason, stats
