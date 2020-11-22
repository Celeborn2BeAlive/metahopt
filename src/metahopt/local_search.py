from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
import logging
from time import process_time
from typing import cast, Sequence, Tuple, Generic, List, Optional, Union

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


@dataclass
class LocalSearchState(Generic[SolutionType]):
    params: "LocalSearch"
    best_score: float
    best_solution: SolutionType
    time: float
    n_iter: int
    # n_stall_iter: int  # TODO
    n_calls: int
    success_direction: Optional[int]

    @classmethod
    def from_base_state(cls, state: "LocalSearchState", *args, **kwargs):
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

    score_func: Union[ScoreFunc, VectorizedScoreFunc]
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

    def init_state(self, starting_point: SolutionType) -> LocalSearchState:
        if self.vectorized:
            score_func_vec = cast(VectorizedScoreFunc, self.score_func)
            init_score = score_func_vec([starting_point])[0]
        else:
            score_func = cast(ScoreFunc, self.score_func)
            init_score = score_func(starting_point)
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
    def get_polling_set(
        self, state: LocalSearchState
    ) -> Union[SizedIterable[SolutionType], Sequence[SolutionType]]:
        """Generate neighborhood."""

    def score(
        self,
        state: LocalSearchState,
        polling_set: Union[SizedIterable[SolutionType], Sequence[SolutionType]],
    ) -> ScoringResults:
        random_order = self.poll_order is PollOrder.random
        if self.vectorized:
            # Typing hints, deactivated for performance
            # score_func = cast(VectorizedScoreFunc, self.score_func)
            # polling_set = cast(Sequence[SolutionType], polling_set)
            return score_vectorized(
                self.score_func,  # type: ignore
                polling_set,  # type: ignore
                random_order,
                self.rng_seed,
            )
        else:
            # Typing hints, deactivated for performance
            # score_func = cast(ScoreFunc, self.score_func)
            # polling_set = cast(SizedIterable[SolutionType], polling_set)
            max_time = None if self.max_time is None else self.max_time - state.time
            max_eval = (
                None if self.max_calls is None else self.max_calls - state.n_calls
            )
            return score_solutions(
                self.score_func,  # type: ignore
                polling_set,  # type: ignore
                max_time,
                max_eval,
                stop_score=None if self.complete_poll else state.best_score,
                random_order=random_order,
                rng_seed=self.rng_seed,
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

    def terminated(self, state: LocalSearchState) -> bool:
        if self.min_score is not None and state.best_score < self.min_score:
            self._logger.debug("Stopping: reached score limit (%s)", self.min_score)
            return True
        if self.max_time is not None and state.time > self.max_time:
            self._logger.debug("Stopping: reached time limit (%s)", self.max_time)
            return True
        if self.max_iter is not None and state.n_iter >= self.max_iter:
            self._logger.debug("Stopping: reached max steps (%s)", self.max_iter)
            return True
        if self.max_calls is not None and state.n_calls >= self.max_calls:
            self._logger.debug(
                "Stopping: reached max score function calls (%s)", self.max_calls
            )
            return True
        return False

    def solve(
        self, starting_point: SolutionType
    ) -> Tuple[LocalSearchState, List[ScoringResults]]:
        self._logger.info(
            "Minimizing %r with %s", self.score_func, self.__class__.__name__
        )
        stats = []
        start_time = process_time()

        state = self.init_state(starting_point)
        while not self.terminated(state):
            # Type casting deactivated for polling_set for performance
            polling_set = self.get_polling_set(state)  # type: ignore
            scoring_res = self.score(state, polling_set)
            stats.append(scoring_res)
            state = self.update(state, scoring_res, start_time)

        n_step = 0 if state.n_iter is None else state.n_iter + 1
        self._logger.info(
            "Finished solving in %s steps (best_score=%s, best_solution=%s)",
            *(n_step, state.best_score, state.best_solution),
        )
        return state, stats
