from dataclasses import dataclass
from enum import Enum
import logging
import math
from time import process_time
from typing import Callable, Collection, Iterable, Optional, Union

import numpy as np

from metahopt.typing import RngSeedType, SolutionType


@dataclass
class ScoringStopReason(Enum):
    ScanComplete = 0
    MaxTime = 1
    MaxEval = 2
    ScoreImprovement = 3


@dataclass
class ScoringResults:
    score: float
    solution: Optional[SolutionType]
    solution_index: Optional[int]
    time: float
    n_eval: int
    n_calls: int
    stop_reason: ScoringStopReason


def _clean_score_params(
    solutions: Union[Iterable[SolutionType], Collection[SolutionType]],
    max_time: Optional[float],
    max_eval: Optional[int],
    max_eval_ratio: Optional[float],
    random_order: bool,
    rng_seed: RngSeedType,
) -> (Iterable[SolutionType], Optional[float], Optional[int]):
    """
    If using max_eval_ratio, solutions needs to be sized.

    Args:
        max_eval:
        max_eval_ratio:
        solutions:

    Returns:
        float or None, int or None: max_time and max_eval
    """
    if max_time is not None and max_time <= 0.0:
        raise ValueError(f"max_time={max_time}, must be greater than 0")

    if max_eval is not None and max_eval < 1:
        raise ValueError(f"max_eval={max_eval}, must be greater than or equal to 1")

    # Randomize solutions iterable before max_eval_ratio
    # If solutions is a generator it is materialized at this point
    if random_order:
        solutions = np.random.default_rng(rng_seed).permutation(solutions)

    if max_eval_ratio is not None:
        if not 0.0 < max_eval_ratio <= 1.0:
            raise ValueError(f"max_eval_ratio={max_eval_ratio}, must be in ]0; 1]")
        n_sol = len(solutions)  # Requires the solutions iterable to have a len()
        max_eval = min(
            math.inf if max_eval is None else max_eval,
            int(n_sol * max_eval_ratio),
        )

    return solutions, max_time, max_eval


def score_solutions(
    score_func: Callable[[SolutionType], float],
    solutions: Iterable[SolutionType],
    max_time: Optional[float] = None,
    max_eval: Optional[int] = None,
    max_eval_ratio: Optional[float] = None,
    stop_score: Optional[float] = None,
    random_order: bool = False,
    rng_seed: RngSeedType = None,
) -> ScoringResults:
    """

    Args:
        score_func:
        solutions (Iterable): The collection of solutions to scan and evaluate.
            If `randomize` is True, the collection is turned into a list,
            materializing it if it is a generator. If `max_eval_ratio` is True, the
            collection must be sized (can be used with `len()`).
        max_time:
        max_eval:
        max_eval_ratio:
        stop_score:
        random_order:
        rng_seed:

    Returns:
        tuple of (float, SolutionType): The best score and the corresponding
        solution found while scanning the collection.
    """
    logger = logging.getLogger("metahopt.scoring")
    logger.debug("Scoring solution set")
    start_time = process_time()  # Before randomization to include it in timing

    solutions, max_time, max_eval = _clean_score_params(
        solutions, max_time, max_eval, max_eval_ratio, random_order, rng_seed
    )

    # Initialization
    score = math.inf
    best_score = math.inf
    best_sol = None
    best_idx = None
    stop_reason = ScoringStopReason.ScanComplete
    n_eval = 0

    # Scoring loop
    for sol in solutions:
        # Termination tests (we want them first to not perform them after last eval)
        if stop_score is not None and score < stop_score:  # Highest priority
            stop_reason = ScoringStopReason.ScoreImprovement
            logger.debug("Stopping: found score improvement")
            break
        if max_time is not None and process_time() - start_time >= max_time:
            stop_reason = ScoringStopReason.MaxTime
            logger.debug("Stopping: reached time limit (%s)", max_time)
            break
        if max_eval is not None and n_eval >= max_eval:
            stop_reason = ScoringStopReason.MaxEval
            logger.debug("Stopping: reached max evaluations (%s)", max_eval)
            break
        # Solution evaluation
        logger.debug("[Iter %s] Evaluating solution %s", n_eval, sol)
        score = score_func(sol)
        n_eval += 1
        if score < best_score:
            best_score = score
            best_sol = sol
            best_idx = n_eval - 1

    # Finalization
    scoring_time = process_time() - start_time
    logger.info("Scored %s solutions in %.3f s", n_eval, scoring_time)
    return ScoringResults(
        best_score, best_sol, best_idx, scoring_time, n_eval, n_eval, stop_reason
    )


def score_vectorized(
    score_func: Callable[[Iterable[SolutionType]], Iterable[SolutionType]],
    solutions: Iterable[SolutionType],
    random_order: bool = False,
    rng_seed: RngSeedType = None,
) -> ScoringResults:
    """

    Args:
        score_func:
        solutions (Iterable): The collection of solutions to scan and evaluate.
            If `randomize` is True, the collection is turned into a list,
            materializing it if it is a generator. If `max_eval_ratio` is True, the
            collection must be sized (can be used with `len()`).
        random_order:
        rng_seed:

    Returns:
        tuple of (float, SolutionType): The best score and the corresponding
        solution found while scanning the collection.
    """
    logger = logging.getLogger("metahopt.scoring")
    logger.debug("Scoring vectorized solution set")
    start_time = process_time()

    if random_order:
        solutions = np.random.default_rng(rng_seed).permutation(solutions)

    scores = np.asanyarray(score_func(solutions))
    best_idx: int = np.argmin(scores)
    best_score = scores[best_idx]
    best_solution = solutions[best_idx]

    scoring_time = process_time() - start_time
    logger.info("Scored %s solutions in %.3f s", len(scores), scoring_time)
    return ScoringResults(
        best_score,
        best_solution,
        best_idx,
        scoring_time,
        len(scores),
        1,
        ScoringStopReason.ScanComplete,
    )
