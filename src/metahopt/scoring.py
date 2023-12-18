import logging
import math
from dataclasses import dataclass
from enum import Enum
from time import process_time
from typing import Generic, Optional, Sequence, Tuple

import numpy as np

from metahopt.typing import (
    ObjectiveFunc,
    RngSeed,
    SizedIterable,
    Solution,
    VectorizedObjectiveFunc,
)
from metahopt.utils import format_time


class ScoringStopReason(Enum):
    """Indicates the reason a scoring function terminated."""

    #: The complete solution set was processed.
    ScanComplete = 0
    #: The time limit was reached while scanning the solution set.
    MaxTime = 1
    #: The maximum number of calls to the scoring function was reached.
    MaxEval = 2
    #: A solution with a better score than the specified limit was found.
    ScoreImprovement = 3


@dataclass(frozen=True)
class ScoringResults(Generic[Solution]):
    """Results returned by a scoring function."""

    score: float
    solution: Optional[Solution]
    solution_index: Optional[int]
    time: float
    n_eval: int
    n_calls: int
    stop_reason: ScoringStopReason


def _clean_score_params(
    solutions: SizedIterable[Solution],
    max_time: Optional[float],
    max_eval: Optional[int],
    max_eval_ratio: Optional[float],
    *,
    random_order: bool,
    rng_seed: RngSeed,
) -> Tuple[SizedIterable[Solution], Optional[float], Optional[int]]:
    """Validate and prepare the parameters of score_solutions().

    Args:
        solutions (iterable of SolutionType): Solution set to be scored. Unless
            `max_eval_ratio` or `random_order` are specified, the only requirement is
            for `solutions` to be iterable. Solutions will be generated only once at
            scoring time.
        max_time (float or None): Maximum time allowed for scoring the solution set. If
            the time limit is reached while evaluating a solution, the scoring will be
            stopped only after this evaluation terminates.
        max_eval (int or None): Maximum number of scoring function calls. Unlimited if
            None.
        max_eval_ratio (float or None): Limits the number of solutions evaluated to a
            ratio of the solution set. Must be in ]0, 1]. Unlimited if None. If
            specified, `solutions` must be sized (needs to have a `len()`).
        random_order (bool): If True, the solution set is shuffled. This triggers the
            generation of the complete solution set if it is a generator.
        rng_seed (RngSeed): Random seed or generator to be used for shuffling
            `solutions` if `random_order` is True.

    Returns:
        iterable of SolutionType, float or None, int or None: Validated and prepared
        values for `solutions`, `max_time` and `max_eval`.
    """
    if max_time is not None and max_time <= 0:
        msg = f"max_time={max_time}, must be greater than 0"
        raise ValueError(msg)

    if max_eval is not None and max_eval < 1:
        msg = f"max_eval={max_eval}, must be greater than or equal to 1"
        raise ValueError(msg)

    # Randomize solutions iterable before max_eval_ratio
    # If solutions is a generator it is materialized at this point
    if random_order:
        solutions = np.random.default_rng(rng_seed).permutation(solutions)

    if max_eval_ratio is not None:
        if not 0 < max_eval_ratio <= 1:
            msg = f"max_eval_ratio={max_eval_ratio}, must be in ]0; 1]"
            raise ValueError(msg)
        n_sol = len(solutions)  # Requires the solutions iterable to have a len()
        n = int(n_sol * max_eval_ratio)
        max_eval = n if max_eval is None else min(n, max_eval)

    return solutions, max_time, max_eval


def score_solutions(
    objective_func: ObjectiveFunc,
    solutions: SizedIterable[Solution],
    max_time: Optional[float] = None,
    max_eval: Optional[int] = None,
    max_eval_ratio: Optional[float] = None,
    stop_score: Optional[float] = None,
    *,
    random_order: bool = False,
    rng_seed: RngSeed = None,
) -> ScoringResults[Solution]:
    """Evaluate all solutions in a collection iteratively.

    Args:
        objective_func (ObjectiveFunc): Objective function for evaluating solutions
            individually.
        solutions (iterable of SolutionType): Solution set to evaluate. Unless
            `max_eval_ratio` or `random_order` are specified, the only requirement is
            for `solutions` to be iterable. Solutions will be generated only once at
            scoring time.
        max_time (float or None, optional, default None): Maximum time allowed for
            scoring the solution set. If the time limit is reached while evaluating a
            solution, the scoring will be stopped only after this evaluation terminates.
        max_eval (int or None, optional, default None): Maximum number of scoring
            function calls. Unlimited if None.
        max_eval_ratio (float or None, optional, default None): Limits the number of
            solutions evaluated to a ratio of the solution set. Must be in ]0, 1].
            Unlimited if None. If specified, `solutions` must be sized (needs to have a
            `len()`).
        stop_score (float, optional, default None): Stops the scoring as soon as a
            better solution is found. No score limit if None.
        random_order (bool, optional, default False): If True, the solution set is
            shuffled. This triggers the generation of the complete solution set if it
            is a generator.
        rng_seed (RngSeed, optional, default None): Random seed or generator to be used
            for shuffling `solutions` if `random_order` is True.

    Returns:
        ScoringResults: The results of scoring the solution set.
    """
    logger = logging.getLogger("metahopt.scoring")
    logger.debug("Scoring solution set")
    start_time = process_time()  # Before randomization to include it in timing

    solutions, max_time, max_eval = _clean_score_params(
        solutions,
        max_time,
        max_eval,
        max_eval_ratio,
        random_order=random_order,
        rng_seed=rng_seed,
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
        score = objective_func(sol)
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
    objective_func: VectorizedObjectiveFunc,
    solutions: Sequence[Solution],
    *,
    random_order: bool = False,
    rng_seed: RngSeed = None,
) -> ScoringResults[Solution]:
    """Evaluate all solutions in a collection with a vectorized objective function.

    Args:
        objective_func (VectorizedObjectiveFunc): Vectorized objective function for
            evaluating a collection of solutions in a single call.
        solutions (iterable of SolutionType): Solution set to evaluate.
        random_order (bool, optional, default False): If True, the solution set is
            shuffled. This triggers the generation of the complete solution set if it
            is a generator.
        rng_seed (RngSeed, optional, default None): Random seed or generator to be used
            for shuffling `solutions` if `random_order` is True.

    Returns:
        ScoringResults: The results of scoring the solution set.
    """
    logger = logging.getLogger("metahopt.scoring")
    logger.debug("Scoring vectorized solution set")
    start_time = process_time()

    if random_order:
        solutions = np.random.default_rng(rng_seed).permutation(solutions)

    scores = np.asanyarray(objective_func(solutions))
    best_idx: int = np.argmin(scores)
    best_score = scores[best_idx]
    best_solution = solutions[best_idx]

    scoring_time = process_time() - start_time
    logger.info("Scored %s solutions in %s", len(scores), format_time(scoring_time))
    return ScoringResults(
        best_score,
        best_solution,
        best_idx,
        scoring_time,
        len(scores),
        1,
        ScoringStopReason.ScanComplete,
    )
