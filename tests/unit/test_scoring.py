import math
from typing import Tuple
from unittest import mock

import pytest
from pytest_mock import MockerFixture

from metahopt.scoring import (
    ScoringResults,
    ScoringStopReason,
    _clean_score_params,
    score_solutions,
    score_vectorized,
)


def test_clean_score_params(mocker):
    solutions = mocker.sentinel.solutions
    rng_seed = mocker.sentinel.rng_seed

    # All default
    res = _clean_score_params(
        solutions, None, None, None, random_order=False, rng_seed=rng_seed
    )
    assert res == (solutions, None, None)

    # With max_time
    res = _clean_score_params(
        solutions, 12, None, None, random_order=False, rng_seed=rng_seed
    )
    assert res == (solutions, 12, None)

    # With max_eval
    res = _clean_score_params(
        solutions, None, 2, None, random_order=False, rng_seed=rng_seed
    )
    assert res == (solutions, None, 2)

    # With max_eval_ratio
    res = _clean_score_params(
        [0, 1], None, None, 0.5, random_order=False, rng_seed=rng_seed
    )
    assert res == ([0, 1], None, 1)

    # With max_eval and max_eval_ratio
    sols = range(10)
    res = _clean_score_params(sols, None, 2, 0.5, random_order=False, rng_seed=rng_seed)
    assert res == (sols, None, 2)

    # With max_eval, max_eval_ratio and an empty solution set
    assert _clean_score_params(
        [], None, 2, 0.5, random_order=False, rng_seed=rng_seed
    ) == ([], None, 0)

    # Random order
    m_default_rng = mocker.patch("numpy.random.default_rng")
    res = _clean_score_params(
        solutions, None, None, None, random_order=True, rng_seed=rng_seed
    )
    assert res == (m_default_rng.return_value.permutation.return_value, None, None)
    m_default_rng.assert_called_once_with(rng_seed)
    m_default_rng.return_value.permutation.assert_called_once_with(solutions)

    with pytest.raises(ValueError, match="max_time=0"):
        _clean_score_params([], 0, 0, None, random_order=False, rng_seed=rng_seed)

    with pytest.raises(ValueError, match="max_eval=0"):
        _clean_score_params([], None, 0, None, random_order=False, rng_seed=rng_seed)

    with pytest.raises(ValueError, match="max_eval_ratio=0"):
        _clean_score_params([], None, None, 0, random_order=False, rng_seed=rng_seed)

    with pytest.raises(ValueError, match="max_eval_ratio=2"):
        _clean_score_params([], None, None, 2, random_order=False, rng_seed=rng_seed)

    # Error if specifying max_eval_ratio with an iterable that has no len()
    with pytest.raises(TypeError, match="object of type 'generator' has no len()"):
        _clean_score_params(
            (x for x in [0, 1]), None, None, 0.5, random_order=False, rng_seed=rng_seed
        )


def test_score_solutions_clean_params_call(mocker: MockerFixture):
    """Check that _clean_score_params() is called."""
    solutions = mocker.sentinel.solutions
    max_time = mocker.sentinel.max_time
    max_eval = mocker.sentinel.max_eval
    max_eval_ratio = mocker.sentinel.max_eval_ratio
    random_order = mocker.sentinel.random_order
    rng_seed = mocker.sentinel.rng_seed

    m_clean_score_params = mocker.patch(
        "metahopt.scoring._clean_score_params",
        return_value=(["sol"], None, None),
    )
    score_solutions(
        lambda _x: 0,
        solutions,
        max_time,
        max_eval,
        max_eval_ratio,
        None,
        random_order=random_order,
        rng_seed=rng_seed,
    )
    m_clean_score_params.assert_called_once_with(
        solutions,
        max_time,
        max_eval,
        max_eval_ratio,
        random_order=random_order,
        rng_seed=rng_seed,
    )


@pytest.fixture
def score_solutions_setup(mocker: MockerFixture) -> Tuple[mock.Mock, mock.Mock]:
    m_identity = mocker.Mock(side_effect=lambda x: x)  # Identity
    m_process_time = mocker.patch(
        "metahopt.scoring.process_time",
        side_effect=lambda: m_identity.call_count,
    )
    return m_identity, m_process_time


def test_score_solutions_standard(score_solutions_setup):
    """Standard complete scoring, no termination criteria"""
    m_identity, m_process_time = score_solutions_setup
    assert score_solutions(m_identity, [2, 0, 1]) == ScoringResults(
        score=0,
        solution=0,
        solution_index=1,
        time=3,
        n_eval=3,
        n_calls=3,
        stop_reason=ScoringStopReason.ScanComplete,
    )
    assert m_identity.call_args_list == [mock.call(sol) for sol in [2, 0, 1]]
    assert m_process_time.call_count == 2


def test_score_solutions_stop_max_time(score_solutions_setup):
    """Limited by max_time"""
    m_identity, m_process_time = score_solutions_setup
    assert score_solutions(m_identity, [2, 0, 1], max_time=1) == ScoringResults(
        score=2,
        solution=2,
        solution_index=0,
        time=1,
        n_eval=1,
        n_calls=1,
        stop_reason=ScoringStopReason.MaxTime,
    )
    assert m_identity.call_args_list == [mock.call(2)]
    assert m_process_time.call_count == 4


def test_score_solutions_stop_max_eval(score_solutions_setup):
    """Limited by max_eval"""
    m_identity, m_process_time = score_solutions_setup
    assert score_solutions(m_identity, [2, 0, 1], max_eval=1) == ScoringResults(
        score=2,
        solution=2,
        solution_index=0,
        time=1,
        n_eval=1,
        n_calls=1,
        stop_reason=ScoringStopReason.MaxEval,
    )
    assert m_identity.call_args_list == [mock.call(2)]
    assert m_process_time.call_count == 2


def test_score_solutions_stop_score(score_solutions_setup):
    """Limited by stop_score"""
    m_identity, m_process_time = score_solutions_setup
    assert score_solutions(m_identity, [2, 0, 1], stop_score=1) == ScoringResults(
        score=0,
        solution=0,
        solution_index=1,
        time=2,
        n_eval=2,
        n_calls=2,
        stop_reason=ScoringStopReason.ScoreImprovement,
    )
    assert m_identity.call_args_list == [mock.call(2), mock.call(0)]
    assert m_process_time.call_count == 2


def test_score_solutions_empty_solution_set(score_solutions_setup):
    """Test with an empty solution set"""
    m_identity, m_process_time = score_solutions_setup
    assert score_solutions(m_identity, []) == ScoringResults(
        score=math.inf,
        solution=None,
        solution_index=None,
        time=0,
        n_eval=0,
        n_calls=0,
        stop_reason=ScoringStopReason.ScanComplete,
    )
    assert m_identity.call_args_list == []
    assert m_process_time.call_count == 2


def test_score_solutions_reach_end(score_solutions_setup):
    """Check that stop_reason is CompleteScan whenever all solutions have been scored,
    even if other termination criteria are met for the last solution.
    """
    m_identity, m_process_time = score_solutions_setup
    assert score_solutions(
        m_identity, [2, 1, 0], max_time=3, max_eval=3, stop_score=1
    ) == ScoringResults(
        score=0,
        solution=0,
        solution_index=2,
        time=3,
        n_eval=3,
        n_calls=3,
        stop_reason=ScoringStopReason.ScanComplete,
    )
    assert m_identity.call_args_list == [mock.call(sol) for sol in [2, 1, 0]]
    assert m_process_time.call_count == 5


def test_score_vectorized(score_solutions_setup):
    """Test with standard input"""
    m_identity, m_process_time = score_solutions_setup

    assert score_vectorized(m_identity, [2, 0, 1]) == ScoringResults(
        score=0,
        solution=0,
        solution_index=1,
        time=1,
        n_eval=3,
        n_calls=1,
        stop_reason=ScoringStopReason.ScanComplete,
    )
    m_identity.assert_called_once_with([2, 0, 1])
    assert m_process_time.call_count == 2


def test_score_vectorized_random_order(mocker: MockerFixture, score_solutions_setup):
    """Test with order randomization"""
    m_identity, m_process_time = score_solutions_setup
    m_default_rng = mocker.patch("numpy.random.default_rng")
    m_permutation = m_default_rng.return_value.permutation
    m_permutation.return_value = [2, 0, 1]
    solutions = mocker.sentinel.solutions
    rng_seed = mocker.sentinel.rng_seed

    assert score_vectorized(
        m_identity, solutions, random_order=True, rng_seed=rng_seed
    ) == ScoringResults(
        score=0,
        solution=0,
        solution_index=1,
        time=1,
        n_eval=3,
        n_calls=1,
        stop_reason=ScoringStopReason.ScanComplete,
    )
    m_identity.assert_called_once_with([2, 0, 1])
    assert m_process_time.call_count == 2
    m_default_rng.assert_called_once_with(rng_seed)
    m_permutation.assert_called_once_with(solutions)
