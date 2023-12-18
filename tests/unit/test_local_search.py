import logging
from dataclasses import replace
from unittest import mock

from pytest_mock import MockerFixture

import metahopt.local_search as mod
from metahopt.local_search import (
    LocalSearch,
    LocalSearchState,
    PollOrder,
    TerminationReason,
)
from metahopt.scoring import ScoringResults, ScoringStopReason


class MyLocalSearch(LocalSearch):
    """Concrete LocalSearch, implementing abstract methods."""

    def poll_set_vectorized(self, state: LocalSearchState):
        return ["sol1", "sol2"]


objective = mock.sentinel.objective
state = mock.sentinel.state
poll_set = mock.sentinel.poll_set
rng_seed = mock.sentinel.rng_seed


def test_local_search_state_update(mocker: MockerFixture):
    mocker.patch("metahopt.local_search.process_time", return_value=1)
    state = LocalSearchState(
        1.0,
        "sol1",
        time=3,
        n_iter=4,
        n_stalled_iter=0,
        n_calls=4,
        n_stalled_calls=0,
        success_direction=None,
    )

    scoring_res = ScoringResults(
        2.0,
        "sol2",
        solution_index=12,
        time=1,
        n_eval=2,
        n_calls=1,
        stop_reason=ScoringStopReason.ScanComplete,
    )
    assert state.update(scoring_res, 0) == LocalSearchState(
        1.0,
        "sol1",
        time=1,
        n_iter=5,
        n_stalled_iter=1,
        n_calls=5,
        n_stalled_calls=1,
        success_direction=None,
    )

    scoring_res = replace(scoring_res, score=0.0)
    assert state.update(scoring_res, 0) == LocalSearchState(
        0.0,
        "sol2",
        time=1,
        n_iter=5,
        n_stalled_iter=0,
        n_calls=5,
        n_stalled_calls=0,
        success_direction=12,
    )


def test_local_search_init():
    solver = MyLocalSearch(objective=objective, poll_order=PollOrder.Success)
    assert solver.poll_order is PollOrder.Success
    assert isinstance(solver._logger, logging.Logger)
    assert solver._logger.name == "metahopt.solver"

    solver = MyLocalSearch(objective=objective, poll_order="random")
    assert solver.poll_order is PollOrder.Random


def test_local_search_init_state():
    solver = MyLocalSearch(objective=lambda _: 42)
    assert solver.init_state("s") == LocalSearchState(42, "s", 0, 0, 0, 1, 0, None)

    solver = MyLocalSearch(objective=lambda _: [42], vectorized=True)
    assert solver.init_state("s") == LocalSearchState(42, "s", 0, 0, 0, 1, 0, None)


def test_local_search_score_vectorized(mocker: MockerFixture):
    m_score_vec = mocker.patch.object(mod, "score_vectorized")

    solver = MyLocalSearch(objective=objective, vectorized=True, rng_seed=rng_seed)
    assert solver.score_vectorized(state, poll_set) is m_score_vec.return_value
    m_score_vec.assert_called_once_with(
        objective, poll_set, random_order=False, rng_seed=rng_seed
    )

    mocker.resetall()
    solver = MyLocalSearch(
        objective=objective,
        vectorized=True,
        poll_order=PollOrder.Random,
        rng_seed=rng_seed,
    )
    assert solver.score_vectorized(state, poll_set) is m_score_vec.return_value
    m_score_vec.assert_called_once_with(
        objective, poll_set, random_order=True, rng_seed=rng_seed
    )


def test_local_search_score_iter(mocker: MockerFixture):
    m_score_solutions = mocker.patch.object(mod, "score_solutions")
    state = mocker.Mock(
        spec_set=["time", "n_calls", "best_score"], time=0, n_calls=0, best_score=0
    )

    solver = MyLocalSearch(
        objective=objective,
        vectorized=False,
        max_time=None,
        max_calls=None,
        complete_poll=True,
        rng_seed=rng_seed,
    )
    assert solver.score_iter(state, poll_set) is m_score_solutions.return_value
    m_score_solutions.assert_called_once_with(
        objective,
        poll_set,
        max_time=None,
        max_eval=None,
        stop_score=None,
        random_order=False,
        rng_seed=rng_seed,
    )


def test_local_search_check_termination():
    base_solver = MyLocalSearch(objective=objective)
    state = LocalSearchState(
        1.0,
        "sol",
        time=3,
        n_iter=4,
        n_stalled_iter=2,
        n_calls=5,
        n_stalled_calls=2,
        success_direction=None,
    )
    assert base_solver.check_termination(state) is None

    solver = replace(base_solver, min_score=2)
    assert solver.check_termination(state) is TerminationReason.MinScore

    solver = replace(base_solver, max_time=2)
    assert solver.check_termination(state) is TerminationReason.MaxTime

    solver = replace(base_solver, max_iter=3)
    assert solver.check_termination(state) is TerminationReason.MaxIter

    solver = replace(base_solver, max_calls=3)
    assert solver.check_termination(state) is TerminationReason.MaxCalls

    solver = replace(base_solver, max_stalled_iter=2)
    assert solver.check_termination(state) is TerminationReason.MaxStalledIter

    solver = replace(base_solver, max_stalled_calls=2)
    assert solver.check_termination(state) is TerminationReason.MaxStalledCalls


def test_local_search_solve_iter(mocker: MockerFixture):
    def objective_func(x):
        return {"sol0": 3, "sol1": 1, "sol2": 2}[x]

    solver = MyLocalSearch(objective=objective_func, max_iter=1)
    res = solver.solve("sol0")
    assert res.state == LocalSearchState(1, "sol1", mocker.ANY, 1, 0, 3, 0, 0)
    assert res.termination_reason is TerminationReason.MaxIter
    assert len(res.stats) == 1


def test_local_search_solve_vectorized(mocker: MockerFixture):
    def objective_func(x):
        d = {"sol0": 3, "sol1": 1, "sol2": 2}
        return [d[v] for v in x]

    solver = MyLocalSearch(objective=objective_func, vectorized=True, max_iter=1)
    res = solver.solve("sol0")
    assert res.state == LocalSearchState(1, "sol1", mocker.ANY, 1, 0, 2, 0, 0)
    assert res.termination_reason is TerminationReason.MaxIter
    assert len(res.stats) == 1
