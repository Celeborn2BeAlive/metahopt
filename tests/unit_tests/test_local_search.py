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

    def neighborhood(self, state: LocalSearchState):
        return ["sol1", "sol2"]


objective_func = mock.sentinel.objective_func
state = mock.sentinel.state
neighborhood = mock.sentinel.neighborhood
rng_seed = mock.sentinel.rng_seed


def test_local_search_state():
    solver = MyLocalSearch(objective_func)
    state = LocalSearchState(
        solver, 1.0, "sol", time=2, n_iter=3, n_calls=4, success_direction=None
    )
    assert state.from_base_state(state) == state


def test_local_search_init():
    solver = MyLocalSearch(objective_func, poll_order=PollOrder.success)
    assert solver.poll_order is PollOrder.success
    assert isinstance(solver._logger, logging.Logger)
    assert solver._logger.name == "metahopt.solver"

    solver = MyLocalSearch(objective_func, poll_order="random")
    assert solver.poll_order is PollOrder.random


def test_local_search_init_state():
    solver = MyLocalSearch(lambda x: 42)
    assert solver.init_state("s") == LocalSearchState(solver, 42, "s", 0, 0, 1, None)

    solver = MyLocalSearch(lambda x: [42], vectorized=True)
    assert solver.init_state("s") == LocalSearchState(solver, 42, "s", 0, 0, 1, None)


def test_local_search_score_vectorized(mocker: MockerFixture):
    m_score_vec = mocker.patch.object(mod, "score_vectorized")

    solver = MyLocalSearch(objective_func, vectorized=True, rng_seed=rng_seed)
    assert solver.score_vectorized(state, neighborhood) is m_score_vec.return_value
    m_score_vec.assert_called_once_with(objective_func, neighborhood, False, rng_seed)

    mocker.resetall()
    solver = MyLocalSearch(
        objective_func, vectorized=True, poll_order=PollOrder.random, rng_seed=rng_seed
    )
    assert solver.score_vectorized(state, neighborhood) is m_score_vec.return_value
    m_score_vec.assert_called_once_with(objective_func, neighborhood, True, rng_seed)


def test_local_search_score_iter(mocker: MockerFixture):
    m_score_solutions = mocker.patch.object(mod, "score_solutions")
    state = mocker.Mock(
        spec_set=["time", "n_calls", "best_score"], time=0, n_calls=0, best_score=0
    )

    solver = MyLocalSearch(
        objective_func,
        vectorized=False,
        max_time=None,
        max_calls=None,
        complete_poll=True,
        rng_seed=rng_seed,
    )
    assert solver.score_iter(state, neighborhood) is m_score_solutions.return_value
    m_score_solutions.assert_called_once_with(
        objective_func,
        neighborhood,
        max_time=None,
        max_eval=None,
        stop_score=None,
        random_order=False,
        rng_seed=rng_seed,
    )


def test_local_search_update(mocker: MockerFixture):
    mocker.patch("metahopt.local_search.process_time", return_value=1)
    solver = MyLocalSearch(objective_func)
    state = LocalSearchState(
        solver, 1.0, "sol1", time=3, n_iter=4, n_calls=4, success_direction=None
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
    assert solver.update(state, scoring_res, 0) == LocalSearchState(
        solver, 1.0, "sol1", time=1, n_iter=5, n_calls=5, success_direction=None
    )

    scoring_res = replace(scoring_res, score=0.0)
    assert solver.update(state, scoring_res, 0) == LocalSearchState(
        solver, 0.0, "sol2", time=1, n_iter=5, n_calls=5, success_direction=12
    )


def test_local_search_check_termination():
    base_solver = MyLocalSearch(objective_func)
    state = LocalSearchState(
        base_solver, 1.0, "sol", time=3, n_iter=4, n_calls=5, success_direction=None
    )
    assert base_solver.check_termination(state) is None

    solver = replace(base_solver, min_score=2)
    assert (
        solver.check_termination(replace(state, params=solver))
        is TerminationReason.min_score
    )

    solver = replace(base_solver, max_time=2)
    assert (
        solver.check_termination(replace(state, params=solver))
        is TerminationReason.max_time
    )

    solver = replace(base_solver, max_iter=3)
    assert (
        solver.check_termination(replace(state, params=solver))
        is TerminationReason.max_iter
    )

    solver = replace(base_solver, max_calls=3)
    assert (
        solver.check_termination(replace(state, params=solver))
        is TerminationReason.max_calls
    )


def test_local_search_solve_iter(mocker: MockerFixture):
    def objective_func(x):
        return dict(sol0=3, sol1=1, sol2=2)[x]

    solver = MyLocalSearch(objective_func, max_iter=1)
    end_state, term_reason, stats = solver.solve("sol0")
    assert end_state == LocalSearchState(solver, 1, "sol1", mocker.ANY, 1, 3, 0)
    assert term_reason is TerminationReason.max_iter
    assert len(stats) == 1


def test_local_search_solve_vectorized(mocker: MockerFixture):
    def objective_func(x):
        d = {"sol0": 3, "sol1": 1, "sol2": 2}
        return [d[v] for v in x]

    solver = MyLocalSearch(objective_func, vectorized=True, max_iter=1)
    end_state, term_reason, stats = solver.solve("sol0")
    assert end_state == LocalSearchState(solver, 1, "sol1", mocker.ANY, 1, 2, 0)
    assert term_reason is TerminationReason.max_iter
    assert len(stats) == 1
