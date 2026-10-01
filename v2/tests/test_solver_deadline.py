"""Exercise real CP-SAT models with a controlled clock and stage termination."""

from dataclasses import replace
import json

import pytest
from ortools.sat.python import cp_model

from interview_scheduler_v2 import Interviewer, InterviewerGroup, SchedulingProblem
from interview_scheduler_v2.optimization import SolveStatus, solve
from interview_scheduler_v2.optimization import cpsat
from test_solver import config, make_slot


def _problem():
    slot = make_slot("one", 8)
    person = Interviewer.create(name="One", group=InterviewerGroup.STUDENT,
                                available_slot_ids=[slot.id])
    return SchedulingProblem((person,), (slot,))


def _controlled_solver(monkeypatch, *, timeout_stage=None, first_elapsed=1.0,
                       feasible_stage=None):
    now = [0.0]
    budgets = []
    real_solver = cp_model.CpSolver

    class StageSolver:
        def __init__(self):
            self.delegate = real_solver()

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def Solve(self, model):
            stage = len(budgets)
            budgets.append(self.parameters.max_time_in_seconds)
            now[0] += first_elapsed if stage == 0 else 0.25
            if stage == timeout_stage:
                return cp_model.UNKNOWN
            status = self.delegate.Solve(model)
            assert status == cp_model.OPTIMAL
            return cp_model.FEASIBLE if stage == feasible_stage else status

    monkeypatch.setattr(cpsat, "perf_counter", lambda: now[0])
    monkeypatch.setattr(cpsat.cp_model, "CpSolver", StageSolver)
    return budgets


def test_early_stage_can_use_full_budget_and_unused_time_carries_forward(monkeypatch):
    budgets = _controlled_solver(monkeypatch)
    result = solve(_problem(), scenario="Budget", config=config())
    assert result.status == SolveStatus.OPTIMAL
    assert budgets[0] == 5.0
    assert budgets[1] == 4.0
    assert all(left > right > 0 for left, right in zip(budgets, budgets[1:]))
    stages = json.loads(result.settings["solver_stages_json"])
    assert all(stage["status"] == "OPTIMAL" for stage in stages)
    assert all(stage["objective"] == stage["best_bound"] for stage in stages)


def test_first_stage_timeout_has_no_usable_schedule(monkeypatch):
    budgets = _controlled_solver(monkeypatch, timeout_stage=0)
    result = solve(_problem(), scenario="Timeout", config=config())
    assert result.status == SolveStatus.UNKNOWN
    assert not result.assignments
    assert budgets == [5.0]


@pytest.mark.parametrize("exhaust_deadline", [False, True])
def test_later_timeout_preserves_incumbent_and_never_claims_optimality(monkeypatch, exhaust_deadline):
    budgets = _controlled_solver(monkeypatch, timeout_stage=1,
                                 first_elapsed=5.0 if exhaust_deadline else 1.0)
    result = solve(_problem(), scenario="Fallback", config=config())
    assert result.status == SolveStatus.FEASIBLE
    assert len(result.assignments) == 1
    assert result.assignments[0].slot_id == "one"
    assert result.objective_metrics["quality_stage_timeout_fallback"] == 1
    assert any(item.code == "SOLUTION_NOT_PROVEN_OPTIMAL" for item in result.diagnostics)
    stages = json.loads(result.settings["solver_stages_json"])
    assert stages[-1]["status"] == ("SKIPPED_TIME_LIMIT" if exhaust_deadline else "UNKNOWN")
    assert len(budgets) == (1 if exhaust_deadline else 2)


def test_unproven_earlier_objective_keeps_feasible_status(monkeypatch):
    _controlled_solver(monkeypatch, feasible_stage=0)
    result = solve(_problem(), scenario="Unproven", config=config())
    assert result.status == SolveStatus.FEASIBLE
    assert json.loads(result.settings["solver_stages_json"])[0]["status"] == "FEASIBLE"


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), 0.0])
def test_time_limit_must_be_finite_and_positive(limit):
    with pytest.raises(ValueError, match="finite and positive"):
        replace(config(), time_limit_seconds=limit)
