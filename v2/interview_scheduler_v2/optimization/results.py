"""Solver result objects shared by the UI and report writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..domain import InterviewerGroup


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Assignment:
    scenario: str
    slot_id: str
    start: datetime
    end: datetime
    interviewer_id: str
    interviewer_name: str
    group: InterviewerGroup
    locked: bool
    preference: int


@dataclass(frozen=True, slots=True)
class InterviewerSummary:
    interviewer_id: str
    interviewer_name: str
    group: InterviewerGroup
    historical_prior_count: int
    new_assignments: int
    cumulative_total: int
    minimum: int
    target: int
    maximum: int
    max_per_day: int
    minimum_shortfall: int
    target_shortfall: int
    maximum_overage: int
    active_days: int
    maximum_assigned_on_day: int
    back_to_back_pairs: int


@dataclass(frozen=True, slots=True)
class SlotSummary:
    slot_id: str
    start: datetime
    end: datetime
    target: int
    capacity: int
    assigned: int
    target_deficit: int
    remaining_capacity: int
    student_assigned: int
    adcom_assigned: int
    student_target: int | None
    adcom_target: int | None


@dataclass(frozen=True, slots=True)
class ConstraintDiagnostic:
    severity: str
    code: str
    message: str
    constraint: str
    interviewer_id: str | None = None
    interviewer_name: str | None = None
    group: InterviewerGroup | None = None
    slot_id: str | None = None
    assignment_date: date | None = None
    expected: int | str | None = None
    actual: int | str | None = None


@dataclass(frozen=True, slots=True)
class SolveResult:
    status: SolveStatus
    scenario: str
    assignments: tuple[Assignment, ...] = ()
    interviewer_summaries: tuple[InterviewerSummary, ...] = ()
    slot_summaries: tuple[SlotSummary, ...] = ()
    diagnostics: tuple[ConstraintDiagnostic, ...] = ()
    objective_metrics: Mapping[str, int | float] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SolveStatus(self.status))
        object.__setattr__(
            self,
            "objective_metrics",
            MappingProxyType(dict(self.objective_metrics)),
        )
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))

    @property
    def succeeded(self) -> bool:
        return self.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}

    @property
    def minimum_shortfalls(self) -> tuple[InterviewerSummary, ...]:
        return tuple(
            summary
            for summary in self.interviewer_summaries
            if summary.minimum_shortfall > 0
        )
