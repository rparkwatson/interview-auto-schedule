"""Policy configuration for the independent v2 scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .domain import InterviewerGroup


class BackToBackPolicy(str, Enum):
    DISCOURAGED = "discouraged"
    HARD = "hard"
    OFF = "off"


class RelaxationMode(str, Enum):
    """Explicitly controls which assignment limits may be exceeded."""

    STRICT = "strict"
    MINIMUMS = "minimums"
    MAXIMUMS = "maximums"
    MINIMUMS_AND_MAXIMUMS = "minimums_and_maximums"


@dataclass(frozen=True, slots=True)
class GroupPolicy:
    """Per-interviewer defaults for one canonical group."""

    min_total: int
    target_total: int
    max_total: int
    max_per_day: int
    min_per_active_day: int = 0


def default_group_policies() -> Mapping[InterviewerGroup, GroupPolicy]:
    return MappingProxyType(
        {
            InterviewerGroup.STUDENT: GroupPolicy(
                min_total=3,
                target_total=4,
                max_total=5,
                max_per_day=2,
                min_per_active_day=0,
            ),
            InterviewerGroup.ADCOM: GroupPolicy(
                min_total=4,
                target_total=4,
                max_total=6,
                max_per_day=2,
                min_per_active_day=0,
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    group_policies: Mapping[InterviewerGroup, GroupPolicy] = field(
        default_factory=default_group_policies
    )
    person_policies: Mapping[str, GroupPolicy] = field(default_factory=dict)
    back_to_back: BackToBackPolicy = BackToBackPolicy.DISCOURAGED
    max_consecutive_slots: int = 2
    capacity_units_per_assignment: int = 1
    student_priority_weight: int = 3
    time_limit_seconds: float = 30.0
    random_seed: int = 2026
    num_search_workers: int = 1
    def __post_init__(self) -> None:
        policies = {
            InterviewerGroup(group): policy
            for group, policy in dict(self.group_policies).items()
        }
        object.__setattr__(self, "group_policies", MappingProxyType(policies))
        object.__setattr__(
            self,
            "person_policies",
            MappingProxyType(
                {
                    str(interviewer_id).strip(): policy
                    for interviewer_id, policy in dict(self.person_policies).items()
                }
            ),
        )
        object.__setattr__(self, "back_to_back", BackToBackPolicy(self.back_to_back))
        if self.capacity_units_per_assignment != 1:
            raise ValueError("v2 requires exactly one capacity unit per assignment")
        if self.max_consecutive_slots < 1:
            raise ValueError("max_consecutive_slots must be at least 1")
        if self.student_priority_weight < 1:
            raise ValueError("student_priority_weight must be at least 1")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if self.num_search_workers < 1:
            raise ValueError("num_search_workers must be at least 1")

    def policy_for(self, interviewer_id: str, group: InterviewerGroup) -> GroupPolicy:
        """Resolve a per-person override before the canonical group default."""

        try:
            return self.person_policies[interviewer_id]
        except KeyError:
            return self.group_policies[InterviewerGroup(group)]


DEFAULT_CONFIG = SchedulerConfig()
