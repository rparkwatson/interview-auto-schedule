"""Immutable, solver-independent v2 scheduling domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

from .identifiers import interviewer_id


CAPACITY_UNITS_PER_ASSIGNMENT = 1


class InterviewerGroup(str, Enum):
    STUDENT = "student"
    ADCOM = "adcom"

    @property
    def label(self) -> str:
        return {
            InterviewerGroup.STUDENT: "Student Interviewer",
            InterviewerGroup.ADCOM: "Adcom Interviewer",
        }[self]


@dataclass(frozen=True, slots=True)
class Interviewer:
    id: str
    name: str
    group: InterviewerGroup
    available_slot_ids: frozenset[str] = field(default_factory=frozenset)
    historical_prior_count: int = 0
    preference_by_slot: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id).strip())
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "group", InterviewerGroup(self.group))
        object.__setattr__(
            self,
            "historical_prior_count",
            int(self.historical_prior_count),
        )
        object.__setattr__(
            self,
            "available_slot_ids",
            frozenset(str(slot_id).strip() for slot_id in self.available_slot_ids),
        )
        object.__setattr__(
            self,
            "preference_by_slot",
            MappingProxyType(
                {
                    str(slot_id).strip(): int(score)
                    for slot_id, score in dict(self.preference_by_slot).items()
                }
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        group: InterviewerGroup,
        explicit_id: str | None = None,
        available_slot_ids: Sequence[str] = (),
        historical_prior_count: int = 0,
        preference_by_slot: Mapping[str, int] | None = None,
    ) -> "Interviewer":
        return cls(
            id=interviewer_id(group, name, explicit_id),
            name=name,
            group=group,
            available_slot_ids=frozenset(available_slot_ids),
            historical_prior_count=historical_prior_count,
            preference_by_slot=preference_by_slot or {},
        )

    def preference_for(self, slot_id: str) -> int:
        """Return a 1-5 preference score; binary availability defaults to 1."""

        return self.preference_by_slot.get(slot_id, 1)


@dataclass(frozen=True, slots=True)
class Slot:
    id: str
    start: datetime
    end: datetime
    capacity: int
    target: int | None = None
    group_targets: Mapping[InterviewerGroup, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", str(self.id).strip())
        object.__setattr__(self, "capacity", int(self.capacity))
        object.__setattr__(
            self,
            "target",
            int(self.target) if self.target is not None else None,
        )
        targets = {
            InterviewerGroup(group): int(value)
            for group, value in dict(self.group_targets).items()
        }
        object.__setattr__(self, "group_targets", MappingProxyType(targets))

    @property
    def local_date(self) -> date:
        return self.start.date()


@dataclass(frozen=True, slots=True)
class LockedAssignment:
    interviewer_id: str
    slot_id: str
    assignment_date: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "interviewer_id", str(self.interviewer_id).strip())
        object.__setattr__(self, "slot_id", str(self.slot_id).strip())


def consecutive_listed_slot_pairs(slots: Sequence[Slot]) -> tuple[tuple[str, str], ...]:
    """Return same-day adjacency using input order, not inferred clock gaps."""

    return tuple(
        (left.id, right.id)
        for left, right in zip(slots, slots[1:])
        if left.local_date == right.local_date
    )


def overlapping_slot_pairs(slots: Sequence[Slot]) -> tuple[tuple[str, str], ...]:
    """Return authoritative slot pairs whose time intervals overlap."""

    pairs: list[tuple[str, str]] = []
    for index, left in enumerate(slots):
        for right in slots[index + 1 :]:
            try:
                overlaps = left.start < right.end and right.start < left.end
            except TypeError:
                # Mixed naive/aware inputs are reported by validation; overlap
                # detection must not mask that more useful diagnostic.
                continue
            if overlaps:
                pairs.append((left.id, right.id))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class SchedulingProblem:
    interviewers: tuple[Interviewer, ...]
    slots: tuple[Slot, ...]
    locked_assignments: tuple[LockedAssignment, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "interviewers", tuple(self.interviewers))
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "locked_assignments", tuple(self.locked_assignments))

    @property
    def adjacency(self) -> tuple[tuple[str, str], ...]:
        return consecutive_listed_slot_pairs(self.slots)

    @property
    def overlaps(self) -> tuple[tuple[str, str], ...]:
        return overlapping_slot_pairs(self.slots)
