from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, FrozenSet

InterviewerType = Literal["Regular", "Senior", "Observer"]

@dataclass(frozen=True)
class Slot:
    id: str
    start: datetime
    end: datetime
    day_key: str
    adjacent_forward: FrozenSet[str]

@dataclass(frozen=True)
class Interviewer:
    id: str
    name: str
    kind: InterviewerType
    max_daily: int
    max_total: int
    available_slots: FrozenSet[str]
    min_total: int = 0
    pre_assigned: int = 0

@dataclass(frozen=True)
class Inputs:
    interviewers: list[Interviewer]
    slots: list[Slot]
    max_pairs_per_slot: dict[str, int]
