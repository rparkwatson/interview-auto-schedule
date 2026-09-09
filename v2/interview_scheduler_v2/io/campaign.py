"""Assemble availability and interview-period inputs into the scheduling domain."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ..domain import (
    Interviewer,
    InterviewerGroup,
    LockedAssignment,
    SchedulingProblem,
    Slot,
)
from ..identifiers import normalize_name
from .source_parsers import (
    ImportNotice,
    ParsedAvailability,
    ParsedSlotSet,
    WorkbookSource,
    parse_adcom_availability,
    parse_student_availability,
    parse_student_schedule,
    parse_time_slot_workbook,
    reconcile_availability,
)


@dataclass(frozen=True, slots=True)
class CampaignImportResult:
    problem: SchedulingProblem
    notices: tuple[ImportNotice, ...]
    interviewer_ids: Mapping[tuple[InterviewerGroup, str], str]
    student_availability: ParsedAvailability
    adcom_availability: ParsedAvailability
    slot_set: ParsedSlotSet
    periods_need_configuration: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interviewer_ids",
            MappingProxyType(dict(self.interviewer_ids)),
        )


def _interviewers_from_availability(
    parsed: ParsedAvailability,
    group: InterviewerGroup,
    *,
    explicit_ids: Mapping[tuple[InterviewerGroup, str], str],
    historical_counts: Mapping[str, int],
) -> tuple[list[Interviewer], dict[tuple[InterviewerGroup, str], str]]:
    slots_by_name: dict[str, set[str]] = {}
    display_by_name: dict[str, str] = {}
    for name in parsed.names:
        normalized = normalize_name(name)
        display_by_name.setdefault(normalized, name)
        slots_by_name.setdefault(normalized, set())
    for entry in parsed.entries:
        if entry.slot_id is None:
            continue
        normalized = normalize_name(entry.interviewer_name)
        display_by_name.setdefault(normalized, entry.interviewer_name)
        slots_by_name.setdefault(normalized, set()).add(entry.slot_id)

    people: list[Interviewer] = []
    generated: dict[tuple[InterviewerGroup, str], str] = {}
    for normalized, name in display_by_name.items():
        explicit_id = explicit_ids.get((group, normalized))
        person = Interviewer.create(
            name=name,
            group=group,
            explicit_id=explicit_id,
            available_slot_ids=sorted(slots_by_name.get(normalized, set())),
        )
        prior = int(historical_counts.get(person.id, 0))
        person = Interviewer(
            id=person.id,
            name=person.name,
            group=person.group,
            available_slot_ids=person.available_slot_ids,
            historical_prior_count=prior,
        )
        people.append(person)
        generated[(group, normalized)] = person.id
    return people, generated


def _assemble_campaign(
    *,
    students: ParsedAvailability,
    adcoms: ParsedAvailability,
    slot_set: ParsedSlotSet,
    explicit_ids: Mapping[tuple[InterviewerGroup, str], str],
    historical_counts: Mapping[str, int],
    locked_assignments: Sequence[LockedAssignment],
    periods_need_configuration: bool,
) -> CampaignImportResult:
    student_people, student_ids = _interviewers_from_availability(
        students,
        InterviewerGroup.STUDENT,
        explicit_ids=explicit_ids,
        historical_counts=historical_counts,
    )
    adcom_people, adcom_ids = _interviewers_from_availability(
        adcoms,
        InterviewerGroup.ADCOM,
        explicit_ids=explicit_ids,
        historical_counts=historical_counts,
    )
    slots = tuple(
        Slot(
            id=item.id,
            start=item.start,
            end=item.end,
            capacity=item.capacity,
            target=item.target,
        )
        for item in slot_set.slots
    )
    problem = SchedulingProblem(
        interviewers=tuple(student_people + adcom_people),
        slots=slots,
        locked_assignments=tuple(locked_assignments),
    )
    notices = tuple(slot_set.notices + students.notices + adcoms.notices)
    return CampaignImportResult(
        problem=problem,
        notices=notices,
        interviewer_ids={**student_ids, **adcom_ids},
        student_availability=students,
        adcom_availability=adcoms,
        slot_set=slot_set,
        periods_need_configuration=periods_need_configuration,
    )


def prepare_campaign_from_availability(
    *,
    student_workbook: WorkbookSource,
    adcom_workbook: WorkbookSource,
    year: int,
    timezone_name: str = "America/New_York",
    explicit_ids: Mapping[tuple[InterviewerGroup, str], str] | None = None,
    historical_counts: Mapping[str, int] | None = None,
    locked_assignments: Sequence[LockedAssignment] = (),
    max_reconciliation_minutes: int = 30,
) -> CampaignImportResult:
    """Build a campaign whose candidate periods come from the Student workbook.

    Candidate periods initially carry zero capacity and target values. The
    administrative workflow replaces them with reviewed counts before solving.
    """

    explicit_ids = explicit_ids or {}
    historical_counts = historical_counts or {}
    student_schedule = parse_student_schedule(
        student_workbook,
        year=year,
        timezone_name=timezone_name,
    )
    if not student_schedule.slot_set.slots:
        raise ValueError(
            "No interview periods were found in the Student availability workbook."
        )
    students = reconcile_availability(
        student_schedule.availability,
        student_schedule.slot_set,
        max_time_shift_minutes=max_reconciliation_minutes,
    )
    adcoms = reconcile_availability(
        parse_adcom_availability(adcom_workbook, year=year),
        student_schedule.slot_set,
        max_time_shift_minutes=max_reconciliation_minutes,
    )
    return _assemble_campaign(
        students=students,
        adcoms=adcoms,
        slot_set=student_schedule.slot_set,
        explicit_ids=explicit_ids,
        historical_counts=historical_counts,
        locked_assignments=locked_assignments,
        periods_need_configuration=True,
    )


def import_campaign(
    *,
    student_workbook: WorkbookSource,
    adcom_workbook: WorkbookSource,
    slot_workbook: WorkbookSource,
    year: int,
    timezone_name: str = "America/New_York",
    explicit_ids: Mapping[tuple[InterviewerGroup, str], str] | None = None,
    historical_counts: Mapping[str, int] | None = None,
    locked_assignments: Sequence[LockedAssignment] = (),
    max_reconciliation_minutes: int = 30,
) -> CampaignImportResult:
    """Parse, reconcile, and combine a complete campaign entirely in memory.

    ``explicit_ids`` uses ``(group, normalized_name)`` keys. Historical counts
    use the resolved interviewer ID, which keeps counts stable even when two
    groups contain the same display name.
    """

    explicit_ids = explicit_ids or {}
    historical_counts = historical_counts or {}

    slot_set = parse_time_slot_workbook(
        slot_workbook,
        year=year,
        timezone_name=timezone_name,
    )
    students = reconcile_availability(
        parse_student_availability(student_workbook, year=year),
        slot_set,
        max_time_shift_minutes=max_reconciliation_minutes,
    )
    adcoms = reconcile_availability(
        parse_adcom_availability(adcom_workbook, year=year),
        slot_set,
        max_time_shift_minutes=max_reconciliation_minutes,
    )

    return _assemble_campaign(
        students=students,
        adcoms=adcoms,
        slot_set=slot_set,
        explicit_ids=explicit_ids,
        historical_counts=historical_counts,
        locked_assignments=locked_assignments,
        periods_need_configuration=False,
    )
