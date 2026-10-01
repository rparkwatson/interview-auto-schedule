"""Pure review-table transformations and construction of solver inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pandas as pd

from .config import BackToBackPolicy, GroupPolicy, SchedulerConfig
from .counts import nonnegative_integer
from .domain import Interviewer, InterviewerGroup, SchedulingProblem, Slot
from .io import CampaignImportResult
from .presentation import format_interview_period

GROUP_BY_LABEL = {group.label: group for group in InterviewerGroup}


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _integer(value: Any, default: int = 0) -> int:
    if not _present(value) or str(value).strip() == "":
        return default
    return nonnegative_integer(value)


def _text(value: Any) -> str:
    return str(value).strip() if _present(value) else ""


def _people_frame(imported: CampaignImportResult) -> pd.DataFrame:
    defaults = SchedulerConfig().group_policies
    rows: list[dict[str, Any]] = []
    for person in imported.problem.interviewers:
        policy = defaults[person.group]
        rows.append(
            {
                "Enabled": True,
                "Interviewer ID": person.id,
                "Interviewer Name": person.name,
                "Group": person.group.label,
                "Availability Slots": len(person.available_slot_ids),
                "Historical Count": person.historical_prior_count,
                "Use Group Defaults": True,
                "Minimum": policy.min_total,
                "Target": policy.target_total,
                "Maximum": policy.max_total,
                "Maximum Per Day": policy.max_per_day,
                "Minimum Per Active Day": policy.min_per_active_day,
            }
        )
    return pd.DataFrame(rows)


def _availability_counts(
    imported: CampaignImportResult,
) -> tuple[dict[str, int], dict[str, int]]:
    student_counts = {slot.id: 0 for slot in imported.problem.slots}
    adcom_counts = {slot.id: 0 for slot in imported.problem.slots}
    for person in imported.problem.interviewers:
        destination = (
            student_counts
            if person.group is InterviewerGroup.STUDENT
            else adcom_counts
        )
        for slot_id in person.available_slot_ids:
            if slot_id in destination:
                destination[slot_id] += 1
    return student_counts, adcom_counts


def _slot_frame(imported: CampaignImportResult) -> pd.DataFrame:
    student_counts, adcom_counts = _availability_counts(imported)
    return pd.DataFrame(
        [
            {
                "Slot ID": slot.id,
                "Interview Period": format_interview_period(slot.start, slot.end),
                "Start": slot.start,
                "End": slot.end,
                "Capacity": (
                    None if imported.periods_need_configuration else slot.capacity
                ),
                "Student Available": student_counts.get(slot.id, 0),
                "Adcom Available": adcom_counts.get(slot.id, 0),
                "Student Target": None,
                "Adcom Target": None,
            }
            for slot in imported.problem.slots
        ]
    )


def _frame_value(value: Any) -> Any:
    """Normalize editor values so harmless dtype changes do not count as edits."""

    if not _present(value):
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return value


def _frame_signature(frame: pd.DataFrame) -> tuple[Any, ...]:
    return (
        tuple(str(column) for column in frame.columns),
        tuple(
            tuple(_frame_value(value) for value in row)
            for row in frame.itertuples(index=False, name=None)
        ),
    )


def _frames_differ(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    return _frame_signature(left) != _frame_signature(right)


def _period_setup_issues(frame: pd.DataFrame) -> list[str]:
    """Return plain-language blockers for the interview-count setup."""

    if frame.empty:
        return ["No interview periods were found."]
    blank_count = 0
    invalid_count = 0
    offered_count = 0
    for _, row in frame.iterrows():
        capacity_value = row.get("Capacity")
        if not _present(capacity_value) or str(capacity_value).strip() == "":
            blank_count += 1
            continue
        try:
            capacity = nonnegative_integer(capacity_value)
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        if capacity > 0:
            offered_count += 1

    issues: list[str] = []
    if blank_count:
        issues.append(
            f"Enter Interviews possible for {blank_count} remaining period"
            f"{'s' if blank_count != 1 else ''}."
        )
    if invalid_count:
        issues.append(
            f"Correct {invalid_count} count value{'s' if invalid_count != 1 else ''}; "
            "counts must be whole numbers of zero or more."
        )
    if not blank_count and not invalid_count and offered_count == 0:
        issues.append("Offer at least one interview period by entering 1 or more.")
    return issues


def _period_setup_totals(frame: pd.DataFrame) -> tuple[int, int, int]:
    configured = 0
    offered = 0
    interviews_possible = 0
    for _, row in frame.iterrows():
        if not _present(row.get("Capacity")):
            continue
        try:
            capacity = nonnegative_integer(row.get("Capacity"))
        except (TypeError, ValueError):
            continue
        configured += 1
        if capacity > 0:
            offered += 1
        interviews_possible += max(0, capacity)
    return configured, offered, interviews_possible


def _replace_period_counts(
    frame: pd.DataFrame,
    parsed_slots: Any,
) -> pd.DataFrame:
    counts = {slot.id: slot for slot in parsed_slots}
    updated = frame.copy()
    for row_index, row in updated.iterrows():
        slot = counts.get(_text(row.get("Slot ID")))
        if slot is None:
            continue
        updated.at[row_index, "Capacity"] = slot.capacity
    return updated


def _merge_people_review(
    current: pd.DataFrame,
    reviewed: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the short administrative table into the full policy table."""

    existing = {
        _text(row.get("Interviewer ID")): row.to_dict()
        for _, row in current.iterrows()
        if _text(row.get("Interviewer ID"))
    }
    defaults = SchedulerConfig().group_policies
    rows: list[dict[str, Any]] = []
    for _, reviewed_row in reviewed.iterrows():
        interviewer_id = _text(reviewed_row.get("Interviewer ID"))
        name = _text(reviewed_row.get("Interviewer Name"))
        group_label = _text(reviewed_row.get("Group"))
        group = GROUP_BY_LABEL.get(group_label, InterviewerGroup.ADCOM)
        if not interviewer_id and name:
            interviewer_id = Interviewer.create(name=name, group=group).id
        policy = defaults[group]
        base = existing.get(
            interviewer_id,
            {
                "Enabled": True,
                "Interviewer ID": interviewer_id or None,
                "Interviewer Name": name or None,
                "Group": group.label,
                "Availability Slots": 0,
                "Historical Count": 0,
                "Use Group Defaults": True,
                "Minimum": policy.min_total,
                "Target": policy.target_total,
                "Maximum": policy.max_total,
                "Maximum Per Day": policy.max_per_day,
                "Minimum Per Active Day": policy.min_per_active_day,
            },
        ).copy()
        for column in (
            "Enabled",
            "Interviewer ID",
            "Interviewer Name",
            "Group",
            "Availability Slots",
            "Historical Count",
        ):
            if column in reviewed_row:
                base[column] = reviewed_row[column]
        base["Interviewer ID"] = interviewer_id or base.get("Interviewer ID")
        rows.append(base)
    return pd.DataFrame(rows, columns=current.columns)


def _reviewed_problem_and_config(
    imported: CampaignImportResult,
    people_frame: pd.DataFrame,
    slot_frame: pd.DataFrame,
    *,
    student_defaults: GroupPolicy,
    adcom_defaults: GroupPolicy,
    student_priority_weight: int,
    back_to_back: BackToBackPolicy,
    maximum_consecutive: int,
    time_limit_seconds: float,
) -> tuple[SchedulingProblem, SchedulerConfig]:
    original_slots = {slot.id: slot for slot in imported.problem.slots}
    slots: list[Slot] = []
    for _, row in slot_frame.iterrows():
        slot_id = _text(row.get("Slot ID", ""))
        source_slot = original_slots.get(slot_id)
        if source_slot is None:
            continue
        capacity = _integer(row.get("Capacity"))
        # An explicit zero means that this candidate period is not being offered.
        if capacity == 0:
            continue
        group_targets: dict[InterviewerGroup, int] = {}
        if _present(row.get("Student Target")):
            group_targets[InterviewerGroup.STUDENT] = _integer(
                row.get("Student Target")
            )
        if _present(row.get("Adcom Target")):
            group_targets[InterviewerGroup.ADCOM] = _integer(
                row.get("Adcom Target")
            )
        slots.append(
            replace(
                source_slot,
                capacity=capacity,
                target=capacity,
                group_targets=group_targets,
            )
        )

    active_slot_ids = {slot.id for slot in slots}
    original_people = {person.id: person for person in imported.problem.interviewers}
    interviewers: list[Interviewer] = []
    person_policies: dict[str, GroupPolicy] = {}
    for _, row in people_frame.iterrows():
        if not bool(row.get("Enabled", True)):
            continue
        name = _text(row.get("Interviewer Name", ""))
        group = GROUP_BY_LABEL.get(_text(row.get("Group", "")))
        if not name or group is None:
            continue
        explicit_id = _text(row.get("Interviewer ID", "")) or None
        source_person = original_people.get(explicit_id or "")
        available = (
            source_person.available_slot_ids & active_slot_ids
            if source_person
            else frozenset()
        )
        preferences = (
            {
                slot_id: score
                for slot_id, score in source_person.preference_by_slot.items()
                if slot_id in active_slot_ids
            }
            if source_person
            else {}
        )
        person = Interviewer.create(
            name=name,
            group=group,
            explicit_id=explicit_id,
            available_slot_ids=available,
            historical_prior_count=_integer(row.get("Historical Count")),
            preference_by_slot=preferences,
        )
        interviewers.append(person)
        if not bool(row.get("Use Group Defaults", True)):
            person_policies[person.id] = GroupPolicy(
                min_total=_integer(row.get("Minimum")),
                target_total=_integer(row.get("Target")),
                max_total=_integer(row.get("Maximum")),
                max_per_day=_integer(row.get("Maximum Per Day")),
                min_per_active_day=_integer(row.get("Minimum Per Active Day")),
            )

    config = SchedulerConfig(
        group_policies={
            InterviewerGroup.STUDENT: student_defaults,
            InterviewerGroup.ADCOM: adcom_defaults,
        },
        person_policies=person_policies,
        back_to_back=back_to_back,
        max_consecutive_slots=maximum_consecutive,
        student_priority_weight=student_priority_weight,
        time_limit_seconds=time_limit_seconds,
        random_seed=2026,
        num_search_workers=1,
    )
    return SchedulingProblem(tuple(interviewers), tuple(slots)), config
