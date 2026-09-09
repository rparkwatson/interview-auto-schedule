"""Administrative-language presentation for technical scheduling messages."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re
from typing import Iterable, Sequence

from ..io.source_parsers import ImportNotice
from ..optimization.results import ConstraintDiagnostic
from ..validation import Severity, ValidationIssue


class AdminMessageLevel(str, Enum):
    BLOCKING = "blocking"
    REVIEW = "review"
    ADJUSTMENT = "adjustment"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class AdminMessage:
    key: str
    level: AdminMessageLevel
    title: str
    summary: str
    action: str | None = None
    details: tuple[str, ...] = ()
    technical_codes: tuple[str, ...] = ()
    affected_count: int = 1

    @property
    def can_continue(self) -> bool:
        return self.level is not AdminMessageLevel.BLOCKING


def format_interview_period(start: datetime, end: datetime) -> str:
    day = start.strftime("%a, %b %d").replace(" 0", " ")
    start_clock = start.strftime("%I:%M %p").lstrip("0")
    end_clock = end.strftime("%I:%M %p").lstrip("0")
    return f"{day} · {start_clock}–{end_clock}"


def _group_by_code(items: Iterable, code_attribute: str = "code") -> dict[str, list]:
    grouped: dict[str, list] = defaultdict(list)
    for item in items:
        grouped[str(getattr(item, code_attribute))].append(item)
    return dict(grouped)


def _detail(message: str, source: str | None = None) -> str:
    return f"{source}: {message}" if source else message


def _generic_level(severity: str) -> AdminMessageLevel:
    if severity == "error":
        return AdminMessageLevel.BLOCKING
    if severity == "warning":
        return AdminMessageLevel.REVIEW
    return AdminMessageLevel.ADJUSTMENT


def present_import_notices(
    notices: Sequence[ImportNotice],
) -> tuple[AdminMessage, ...]:
    messages: list[AdminMessage] = []
    for code, grouped in _group_by_code(notices).items():
        first = grouped[0]
        details = tuple(_detail(item.message, item.source) for item in grouped)
        count = len(grouped)
        if code == "slot_time_reconciled":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.ADJUSTMENT,
                    title="Availability times were matched automatically",
                    summary=(
                        f"The tool matched {count} availability time"
                        f"{'s' if count != 1 else ''} to the corresponding interview "
                        "periods found in the Student file."
                    ),
                    action="Review the affected dates if those source times should not be adjusted.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "adcom_roster_missing":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Check for missing Adcom interviewers",
                    summary=(
                        "The Adcom file lists people found in the availability grid, "
                        "but it does not contain a separate complete roster."
                    ),
                    action="Add any missing Adcom interviewers in the next step.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "student_roster_missing":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Check for missing Student interviewers",
                    summary="The Student file does not contain a separate complete roster.",
                    action="Add anyone with no availability in the next step.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "availability_slot_ignored":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Some availability could not be matched",
                    summary=(
                        f"{count} availability period{'s' if count != 1 else ''} "
                        "did not match a period in the interview schedule and were not used."
                    ),
                    action=(
                        "Review the affected rows and correct the availability file "
                        "before uploading it again."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code in {"name_whitespace_normalized", "duplicate_availability_removed"}:
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.ADJUSTMENT,
                    title=(
                        "Name formatting was cleaned up"
                        if code == "name_whitespace_normalized"
                        else "Duplicate availability entries were removed"
                    ),
                    summary=(
                        "Extra spacing in interviewer names was corrected automatically."
                        if code == "name_whitespace_normalized"
                        else "Repeated copies of the same availability were kept only once."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        else:
            level = _generic_level(first.severity)
            if code in {
                "invalid_slot_label",
                "invalid_capacity",
                "negative_capacity",
                "invalid_target",
                "target_outside_capacity",
            }:
                summary = (
                    "One or more interview-period rows contains a date, time, "
                    "staffing need, or maximum that could not be used."
                )
            elif code == "duplicate_slot":
                summary = "The same interview period appears more than once in the schedule file."
            elif code in {"student_name_not_in_roster", "duplicate_student_roster_name"}:
                summary = (
                    "A Student interviewer name is missing from the roster or appears "
                    "more than once."
                )
            elif code == "conflicting_student_period":
                summary = (
                    "The Student file defines the same interview period with more "
                    "than one end time."
                )
            elif code == "sheet_ignored_no_time_range":
                summary = (
                    "A worksheet was skipped because no recognizable interview time "
                    "range was found."
                )
            else:
                summary = "One or more uploaded rows needs to be reviewed."
            messages.append(
                AdminMessage(
                    key=code,
                    level=level,
                    title=(
                        "A file needs to be corrected"
                        if level is AdminMessageLevel.BLOCKING
                        else "Review imported information"
                    ),
                    summary=summary,
                    action=(
                        "Correct the affected source rows and upload the files again."
                        if level is AdminMessageLevel.BLOCKING
                        else "Review the details before creating the schedule."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
    return tuple(
        sorted(
            messages,
            key=lambda item: (
                {
                    AdminMessageLevel.BLOCKING: 0,
                    AdminMessageLevel.REVIEW: 1,
                    AdminMessageLevel.ADJUSTMENT: 2,
                    AdminMessageLevel.COMPLETE: 3,
                }[item.level],
                item.title,
            ),
        )
    )


def present_validation_issues(
    issues: Sequence[ValidationIssue],
) -> tuple[AdminMessage, ...]:
    messages: list[AdminMessage] = []
    for code, grouped in _group_by_code(issues).items():
        first = grouped[0]
        count = len(grouped)
        details = tuple(item.message for item in grouped)
        level = (
            AdminMessageLevel.BLOCKING
            if first.severity is Severity.ERROR
            else AdminMessageLevel.REVIEW
        )
        if code == "TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM":
            context = first.context
            requested = context.get("target_demand", "the requested number of")
            available = context.get("assignable_upper_bound", "fewer")
            shortfall = context.get("unavoidable_target_deficit", "some")
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Not all requested interview seats can be filled",
                    summary=(
                        f"You requested {requested} interview assignments. Current "
                        f"availability and assignment limits allow up to {available}, "
                        f"so at least {shortfall} seat{'s' if shortfall != 1 else ''} "
                        "will remain unfilled."
                    ),
                    action="Add availability, increase assignment limits, or reduce the interviews needed.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "MIN_TOTAL_INFEASIBLE":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Some interviewers cannot reach their required minimum",
                    summary=(
                        f"Current availability and limits prevent {count} interviewer"
                        f"{'s' if count != 1 else ''} from receiving the required number of assignments."
                    ),
                    action="Add availability, lower the minimum, or use an exception option after the standard attempt fails.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "AGGREGATE_MINIMUM_EXCEEDS_CAPACITY":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Required minimums exceed the available interview seats",
                    summary=(
                        "The combined minimum assignments require more seats than "
                        "the interview schedule currently contains."
                    ),
                    action=(
                        "Add interview seats, lower one or more minimums, or use a "
                        "minimum exception after the standard attempt fails."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code in {"NO_INTERVIEWERS", "NO_SLOTS"}:
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title=(
                        "At least one interviewer is needed"
                        if code == "NO_INTERVIEWERS"
                        else "At least one interview period is needed"
                    ),
                    summary=(
                        "No included interviewers remain in the reviewed list."
                        if code == "NO_INTERVIEWERS"
                        else "No usable interview periods remain in the reviewed schedule."
                    ),
                    action="Return to Step 2, add the missing information, and try again.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code in {
            "INVALID_TOTAL_POLICY_ORDER",
            "INVALID_DAILY_POLICY_ORDER",
            "NEGATIVE_GROUP_POLICY",
        }:
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Assignment rules need to be corrected",
                    summary=(
                        "Minimums, preferred totals, maximums, and daily limits "
                        "must be entered in a valid order."
                    ),
                    action="Check the assignment rules and try again.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "PRIOR_EXCEEDS_MAX_TOTAL":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="A prior interview count exceeds the selected maximum",
                    summary=(
                        "One or more people already has more interviews recorded than "
                        "the maximum currently allowed."
                    ),
                    action=(
                        "Review Interviews already assigned and the person's limits "
                        "in Step 2."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code in {
            "LOCKS_EXCEED_MAX_TOTAL",
            "LOCKS_EXCEED_MAX_PER_DAY",
            "LOCKS_EXCEED_MAX_CONSECUTIVE",
            "LOCKS_EXCEED_SLOT_CAPACITY",
        }:
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Existing assignments conflict with the selected limits",
                    summary=(
                        "One or more prior or preassigned interviews exceeds a person, "
                        "day, consecutive-period, or interview-period limit."
                    ),
                    action="Review prior counts, preassigned interviews, and limits in Step 2.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code.startswith("LOCK") or code.startswith("UNKNOWN_LOCKED"):
            messages.append(
                AdminMessage(
                    key=code,
                    level=level,
                    title="A preassigned interview needs attention",
                    summary=(
                        "A preassigned interview conflicts with availability, dates, "
                        "another assignment, or an interview-period limit."
                    ),
                    action="Review the selected interviewer and interview period.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        else:
            family_copy = {
                "identity": "A person or interview period is missing information or appears more than once.",
                "slot_time": "One or more interview dates or times is incomplete or conflicts with another period.",
                "capacity": "The number of assignments requested is not compatible with an available limit.",
                "target": "A requested staffing value cannot be used as entered.",
                "availability": "An availability entry does not match the reviewed interview schedule.",
                "total_limit": "A person's prior, minimum, or maximum assignment totals conflict.",
                "daily_limit": "A person's same-day assignment rules conflict.",
                "lock": "A preassigned interview needs to be reviewed.",
                "consecutive": "A consecutive-interview rule needs to be reviewed.",
                "config": "One or more assignment rules is incomplete or inconsistent.",
            }
            messages.append(
                AdminMessage(
                    key=code,
                    level=level,
                    title=(
                        "Information needs to be corrected"
                        if level is AdminMessageLevel.BLOCKING
                        else "Review recommended"
                    ),
                    summary=family_copy.get(
                        first.family.value,
                        "One or more reviewed values needs attention.",
                    ),
                    action=(
                        "Correct the affected information before creating the schedule."
                        if level is AdminMessageLevel.BLOCKING
                        else "Review this item before distributing the schedule."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
    return tuple(messages)


def present_diagnostics(
    diagnostics: Sequence[ConstraintDiagnostic],
) -> tuple[AdminMessage, ...]:
    messages: list[AdminMessage] = []
    for code, grouped in _group_by_code(diagnostics).items():
        first = grouped[0]
        details = tuple(item.message for item in grouped)
        count = len(grouped)
        if code == "MINIMUM_RELAXED":
            total = sum(max(0, int(item.expected or 0) - int(item.actual or 0)) for item in grouped)
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Some interviewers are below their required minimum",
                    summary=(
                        f"{count} interviewer{'s are' if count != 1 else ' is'} below "
                        f"the required minimum by {total} assignment{'s' if total != 1 else ''} in total."
                    ),
                    action="Review the named interviewers before approving this schedule.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "MIN_TOTAL_INFEASIBLE":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Some interviewers cannot reach their required minimum",
                    summary=(
                        f"Current availability and limits prevent {count} interviewer"
                        f"{'s' if count != 1 else ''} from receiving the required "
                        "number of assignments."
                    ),
                    action=(
                        "Add availability, lower the minimum, or choose a minimum "
                        "exception below."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code in {"MAXIMUM_RELAXED", "DAILY_MAXIMUM_RELAXED"}:
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Some assignment limits were exceeded",
                    summary=(
                        f"{count} interviewer assignment limit"
                        f"{'s were' if count != 1 else ' was'} exceeded to create this schedule."
                    ),
                    action="Review the affected interviewers before approving this schedule.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "SLOT_TARGET_UNMET":
            total = sum(max(0, int(item.expected or 0) - int(item.actual or 0)) for item in grouped)
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Some interview periods are not fully staffed",
                    summary=(
                        f"{count} interview period{'s have' if count != 1 else ' has'} "
                        f"a combined total of {total} unfilled interview seat"
                        f"{'s' if total != 1 else ''}."
                    ),
                    action="Review the affected periods before distributing the schedule.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "GROUP_SLOT_TARGET_UNMET":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Some group staffing preferences were not met",
                    summary=(
                        f"{count} interview period{'s do' if count != 1 else ' does'} "
                        "not have the requested Student or Adcom mix."
                    ),
                    action="Review the affected periods and group assignments.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "BACK_TO_BACK_ASSIGNED":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Consecutive interviews were scheduled",
                    summary=(
                        f"{count} interviewer{'s have' if count != 1 else ' has'} "
                        "one or more consecutive interview periods."
                    ),
                    action="Review these assignments if consecutive interviews should be changed.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "SOLUTION_NOT_PROVEN_OPTIMAL":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.ADJUSTMENT,
                    title="A valid schedule was created within the time limit",
                    summary=(
                        "The schedule follows the selected rules, although the tool "
                        "did not finish comparing every possible alternative."
                    ),
                    action="You may use this schedule or run it again with a longer advanced time limit.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM":
            # Validation context is flattened when it becomes a solver
            # diagnostic. Recover the three useful counts when they are present,
            # while keeping the original text only in the detail view.
            count_match = re.search(
                r"target demand\s+(\d+).*?upper bound\s+(\d+).*?at least\s+(\d+)",
                first.message,
                flags=re.IGNORECASE,
            )
            if count_match:
                requested, possible, shortfall = (
                    int(value) for value in count_match.groups()
                )
                summary = (
                    f"You requested {requested} interview assignments. Current "
                    f"availability and assignment limits allow up to {possible}, "
                    f"so at least {shortfall} seat"
                    f"{'s' if shortfall != 1 else ''} will remain unfilled."
                )
            else:
                summary = (
                    "The number of interviews requested is higher than current "
                    "availability and assignment limits can support."
                )
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Not all requested interview seats can be filled",
                    summary=summary,
                    action=(
                        "Add availability, increase assignment limits, reduce the "
                        "interviews needed, or review the unfilled periods before "
                        "distributing the schedule."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "GROUP_TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="A group staffing preference cannot be fully met",
                    summary=(
                        "The preferred Student or Adcom staffing count is higher "
                        "than that group's availability and assignment limits allow."
                    ),
                    action="Review the group mix in the affected interview periods.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "ACTIVE_DAY_MINIMUM_RELAXED":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.REVIEW,
                    title="Some same-day minimums were not met",
                    summary=(
                        f"{count} interviewer day{'s' if count != 1 else ''} received "
                        "fewer assignments than the selected same-day minimum."
                    ),
                    action="Review the affected interviewers and dates before approval.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "AGGREGATE_MINIMUM_EXCEEDS_CAPACITY":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Required minimums exceed the available interview seats",
                    summary=(
                        "The combined required assignments are greater than the "
                        "number of interview seats in the current schedule."
                    ),
                    action=(
                        "Add interview seats, lower minimums, or choose a minimum "
                        "exception below."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code == "PRIOR_EXCEEDS_MAX_TOTAL":
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="A prior interview count exceeds the selected maximum",
                    summary=(
                        "One or more people already has more interviews recorded than "
                        "the maximum currently allowed."
                    ),
                    action=(
                        "Review Interviews already assigned and the person's limits "
                        "in Step 2."
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code in {
            "LOCKS_EXCEED_MAX_TOTAL",
            "LOCKS_EXCEED_MAX_PER_DAY",
            "LOCKS_EXCEED_MAX_CONSECUTIVE",
            "LOCKS_EXCEED_SLOT_CAPACITY",
        }:
            messages.append(
                AdminMessage(
                    key=code,
                    level=AdminMessageLevel.BLOCKING,
                    title="Existing assignments conflict with the selected limits",
                    summary=(
                        "One or more prior or preassigned interviews exceeds a person, "
                        "day, consecutive-period, or interview-period limit."
                    ),
                    action="Review prior counts, preassigned interviews, and limits in Step 2.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        elif code.startswith("LOCK") or code.startswith("UNKNOWN_LOCKED"):
            messages.append(
                AdminMessage(
                    key=code,
                    level=(
                        AdminMessageLevel.BLOCKING
                        if first.severity == "error"
                        else AdminMessageLevel.REVIEW
                    ),
                    title="A preassigned interview needs attention",
                    summary=(
                        "A preassigned interview conflicts with availability, dates, "
                        "another assignment, or an interview-period limit."
                    ),
                    action="Review the selected interviewer and interview period in Step 2.",
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
        else:
            level = _generic_level(first.severity)
            constraint_copy = {
                "identity": "A person or interview period is missing information or appears more than once.",
                "slot_time": "One or more interview dates or times is incomplete or conflicts with another period.",
                "capacity": "The number of assignments requested is not compatible with an available limit.",
                "target": "A requested staffing value cannot be met as entered.",
                "availability": "An availability entry does not match the reviewed interview schedule.",
                "total_limit": "A person's prior, minimum, or maximum assignment totals conflict.",
                "daily_limit": "A person's same-day assignment rules conflict.",
                "lock": "A preassigned interview needs to be reviewed.",
                "consecutive": "A consecutive-interview rule needs to be reviewed.",
                "config": "One or more assignment rules is incomplete or inconsistent.",
            }
            messages.append(
                AdminMessage(
                    key=code,
                    level=level,
                    title=(
                        "A schedule item needs attention"
                        if level is not AdminMessageLevel.ADJUSTMENT
                        else "Scheduling note"
                    ),
                    summary=constraint_copy.get(
                        first.constraint,
                        "One or more schedule items needs attention.",
                    ),
                    action=(
                        "Review this item before distributing the schedule."
                        if level is AdminMessageLevel.REVIEW
                        else None
                    ),
                    details=details,
                    technical_codes=(code,),
                    affected_count=count,
                )
            )
    return tuple(messages)
