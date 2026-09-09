"""Build the required scenario workbook entirely in memory."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from io import BytesIO
import re
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import xlsxwriter

from ..config import SchedulerConfig
from ..domain import InterviewerGroup, SchedulingProblem
from ..io.source_parsers import ImportNotice
from ..optimization.results import SolveResult


REQUIRED_SHEETS = (
    "Assignments",
    "Schedule_By_Slot",
    "Interviewer_Summary",
    "Student_only Schedule_by_slot",
    "Adcom_only Schedule_by_slot",
    "Group_Summary",
    "Slot_Summary",
    "Constraint_Diagnostics",
    "Run_Settings",
)

SIMPLIFIED_SCHEDULE_SHEET = "Schedule"


def _scenario_stem(scenario: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(scenario).strip()).strip("_")
    return safe[:80] or "scenario"


def scenario_filename(
    scenario: str,
    *,
    generated_at: datetime | None = None,
) -> str:
    generated_at = generated_at or datetime.now(ZoneInfo("America/New_York"))
    return (
        f"{_scenario_stem(scenario)}_interview_schedule_"
        f"{generated_at:%Y%m%d_%H%M%S}.xlsx"
    )


def simplified_schedule_filename(
    scenario: str,
    *,
    generated_at: datetime | None = None,
) -> str:
    """Return a scenario-specific filename for the simplified schedule."""

    generated_at = generated_at or datetime.now(ZoneInfo("America/New_York"))
    return (
        f"{_scenario_stem(scenario)}_simplified_schedule_"
        f"{generated_at:%Y%m%d_%H%M%S}.xlsx"
    )


def _excel_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=None)


def _group_label(group: InterviewerGroup | None) -> str:
    return group.label if group is not None else ""


def _group_column_header(position: int) -> str:
    """Return Group A through Group Z, then Group AA and beyond."""

    if position < 1:
        raise ValueError("Group column positions start at 1.")
    letters = ""
    value = position
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return f"Group {letters}"


def _write_table(
    workbook: xlsxwriter.Workbook,
    worksheet: xlsxwriter.worksheet.Worksheet,
    *,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    table_name: str,
    formats: dict[str, Any],
    widths: dict[str, float] | None = None,
) -> None:
    row_values = list(rows)
    for column, header in enumerate(headers):
        worksheet.write(0, column, header, formats["header"])
    for row_index, row in enumerate(row_values, start=1):
        for column, value in enumerate(row):
            if isinstance(value, datetime):
                worksheet.write_datetime(
                    row_index,
                    column,
                    _excel_datetime(value),
                    formats["datetime"],
                )
            elif isinstance(value, bool):
                worksheet.write(row_index, column, "Yes" if value else "No")
            else:
                worksheet.write(row_index, column, value)
    if row_values:
        worksheet.add_table(
            0,
            0,
            len(row_values),
            len(headers) - 1,
            {
                "name": table_name,
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in headers],
            },
        )
    else:
        worksheet.autofilter(0, 0, 0, len(headers) - 1)
    worksheet.freeze_panes(1, 0)
    for column, header in enumerate(headers):
        default_width = max(11, min(28, len(header) + 2))
        worksheet.set_column(column, column, (widths or {}).get(header, default_width))


def build_simplified_schedule_workbook(result: SolveResult) -> bytes:
    """Return a one-sheet schedule organized by period and concurrent group."""

    output = BytesIO()
    workbook = xlsxwriter.Workbook(
        output,
        {"in_memory": True, "remove_timezone": True},
    )
    workbook.set_properties(
        {
            "title": f"Simplified interview schedule: {result.scenario}",
            "subject": "Interviewers assigned by date, start time, and group",
            "author": "Interview Scheduler v2",
            "comments": "Generated from the same scheduling result as the full report.",
        }
    )
    header_format = workbook.add_format(
        {
            "bold": True,
            "font_color": "#FFFFFF",
            "bg_color": "#003B5C",
            "border": 1,
            "align": "center",
            "valign": "vcenter",
        }
    )
    date_format = workbook.add_format({"num_format": "mm/dd"})
    time_format = workbook.add_format({"num_format": "h:mm AM/PM"})
    unavailable_format = workbook.add_format({"bg_color": "#E7E6E6"})

    worksheet = workbook.add_worksheet(SIMPLIFIED_SCHEDULE_SHEET)
    worksheet.hide_gridlines(2)
    worksheet.freeze_panes(1, 2)

    assignments_by_slot: dict[str, list] = defaultdict(list)
    for assignment in result.assignments:
        assignments_by_slot[assignment.slot_id].append(assignment)

    slot_summaries = sorted(
        result.slot_summaries,
        key=lambda item: (item.start, item.slot_id),
    )
    maximum_groups = max(
        (
            max(
                slot.capacity,
                len(assignments_by_slot.get(slot.slot_id, ())),
            )
            for slot in slot_summaries
        ),
        default=0,
    )
    headers = (
        "Slot Date",
        "Start Time",
        *(_group_column_header(index) for index in range(1, maximum_groups + 1)),
    )
    for column, header in enumerate(headers):
        worksheet.write(0, column, header, header_format)

    for row_index, slot in enumerate(slot_summaries, start=1):
        start = _excel_datetime(slot.start)
        worksheet.write_datetime(row_index, 0, start, date_format)
        worksheet.write_datetime(row_index, 1, start, time_format)
        assignments = assignments_by_slot.get(slot.slot_id, ())
        for group_index in range(maximum_groups):
            column = group_index + 2
            if group_index < len(assignments):
                worksheet.write(
                    row_index,
                    column,
                    assignments[group_index].interviewer_name,
                )
            elif group_index >= slot.capacity:
                worksheet.write_blank(
                    row_index,
                    column,
                    None,
                    unavailable_format,
                )

    if slot_summaries:
        worksheet.add_table(
            0,
            0,
            len(slot_summaries),
            len(headers) - 1,
            {
                "name": "SimplifiedScheduleTable",
                "style": "Table Style Medium 2",
                "columns": [{"header": header} for header in headers],
            },
        )
    else:
        worksheet.autofilter(0, 0, 0, len(headers) - 1)

    worksheet.set_row(0, 24)
    worksheet.set_column(0, 0, 11)
    worksheet.set_column(1, 1, 13)
    if maximum_groups:
        worksheet.set_column(2, maximum_groups + 1, 24)

    workbook.close()
    return output.getvalue()


def build_workbook(
    result: SolveResult,
    problem: SchedulingProblem,
    config: SchedulerConfig,
    *,
    import_notices: Sequence[ImportNotice] = (),
) -> bytes:
    """Return the required report as XLSX bytes without a shared output file."""

    output = BytesIO()
    workbook = xlsxwriter.Workbook(
        output,
        {"in_memory": True, "remove_timezone": True},
    )
    workbook.set_properties(
        {
            "title": f"Interview schedule: {result.scenario}",
            "subject": "Student and Adcom interviewer assignments",
            "author": "Interview Scheduler v2",
            "comments": "Generated in memory; no shared report file was written.",
        }
    )
    formats = {
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#003B5C",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "datetime": workbook.add_format({"num_format": "yyyy-mm-dd hh:mm AM/PM"}),
        "key": workbook.add_format(
            {"bold": True, "font_color": "#003B5C", "bg_color": "#DCE6F1"}
        ),
        "warning": workbook.add_format({"bg_color": "#FFF2CC"}),
        "error": workbook.add_format({"bg_color": "#F4CCCC"}),
    }
    worksheets = {
        sheet_name: workbook.add_worksheet(sheet_name)
        for sheet_name in REQUIRED_SHEETS
    }

    assignments_by_slot: dict[str, list] = defaultdict(list)
    for assignment in result.assignments:
        assignments_by_slot[assignment.slot_id].append(assignment)

    worksheet = worksheets["Assignments"]
    assignment_headers = (
        "Scenario",
        "Slot ID",
        "Start",
        "End",
        "Interviewer ID",
        "Interviewer Name",
        "Group",
        "Locked",
        "Preference",
    )
    _write_table(
        workbook,
        worksheet,
        headers=assignment_headers,
        rows=(
            (
                item.scenario,
                item.slot_id,
                item.start,
                item.end,
                item.interviewer_id,
                item.interviewer_name,
                item.group.label,
                item.locked,
                item.preference,
            )
            for item in result.assignments
        ),
        table_name="AssignmentsTable",
        formats=formats,
        widths={"Start": 22, "End": 22, "Interviewer Name": 24, "Group": 22},
    )

    def schedule_rows(group: InterviewerGroup | None = None):
        for slot_summary in result.slot_summaries:
            slot_assignments = [
                item
                for item in assignments_by_slot.get(slot_summary.slot_id, [])
                if group is None or item.group is group
            ]
            yield (
                slot_summary.slot_id,
                slot_summary.start,
                slot_summary.end,
                slot_summary.target,
                (
                    slot_summary.student_target
                    if group is InterviewerGroup.STUDENT
                    else slot_summary.adcom_target
                    if group is InterviewerGroup.ADCOM
                    else None
                ),
                slot_summary.capacity,
                len(slot_assignments),
                ", ".join(item.interviewer_name for item in slot_assignments),
            )

    schedule_headers = (
        "Slot ID",
        "Start",
        "End",
        "Total Target",
        "Group Target",
        "Shared Capacity",
        "Assigned",
        "Interviewers",
    )
    for sheet_name, group, table_name in (
        ("Schedule_By_Slot", None, "ScheduleBySlotTable"),
        (
            "Student_only Schedule_by_slot",
            InterviewerGroup.STUDENT,
            "StudentScheduleTable",
        ),
        (
            "Adcom_only Schedule_by_slot",
            InterviewerGroup.ADCOM,
            "AdcomScheduleTable",
        ),
    ):
        worksheet = worksheets[sheet_name]
        _write_table(
            workbook,
            worksheet,
            headers=schedule_headers,
            rows=schedule_rows(group),
            table_name=table_name,
            formats=formats,
            widths={"Start": 22, "End": 22, "Interviewers": 55},
        )

    worksheet = worksheets["Interviewer_Summary"]
    interviewer_headers = (
        "Interviewer ID",
        "Interviewer Name",
        "Group",
        "Historical Count",
        "New Assignments",
        "Cumulative Total",
        "Minimum",
        "Target",
        "Maximum",
        "Minimum Met",
        "Minimum Shortfall",
        "Target Shortfall",
        "Maximum Overage",
        "Active Days",
        "Maximum Assigned on Day",
        "Back-to-Back Pairs",
    )
    _write_table(
        workbook,
        worksheet,
        headers=interviewer_headers,
        rows=(
            (
                item.interviewer_id,
                item.interviewer_name,
                item.group.label,
                item.historical_prior_count,
                item.new_assignments,
                item.cumulative_total,
                item.minimum,
                item.target,
                item.maximum,
                item.minimum_shortfall == 0,
                item.minimum_shortfall,
                item.target_shortfall,
                item.maximum_overage,
                item.active_days,
                item.maximum_assigned_on_day,
                item.back_to_back_pairs,
            )
            for item in result.interviewer_summaries
        ),
        table_name="InterviewerSummaryTable",
        formats=formats,
        widths={"Interviewer Name": 24, "Group": 22},
    )
    if result.interviewer_summaries:
        minimum_shortfall_col = interviewer_headers.index("Minimum Shortfall")
        maximum_overage_col = interviewer_headers.index("Maximum Overage")
        last_row = len(result.interviewer_summaries)
        worksheet.conditional_format(
            1,
            minimum_shortfall_col,
            last_row,
            minimum_shortfall_col,
            {"type": "cell", "criteria": ">", "value": 0, "format": formats["warning"]},
        )
        worksheet.conditional_format(
            1,
            maximum_overage_col,
            last_row,
            maximum_overage_col,
            {"type": "cell", "criteria": ">", "value": 0, "format": formats["error"]},
        )

    worksheet = worksheets["Group_Summary"]
    group_headers = (
        "Group",
        "Interviewers",
        "Historical Count",
        "New Assignments",
        "Cumulative Total",
        "Minimum Required",
        "Target Total",
        "Maximum Allowed",
        "Below Minimum",
        "Minimum Shortfall",
        "Above Maximum",
        "Maximum Overage",
        "Back-to-Back Pairs",
    )
    group_rows = []
    for group in InterviewerGroup:
        summaries = [item for item in result.interviewer_summaries if item.group is group]
        group_rows.append(
            (
                group.label,
                len(summaries),
                sum(item.historical_prior_count for item in summaries),
                sum(item.new_assignments for item in summaries),
                sum(item.cumulative_total for item in summaries),
                sum(item.minimum for item in summaries),
                sum(item.target for item in summaries),
                sum(item.maximum for item in summaries),
                sum(item.minimum_shortfall > 0 for item in summaries),
                sum(item.minimum_shortfall for item in summaries),
                sum(item.maximum_overage > 0 for item in summaries),
                sum(item.maximum_overage for item in summaries),
                sum(item.back_to_back_pairs for item in summaries),
            )
        )
    _write_table(
        workbook,
        worksheet,
        headers=group_headers,
        rows=group_rows,
        table_name="GroupSummaryTable",
        formats=formats,
        widths={"Group": 22},
    )

    worksheet = worksheets["Slot_Summary"]
    slot_headers = (
        "Slot ID",
        "Start",
        "End",
        "Target",
        "Shared Capacity",
        "Assigned",
        "Target Deficit",
        "Remaining Capacity",
        "Student Target",
        "Student Assigned",
        "Adcom Target",
        "Adcom Assigned",
    )
    _write_table(
        workbook,
        worksheet,
        headers=slot_headers,
        rows=(
            (
                item.slot_id,
                item.start,
                item.end,
                item.target,
                item.capacity,
                item.assigned,
                item.target_deficit,
                item.remaining_capacity,
                item.student_target,
                item.student_assigned,
                item.adcom_target,
                item.adcom_assigned,
            )
            for item in result.slot_summaries
        ),
        table_name="SlotSummaryTable",
        formats=formats,
        widths={"Start": 22, "End": 22},
    )

    worksheet = worksheets["Constraint_Diagnostics"]
    diagnostic_headers = (
        "Severity",
        "Code",
        "Constraint",
        "Interviewer ID",
        "Interviewer Name",
        "Group",
        "Slot ID",
        "Date",
        "Expected",
        "Actual",
        "Message",
        "Source",
    )
    diagnostic_rows = [
        (
            item.severity,
            item.code,
            item.constraint,
            item.interviewer_id,
            item.interviewer_name,
            _group_label(item.group),
            item.slot_id,
            item.assignment_date.isoformat() if item.assignment_date else None,
            item.expected,
            item.actual,
            item.message,
            None,
        )
        for item in result.diagnostics
    ]
    diagnostic_rows.extend(
        (
            notice.severity,
            notice.code,
            "source_import",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            notice.message,
            notice.source,
        )
        for notice in import_notices
    )
    _write_table(
        workbook,
        worksheet,
        headers=diagnostic_headers,
        rows=diagnostic_rows,
        table_name="ConstraintDiagnosticsTable",
        formats=formats,
        widths={"Message": 70, "Source": 28, "Interviewer Name": 24},
    )

    worksheet = worksheets["Run_Settings"]
    run_rows: list[tuple[Any, Any]] = [
        ("Scenario", result.scenario),
        ("Status", result.status.value),
        ("Generated At ET", datetime.now(ZoneInfo("America/New_York")).isoformat()),
        ("Solver Wall Time Seconds", round(result.wall_time_seconds, 3)),
        ("Interviewer Count", len(problem.interviewers)),
        ("Slot Count", len(problem.slots)),
        ("Assignment Count", len(result.assignments)),
        ("Timezone", str(problem.slots[0].start.tzinfo) if problem.slots else ""),
        ("Preference Scale", "1 (available/default) through 5 (most preferred)"),
    ]
    for group in InterviewerGroup:
        policy = config.group_policies[group]
        prefix = group.label
        run_rows.extend(
            (
                (f"{prefix} Minimum", policy.min_total),
                (f"{prefix} Target", policy.target_total),
                (f"{prefix} Maximum", policy.max_total),
                (f"{prefix} Maximum Per Day", policy.max_per_day),
                (f"{prefix} Minimum Per Active Day", policy.min_per_active_day),
            )
        )
    run_rows.extend((f"Setting: {key}", value) for key, value in result.settings.items())
    run_rows.extend(
        (f"Objective: {key}", value)
        for key, value in result.objective_metrics.items()
    )
    _write_table(
        workbook,
        worksheet,
        headers=("Setting", "Value"),
        rows=run_rows,
        table_name="RunSettingsTable",
        formats=formats,
        widths={"Setting": 42, "Value": 45},
    )

    workbook.close()
    return output.getvalue()
