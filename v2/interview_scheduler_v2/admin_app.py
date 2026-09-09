"""Plain-language Streamlit workflow for administrative scheduling users."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from .config import BackToBackPolicy, GroupPolicy, RelaxationMode, SchedulerConfig
from .domain import (
    Interviewer,
    InterviewerGroup,
    SchedulingProblem,
    Slot,
)
from .io import (
    CampaignImportResult,
    PeriodTemplateError,
    build_interview_period_template,
    parse_completed_interview_period_template,
    period_template_filename,
    prepare_campaign_from_availability,
)
from .optimization import SolveResult, SolveStatus, solve
from .presentation import (
    AdminMessage,
    AdminMessageLevel,
    BRAND_CSS,
    SCHEDULE_AVAILABILITY_SECTION,
    SCHEDULE_COUNTS_SECTION,
    SCHEDULE_DOWNLOADS_SECTION,
    SCHEDULE_RESULTS_SECTION,
    SCHEDULE_RULES_SECTION,
    build_schedule_journey,
    format_interview_period,
    present_diagnostics,
    present_import_notices,
    present_validation_issues,
    schedule_journey_html,
)
from .reporting import (
    build_simplified_schedule_workbook,
    build_workbook,
    scenario_filename,
    simplified_schedule_filename,
)
from .validation import ValidationReport, validate_problem


GROUP_BY_LABEL = {group.label: group for group in InterviewerGroup}
EXCEPTION_OPTIONS = {
    "Allow assignments below required minimums": RelaxationMode.MINIMUMS,
    "Allow assignments above maximum limits": RelaxationMode.MAXIMUMS,
    "Allow both types of exception": RelaxationMode.MINIMUMS_AND_MAXIMUMS,
}
STUDENT_PRIORITY_OPTIONS = {
    "Students first when openings are limited (recommended)": 3,
    "Treat Student and Adcom goals equally": 1,
    "Strongly prioritize Student goals": 6,
}
CONSECUTIVE_OPTIONS = {
    "Avoid consecutive interviews when possible (recommended)": BackToBackPolicy.DISCOURAGED,
    "Never schedule consecutive interviews": BackToBackPolicy.HARD,
    "Consecutive interviews are acceptable": BackToBackPolicy.OFF,
}


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
    return int(value)


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


def _clear_schedule_outputs() -> None:
    for key in (
        "v2_problem",
        "v2_config",
        "v2_validation",
        "v2_result",
        "v2_report",
        "v2_simplified_report",
        "v2_download_reviewed",
        "v2_exception_confirmed",
    ):
        st.session_state.pop(key, None)


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


def _invalidate_result_after_edit(*, changed: bool, reason: str) -> bool:
    """Clear stale scheduling output and remember why it was cleared."""

    if not changed:
        return False
    had_result = st.session_state.get("v2_result") is not None
    _clear_schedule_outputs()
    if had_result:
        st.session_state["v2_stale_result_notice"] = reason
        st.toast("Previous scheduling result cleared.")
    return had_result


def _initialize_review(imported: CampaignImportResult) -> None:
    st.session_state["v2_people"] = _people_frame(imported)
    st.session_state["v2_slots"] = _slot_frame(imported)
    st.session_state["v2_slot_editor_revision"] = 0
    st.session_state["v2_period_upload_revision"] = 0
    _clear_schedule_outputs()
    st.session_state.pop("v2_stale_result_notice", None)


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
            capacity_number = float(capacity_value)
        except (TypeError, ValueError):
            invalid_count += 1
            continue
        if capacity_number < 0 or not capacity_number.is_integer():
            invalid_count += 1
            continue
        capacity = int(capacity_number)
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


def _current_schedule_journey():
    """Return progress derived from completed workflow milestones."""

    imported: CampaignImportResult | None = st.session_state.get("v2_import")
    availability_ready = imported is not None
    slots = st.session_state.get("v2_slots")
    counts_ready = (
        availability_ready
        and isinstance(slots, pd.DataFrame)
        and not _period_setup_issues(slots)
    )
    import_has_errors = bool(
        imported
        and any(notice.severity == "error" for notice in imported.notices)
    )
    ready_to_run = bool(counts_ready and not import_has_errors)
    result: SolveResult | None = st.session_state.get("v2_result")
    schedule_ready = bool(result and result.succeeded)
    downloads_ready = False
    if result and result.succeeded:
        relaxation_mode = RelaxationMode(
            result.settings.get("relaxation_mode", RelaxationMode.STRICT.value)
        )
        downloads_ready = (
            relaxation_mode is RelaxationMode.STRICT
            or bool(st.session_state.get("v2_download_reviewed"))
        )
    return build_schedule_journey(
        availability_ready=availability_ready,
        counts_ready=bool(counts_ready),
        ready_to_run=ready_to_run,
        run_attempted=result is not None,
        schedule_ready=schedule_ready,
        downloads_ready=downloads_ready,
    )


def _show_brand_and_progress() -> None:
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    st.markdown(
        schedule_journey_html(_current_schedule_journey()),
        unsafe_allow_html=True,
    )


def _show_section_anchor(anchor_id: str) -> None:
    """Add a stable in-page destination without changing the visible layout."""

    st.markdown(
        f'<span id="{anchor_id}" class="schedule-section-anchor"></span>',
        unsafe_allow_html=True,
    )


def _period_setup_totals(frame: pd.DataFrame) -> tuple[int, int, int]:
    configured = 0
    offered = 0
    interviews_possible = 0
    for _, row in frame.iterrows():
        if not _present(row.get("Capacity")):
            continue
        try:
            capacity = int(row.get("Capacity"))
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


def _policy_controls(label: str, defaults: GroupPolicy, key: str) -> GroupPolicy:
    st.markdown(f"**{label}**")
    first, second, third, fourth = st.columns(4)
    minimum = first.number_input(
        "Required minimum",
        min_value=0,
        value=defaults.min_total,
        step=1,
        key=f"{key}_min",
        help=(
            "The fewest total interviews each person should have, including "
            "interviews already assigned."
        ),
    )
    target = second.number_input(
        "Preferred total",
        min_value=0,
        value=defaults.target_total,
        step=1,
        key=f"{key}_target",
        help="The total the scheduler should aim for when enough openings are available.",
    )
    maximum = third.number_input(
        "Maximum total",
        min_value=0,
        value=defaults.max_total,
        step=1,
        key=f"{key}_max",
        help=(
            "The most total interviews a person should have, including interviews "
            "already assigned."
        ),
    )
    maximum_per_day = fourth.number_input(
        "Maximum per day",
        min_value=0,
        value=defaults.max_per_day,
        step=1,
        key=f"{key}_day_max",
    )
    minimum_active_day = st.number_input(
        "Minimum on a day when assigned",
        min_value=0,
        value=defaults.min_per_active_day,
        step=1,
        key=f"{key}_day_min",
        help=(
            "Leave this at zero unless anyone scheduled that day must receive more "
            "than one interview."
        ),
    )
    return GroupPolicy(
        int(minimum),
        int(target),
        int(maximum),
        int(maximum_per_day),
        int(minimum_active_day),
    )


def _render_admin_messages(
    messages: tuple[AdminMessage, ...] | list[AdminMessage],
    *,
    details_label: str = "View details",
    show_technical_details: bool = True,
) -> None:
    for message in messages:
        body = f"**{message.title}**\n\n{message.summary}"
        if message.action:
            body += f"\n\n**Next step:** {message.action}"
        if message.level == AdminMessageLevel.BLOCKING:
            st.error(body)
        elif message.level == AdminMessageLevel.REVIEW:
            st.warning(body)
        elif message.level == AdminMessageLevel.COMPLETE:
            st.success(body)
        else:
            st.info(body)

    detail_messages = [message for message in messages if message.details]
    if show_technical_details and detail_messages:
        with st.expander(details_label, expanded=False):
            for message in detail_messages:
                st.markdown(f"**{message.title}**")
                for detail in message.details:
                    st.markdown(f"- {detail}")
                if message.technical_codes:
                    st.caption("Reference: " + ", ".join(message.technical_codes))


def _show_file_checks(imported: CampaignImportResult) -> None:
    st.markdown("### File check results")
    if not imported.notices:
        st.success(
            "**Files checked successfully**\n\n"
            "The interview periods were found and all availability records matched."
        )
        return
    _render_admin_messages(
        present_import_notices(imported.notices),
        details_label="View affected source rows and technical references",
    )


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


def _run_schedule(
    problem: SchedulingProblem,
    config: SchedulerConfig,
    *,
    relaxation_mode: RelaxationMode,
) -> None:
    report = validate_problem(problem, config)
    result = solve(
        problem,
        scenario=st.session_state.get("v2_scenario", "Schedule"),
        config=config,
        relaxation_mode=relaxation_mode,
    )
    st.session_state["v2_problem"] = problem
    st.session_state["v2_config"] = config
    st.session_state["v2_validation"] = report
    st.session_state["v2_result"] = result
    st.session_state.pop("v2_report", None)
    st.session_state.pop("v2_simplified_report", None)
    st.session_state.pop("v2_download_reviewed", None)
    st.session_state.pop("v2_stale_result_notice", None)


def _show_upload_step() -> CampaignImportResult | None:
    _show_section_anchor(SCHEDULE_AVAILABILITY_SECTION)
    imported = st.session_state.get("v2_import")
    label = (
        "✓ 1 · Availability checked"
        if imported is not None
        else "1 · Upload availability files"
    )
    with st.expander(label, expanded=imported is None):
        st.write(
            "Upload one availability file for each interviewer group. The tool will "
            "build the interview date-and-time list from the Student file."
        )
        left, right = st.columns(2)
        student_file = left.file_uploader(
            "Student interviewer availability",
            type=["xlsx"],
            key="student_file",
            help=(
                "The date headings and time worksheets in this file determine the "
                "interview periods used in the next step."
            ),
        )
        adcom_file = right.file_uploader(
            "Adcom interviewer availability",
            type=["xlsx"],
            key="adcom_file",
        )
        settings_left, settings_middle, settings_right = st.columns(3)
        scenario = settings_left.text_input(
            "Schedule name",
            value=st.session_state.get("v2_scenario", "Winter 2026 Round 2"),
            help="This name will be included in the downloaded workbook filename.",
        )
        campaign_year = settings_middle.number_input(
            "Interview year",
            min_value=2025,
            max_value=2100,
            value=2026,
            step=1,
        )
        settings_right.text_input(
            "Time zone",
            value="Eastern Time",
            disabled=True,
            help="All dates and times are interpreted in Eastern Time.",
        )
        if st.button(
            "Find interview periods and continue",
            type="primary",
            disabled=not (student_file and adcom_file),
        ):
            try:
                imported = prepare_campaign_from_availability(
                    student_workbook=student_file.getvalue(),
                    adcom_workbook=adcom_file.getvalue(),
                    year=int(campaign_year),
                    timezone_name="America/New_York",
                )
            except Exception as exc:  # Streamlit needs a friendly import boundary.
                st.error(
                    "**The files could not be checked**\n\n"
                    "Confirm that each file is the correct Excel workbook and uses "
                    "the expected availability layout."
                )
                with st.expander("View technical error details"):
                    st.code(str(exc))
            else:
                st.session_state["v2_import"] = imported
                st.session_state["v2_scenario"] = scenario.strip() or "Schedule"
                st.session_state["v2_campaign_year"] = int(campaign_year)
                st.session_state["v2_timezone"] = "America/New_York"
                _initialize_review(imported)
                st.rerun()
    return st.session_state.get("v2_import")


def _show_review_step(
    imported: CampaignImportResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _show_section_anchor(SCHEDULE_COUNTS_SECTION)
    with st.expander("2 · Review people and set interview counts", expanded=True):
        st.markdown("#### Interviewers")
        st.write(
            "Confirm who should be included, each person's group, availability count, "
            "and any interviews they have already completed or been assigned."
        )
        full_people = st.session_state["v2_people"]
        basic_people = full_people[
            [
                "Enabled",
                "Interviewer ID",
                "Interviewer Name",
                "Group",
                "Availability Slots",
                "Historical Count",
            ]
        ].copy()
        people_review = st.data_editor(
            basic_people,
            key="people_review_editor",
            hide_index=True,
            num_rows="dynamic",
            use_container_width=True,
            column_order=(
                "Enabled",
                "Interviewer Name",
                "Group",
                "Availability Slots",
                "Historical Count",
            ),
            disabled=["Interviewer ID", "Availability Slots"],
            column_config={
                "Enabled": st.column_config.CheckboxColumn(
                    "Include", default=True, help="Clear this box to leave someone out."
                ),
                "Interviewer ID": None,
                "Interviewer Name": st.column_config.TextColumn(
                    "Name", required=True
                ),
                "Group": st.column_config.SelectboxColumn(
                    "Interviewer group",
                    options=list(GROUP_BY_LABEL),
                    required=True,
                ),
                "Availability Slots": st.column_config.NumberColumn(
                    "Available periods",
                    help="Counted from the uploaded availability file.",
                ),
                "Historical Count": st.column_config.NumberColumn(
                    "Interviews already assigned",
                    min_value=0,
                    step=1,
                    help="These count toward the person's total assignment limits.",
                ),
            },
        )
        reviewed_people = _merge_people_review(full_people, people_review)
        _invalidate_result_after_edit(
            changed=_frames_differ(full_people, reviewed_people),
            reason=(
                "The interviewer list, group assignments, inclusion selections, or "
                "prior interview counts changed."
            ),
        )
        full_people = reviewed_people
        st.session_state["v2_people"] = reviewed_people

        if st.checkbox(
            "Edit assignment limits for individual interviewers",
            value=False,
            help="Use this only when one person's limits differ from their group's rules.",
        ):
            st.caption(
                "Clear ‘Use group rules’ for a person before entering their individual limits."
            )
            previous_people_limits = full_people
            full_people = st.data_editor(
                full_people,
                key="people_limits_editor",
                hide_index=True,
                num_rows="dynamic",
                use_container_width=True,
                column_order=(
                    "Interviewer Name",
                    "Group",
                    "Use Group Defaults",
                    "Minimum",
                    "Target",
                    "Maximum",
                    "Maximum Per Day",
                    "Minimum Per Active Day",
                ),
                disabled=["Interviewer ID", "Availability Slots"],
                column_config={
                    "Interviewer ID": None,
                    "Interviewer Name": st.column_config.TextColumn("Name"),
                    "Group": st.column_config.SelectboxColumn(
                        "Group", options=list(GROUP_BY_LABEL), required=True
                    ),
                    "Use Group Defaults": st.column_config.CheckboxColumn(
                        "Use group rules", default=True
                    ),
                    "Minimum": st.column_config.NumberColumn(
                        "Required minimum", min_value=0, step=1
                    ),
                    "Target": st.column_config.NumberColumn(
                        "Preferred total", min_value=0, step=1
                    ),
                    "Maximum": st.column_config.NumberColumn(
                        "Maximum total", min_value=0, step=1
                    ),
                    "Maximum Per Day": st.column_config.NumberColumn(
                        "Maximum per day", min_value=0, step=1
                    ),
                    "Minimum Per Active Day": st.column_config.NumberColumn(
                        "Minimum when assigned that day", min_value=0, step=1
                    ),
                },
            )
            _invalidate_result_after_edit(
                changed=_frames_differ(previous_people_limits, full_people),
                reason="One or more individual interviewer assignment limits changed.",
            )
            st.session_state["v2_people"] = full_people

        st.divider()
        st.markdown("#### Interview periods and counts")
        st.write(
            "These dates and times came from the Student availability file. Enter how "
            "many interviews can run in each period. One interviewer fills one seat."
        )

        enter_tab, excel_tab = st.tabs(["Enter counts here", "Use an Excel worksheet"])
        with enter_tab:
            st.caption(
                "Enter the number of interviews that can run in every period. "
                "Use 0 when a period will not be offered."
            )
            bulk_left, bulk_right = st.columns([1, 2])
            bulk_count = bulk_left.number_input(
                "Use the same count for every period",
                min_value=0,
                value=1,
                step=1,
                key="v2_bulk_capacity",
            )
            if bulk_right.button(
                "Apply this count to every period",
                key="v2_apply_bulk_counts",
                help="You can adjust individual periods in the table afterward.",
            ):
                previous_bulk_slots = st.session_state["v2_slots"]
                updated = previous_bulk_slots.copy()
                updated["Capacity"] = int(bulk_count)
                _invalidate_result_after_edit(
                    changed=_frames_differ(previous_bulk_slots, updated),
                    reason="The number of interviews possible in one or more periods changed.",
                )
                st.session_state["v2_slots"] = updated
                st.session_state["v2_slot_editor_revision"] += 1
                st.rerun()

            previous_slots = st.session_state["v2_slots"]
            slots = st.data_editor(
                previous_slots,
                key=(
                    "slot_review_editor_"
                    f"{st.session_state.get('v2_slot_editor_revision', 0)}"
                ),
                hide_index=True,
                use_container_width=True,
                column_order=(
                    "Interview Period",
                    "Student Available",
                    "Adcom Available",
                    "Capacity",
                ),
                disabled=[
                    "Slot ID",
                    "Interview Period",
                    "Start",
                    "End",
                    "Student Available",
                    "Adcom Available",
                ],
                column_config={
                    "Slot ID": None,
                    "Start": None,
                    "End": None,
                    "Interview Period": st.column_config.TextColumn(
                        "Interview period"
                    ),
                    "Student Available": st.column_config.NumberColumn(
                        "Student available",
                        help="Student interviewers who marked this period as available.",
                    ),
                    "Adcom Available": st.column_config.NumberColumn(
                        "Adcom available",
                        help="Adcom interviewers whose availability matched this period.",
                    ),
                    "Capacity": st.column_config.NumberColumn(
                        "Interviews possible",
                        min_value=0,
                        step=1,
                        required=True,
                        help="Use 0 when this period will not be offered.",
                    ),
                },
            )
            _invalidate_result_after_edit(
                changed=_frames_differ(previous_slots, slots),
                reason="The number of interviews possible in one or more periods changed.",
            )
            st.session_state["v2_slots"] = slots

        with excel_tab:
            st.write(
                "Download the worksheet, fill in the yellow count cells, save it, "
                "and upload the completed copy here."
            )
            student_counts, adcom_counts = _availability_counts(imported)
            template = build_interview_period_template(
                imported.problem.slots,
                scenario=st.session_state.get("v2_scenario", "Schedule"),
                student_available=student_counts,
                adcom_available=adcom_counts,
                timezone_name=st.session_state.get(
                    "v2_timezone", "America/New_York"
                ),
            )
            st.download_button(
                "Download interview-period worksheet",
                data=template,
                file_name=period_template_filename(
                    st.session_state.get("v2_scenario", "Schedule")
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                key="v2_download_period_template",
            )
            completed_file = st.file_uploader(
                "Completed interview-period worksheet",
                type=["xlsx"],
                key=(
                    "v2_completed_period_file_"
                    f"{st.session_state.get('v2_period_upload_revision', 0)}"
                ),
            )
            if st.button(
                "Use counts from this worksheet",
                key="v2_apply_period_template",
                disabled=completed_file is None,
            ):
                try:
                    parsed_periods = parse_completed_interview_period_template(
                        completed_file.getvalue(),
                        expected_slots=imported.problem.slots,
                        year=int(st.session_state.get("v2_campaign_year", 2026)),
                        timezone_name=st.session_state.get(
                            "v2_timezone", "America/New_York"
                        ),
                    )
                except PeriodTemplateError as exc:
                    st.error(f"**This worksheet could not be used**\n\n{exc}")
                except Exception as exc:
                    st.error(
                        "**This worksheet could not be used**\n\n"
                        "Confirm that it is the completed interview-period worksheet "
                        "downloaded from this schedule setup."
                    )
                    with st.expander("View technical error details"):
                        st.code(str(exc))
                else:
                    previous_uploaded_slots = st.session_state["v2_slots"]
                    uploaded_slots = _replace_period_counts(
                        previous_uploaded_slots, parsed_periods.slots
                    )
                    _invalidate_result_after_edit(
                        changed=_frames_differ(
                            previous_uploaded_slots,
                            uploaded_slots,
                        ),
                        reason="The number of interviews possible in one or more periods changed.",
                    )
                    st.session_state["v2_slots"] = uploaded_slots
                    st.session_state["v2_slot_editor_revision"] += 1
                    st.session_state["v2_period_upload_revision"] += 1
                    st.session_state["v2_period_import_success"] = True
                    st.rerun()
            if st.session_state.pop("v2_period_import_success", False):
                st.success("The interview counts were added from the worksheet.")

        configured, offered, interviews_possible = _period_setup_totals(
            st.session_state["v2_slots"]
        )
        period_metrics = st.columns(3)
        period_metrics[0].metric(
            "Periods completed",
            f"{configured} of {len(st.session_state['v2_slots'])}",
        )
        period_metrics[1].metric("Periods offered", offered)
        period_metrics[2].metric("Interviews possible", interviews_possible)
        setup_issues = _period_setup_issues(st.session_state["v2_slots"])
        if setup_issues:
            st.warning("\n\n".join(setup_issues))
        else:
            st.success("Interview-period counts are complete.")

        if st.checkbox(
            "Set preferred Student or Adcom staffing by interview period",
            value=False,
            help=(
                "These are preferences within shared capacity, not separate seats "
                "reserved for each group."
            ),
        ):
            previous_group_slots = st.session_state["v2_slots"]
            slots = st.data_editor(
                previous_group_slots,
                key=(
                    "slot_group_editor_"
                    f"{st.session_state.get('v2_slot_editor_revision', 0)}"
                ),
                hide_index=True,
                use_container_width=True,
                column_order=(
                    "Interview Period",
                    "Capacity",
                    "Student Target",
                    "Adcom Target",
                ),
                disabled=["Slot ID", "Interview Period", "Start", "End", "Capacity"],
                column_config={
                    "Slot ID": None,
                    "Start": None,
                    "End": None,
                    "Interview Period": st.column_config.TextColumn(
                        "Interview period"
                    ),
                    "Capacity": st.column_config.NumberColumn(
                        "Interviews possible", min_value=0, step=1
                    ),
                    "Student Target": st.column_config.NumberColumn(
                        "Preferred Student count", min_value=0, step=1
                    ),
                    "Adcom Target": st.column_config.NumberColumn(
                        "Preferred Adcom count", min_value=0, step=1
                    ),
                },
            )
            _invalidate_result_after_edit(
                changed=_frames_differ(previous_group_slots, slots),
                reason="One or more preferred group staffing counts changed.",
            )
            st.session_state["v2_slots"] = slots

    return st.session_state["v2_people"], st.session_state["v2_slots"]


def _show_rules_step(
    imported: CampaignImportResult,
    people: pd.DataFrame,
    slots: pd.DataFrame,
    *,
    import_has_errors: bool,
    period_setup_issues: list[str],
) -> None:
    _show_section_anchor(SCHEDULE_RULES_SECTION)
    with st.expander("3 · Confirm rules and create schedule", expanded=True):
        stale_reason = st.session_state.get("v2_stale_result_notice")
        if stale_reason:
            st.warning(
                "**Previous result cleared**\n\n"
                f"{stale_reason} Create the schedule again so the results use "
                "the current information."
            )
        defaults = SchedulerConfig().group_policies
        use_recommended = st.checkbox(
            "Use the recommended group assignment rules",
            value=True,
            help="These are the default limits supplied for this scheduling process.",
        )
        if use_recommended:
            left, right = st.columns(2)
            left.info(
                "**Student Interviewers**\n\n"
                "At least 3 · Aim for 4 · No more than 5 total · No more than 2 per day"
            )
            right.info(
                "**Adcom Interviewers**\n\n"
                "At least 4 · Aim for 4 · No more than 6 total · No more than 2 per day"
            )
            student_policy = defaults[InterviewerGroup.STUDENT]
            adcom_policy = defaults[InterviewerGroup.ADCOM]
        else:
            st.write("Enter the group rules that should apply to this schedule.")
            student_policy = _policy_controls(
                "Student Interviewers",
                defaults[InterviewerGroup.STUDENT],
                "student",
            )
            adcom_policy = _policy_controls(
                "Adcom Interviewers",
                defaults[InterviewerGroup.ADCOM],
                "adcom",
            )

        priority_label = st.selectbox(
            "When there are not enough openings",
            options=list(STUDENT_PRIORITY_OPTIONS),
            index=0,
        )
        consecutive_label = st.selectbox(
            "Consecutive interview periods",
            options=list(CONSECUTIVE_OPTIONS),
            index=0,
        )
        with st.expander("Advanced scheduling settings", expanded=False):
            maximum_consecutive = st.number_input(
                "Maximum consecutive interview periods",
                min_value=1,
                value=2,
                step=1,
                help="This limit always applies, including when consecutive periods are discouraged.",
            )
            time_limit = st.number_input(
                "Time allowed to create the schedule (seconds)",
                min_value=5,
                value=30,
                step=5,
                help="Increase this only if the tool cannot finish a large or complex schedule.",
            )

        if import_has_errors:
            st.error(
                "Correct the blocking availability-file issues before creating a schedule."
            )
        if period_setup_issues:
            st.error(
                "Complete the interview counts in Step 2 before creating a schedule."
            )
        if st.button(
            "Create schedule using standard rules",
            type="primary",
            disabled=import_has_errors or bool(period_setup_issues),
        ):
            try:
                problem, config = _reviewed_problem_and_config(
                    imported,
                    people,
                    slots,
                    student_defaults=student_policy,
                    adcom_defaults=adcom_policy,
                    student_priority_weight=STUDENT_PRIORITY_OPTIONS[priority_label],
                    back_to_back=CONSECUTIVE_OPTIONS[consecutive_label],
                    maximum_consecutive=int(maximum_consecutive),
                    time_limit_seconds=float(time_limit),
                )
                _run_schedule(
                    problem,
                    config,
                    relaxation_mode=RelaxationMode.STRICT,
                )
            except Exception as exc:  # Keep malformed edits inside a friendly boundary.
                st.error(
                    "**The schedule could not be created from the reviewed information**\n\n"
                    "Check for blank names or assignment limits entered in the wrong order."
                )
                with st.expander("View technical error details"):
                    st.code(str(exc))
            else:
                st.rerun()


def _status_for_interviewer(summary: Any) -> str:
    if summary.minimum_shortfall > 0:
        return f"Below minimum by {summary.minimum_shortfall}"
    if summary.maximum_overage > 0:
        return f"Above maximum by {summary.maximum_overage}"
    if summary.maximum_assigned_on_day > summary.max_per_day:
        return (
            "Above daily maximum by "
            f"{summary.maximum_assigned_on_day - summary.max_per_day}"
        )
    if summary.target_shortfall > 0:
        return f"Below preferred total by {summary.target_shortfall}"
    return "Preferred total met"


def _show_exception_retry() -> None:
    st.markdown("#### If an exception is acceptable")
    st.write(
        "The standard rules did not produce a schedule. Choose an exception only "
        "after reviewing the issues above. The results will name every affected person."
    )
    selection = st.selectbox(
        "Exception to allow",
        options=list(EXCEPTION_OPTIONS),
        index=0,
    )
    impacts = {
        "Allow assignments below required minimums": (
            "Some people may receive fewer interviews than their required minimum."
        ),
        "Allow assignments above maximum limits": (
            "Some people may exceed their total or daily maximum."
        ),
        "Allow both types of exception": (
            "Some people may fall below minimums or exceed total or daily maximums."
        ),
    }
    st.warning(f"**What this allows:** {impacts[selection]}")
    confirmed = st.checkbox(
        "I understand this exception and will review every affected interviewer",
        key="v2_exception_confirmed",
    )
    if st.button(
        "Try again with this exception",
        type="primary",
        disabled=not confirmed,
    ):
        _run_schedule(
            st.session_state["v2_problem"],
            st.session_state["v2_config"],
            relaxation_mode=EXCEPTION_OPTIONS[selection],
        )
        st.rerun()


def _show_failure(result: SolveResult, report: ValidationReport | None) -> None:
    if result.status.value == SolveStatus.INFEASIBLE.value:
        st.error(
            "**A schedule could not be created using the selected rules**\n\n"
            "No set of assignments can meet every required minimum, maximum, daily "
            "limit, availability, and interview-period capacity at the same time."
        )
    elif result.status.value == SolveStatus.INVALID.value:
        st.error(
            "**Some schedule information needs to be corrected**\n\n"
            "Review the items below, correct them in Steps 2 or 3, and try again."
        )
    else:
        st.error(
            "**The scheduler did not finish creating a schedule**\n\n"
            "Try again. If this continues, increase the time allowed under Advanced "
            "scheduling settings."
        )

    messages = present_diagnostics(result.diagnostics)
    if not messages and report is not None:
        messages = present_validation_issues(report.issues)
    if messages:
        _render_admin_messages(
            messages,
            details_label="View affected records and technical references",
        )
    if result.status.value == SolveStatus.INFEASIBLE.value:
        _show_exception_retry()


def _show_success(
    result: SolveResult,
    imported: CampaignImportResult,
) -> None:
    relaxation_mode = RelaxationMode(
        result.settings.get("relaxation_mode", RelaxationMode.STRICT.value)
    )
    messages = present_diagnostics(result.diagnostics)
    review_messages = [
        message
        for message in messages
        if message.level in {AdminMessageLevel.BLOCKING, AdminMessageLevel.REVIEW}
    ]
    adjustment_messages = [
        message
        for message in messages
        if message.level == AdminMessageLevel.ADJUSTMENT
    ]
    if relaxation_mode != RelaxationMode.STRICT:
        st.warning(
            "**Schedule created with exceptions — review required**\n\n"
            "The schedule uses the exception selected after the standard attempt failed."
        )
    elif review_messages:
        st.warning(
            "**Schedule created — review recommended**\n\n"
            "Review the items below before distributing the schedule."
        )
    else:
        st.success(
            "**Schedule created — ready to download**\n\n"
            "The schedule follows the selected rules and all interview periods are staffed."
        )

    fully_staffed = sum(item.target_deficit == 0 for item in result.slot_summaries)
    unfilled = sum(item.target_deficit for item in result.slot_summaries)
    metrics = st.columns(4)
    metrics[0].metric("Interview assignments", len(result.assignments))
    metrics[1].metric(
        "Fully staffed periods", f"{fully_staffed} of {len(result.slot_summaries)}"
    )
    metrics[2].metric("Unfilled interview seats", unfilled)
    metrics[3].metric(
        "People below required minimum", len(result.minimum_shortfalls)
    )

    if review_messages:
        st.markdown("#### Items to review")
        _render_admin_messages(
            review_messages,
            details_label="View affected records and technical references",
        )
    exception_people = [
        item
        for item in result.interviewer_summaries
        if item.minimum_shortfall > 0
        or item.maximum_overage > 0
        or item.maximum_assigned_on_day > item.max_per_day
    ]
    if exception_people:
        st.markdown("##### Interviewers affected by exceptions")
        st.write(
            "These are the specific people whose required minimum, total maximum, "
            "or daily maximum was not met."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Interviewer": item.interviewer_name,
                        "Group": item.group.label,
                        "Total after scheduling": item.cumulative_total,
                        "Required minimum": item.minimum,
                        "Maximum total": item.maximum,
                        "Most in one day": item.maximum_assigned_on_day,
                        "Maximum per day": item.max_per_day,
                        "What to review": _status_for_interviewer(item),
                    }
                    for item in exception_people
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    if adjustment_messages:
        with st.expander("Scheduling notes", expanded=False):
            _render_admin_messages(
                adjustment_messages,
                show_technical_details=False,
            )

    assignment_rows = [
        {
            "Interview period": format_interview_period(item.start, item.end),
            "Interviewer": item.interviewer_name,
            "Group": item.group.label,
            "Preference": item.preference,
        }
        for item in result.assignments
    ]
    interviewer_rows = [
        {
            "Interviewer": item.interviewer_name,
            "Group": item.group.label,
            "Already assigned": item.historical_prior_count,
            "Scheduled now": item.new_assignments,
            "Total after scheduling": item.cumulative_total,
            "Required minimum": item.minimum,
            "Preferred total": item.target,
            "Maximum total": item.maximum,
            "Status": _status_for_interviewer(item),
        }
        for item in result.interviewer_summaries
    ]
    slot_rows = [
        {
            "Interview period": format_interview_period(item.start, item.end),
            "Interviews needed": item.target,
            "Scheduled": item.assigned,
            "Unfilled": item.target_deficit,
            "Student": item.student_assigned,
            "Adcom": item.adcom_assigned,
            "Status": "Fully staffed" if item.target_deficit == 0 else "Needs review",
        }
        for item in result.slot_summaries
    ]
    schedule_tab, people_tab, coverage_tab = st.tabs(
        ["Schedule", "Interviewer totals", "Interview period coverage"]
    )
    with schedule_tab:
        st.dataframe(
            pd.DataFrame(assignment_rows),
            hide_index=True,
            use_container_width=True,
        )
    with people_tab:
        st.dataframe(
            pd.DataFrame(interviewer_rows),
            hide_index=True,
            use_container_width=True,
        )
    with coverage_tab:
        st.dataframe(
            pd.DataFrame(slot_rows),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Technical scheduling details", expanded=False):
        st.caption(
            f"Result: {result.status.value} · Processing time: "
            f"{result.wall_time_seconds:.2f} seconds · Exception mode: "
            f"{relaxation_mode.value}"
        )
        if result.diagnostics:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Severity": item.severity,
                            "Reference": item.code,
                            "Interviewer ID": item.interviewer_id,
                            "Interviewer": item.interviewer_name,
                            "Slot ID": item.slot_id,
                            "Expected": item.expected,
                            "Actual": item.actual,
                            "Technical message": item.message,
                        }
                        for item in result.diagnostics
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

    if "v2_report" not in st.session_state:
        st.session_state["v2_report"] = build_workbook(
            result,
            st.session_state["v2_problem"],
            st.session_state["v2_config"],
            import_notices=imported.notices,
        )
    if "v2_simplified_report" not in st.session_state:
        st.session_state["v2_simplified_report"] = (
            build_simplified_schedule_workbook(result)
        )
    _show_section_anchor(SCHEDULE_DOWNLOADS_SECTION)
    download_disabled = False
    if relaxation_mode != RelaxationMode.STRICT:
        download_disabled = not st.checkbox(
            "I reviewed the exception results and the affected interviewers",
            key="v2_download_reviewed",
        )
    generated_at = datetime.now(ZoneInfo("America/New_York"))
    st.markdown("#### Download schedule files")
    st.caption(
        "The full workbook includes totals and scheduling details. The simplified "
        "workbook lists one interview period per row, with one interviewer in each "
        "Group column. A blank Group cell is an unfilled interview; a gray cell is "
        "outside that period's capacity."
    )
    full_download, simplified_download = st.columns(2)
    full_download.download_button(
        "Download full schedule workbook",
        data=st.session_state["v2_report"],
        file_name=scenario_filename(result.scenario, generated_at=generated_at),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        disabled=download_disabled,
    )
    simplified_download.download_button(
        "Download simplified schedule",
        data=st.session_state["v2_simplified_report"],
        file_name=simplified_schedule_filename(
            result.scenario,
            generated_at=generated_at,
        ),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=download_disabled,
    )


def _show_results(imported: CampaignImportResult) -> None:
    result: SolveResult | None = st.session_state.get("v2_result")
    if result is None:
        return
    _show_section_anchor(SCHEDULE_RESULTS_SECTION)
    st.markdown("### 4 · Review and download")
    report: ValidationReport | None = st.session_state.get("v2_validation")
    if result.succeeded:
        _show_success(result, imported)
    else:
        _show_failure(result, report)


def main() -> None:
    st.set_page_config(
        page_title="Interview Scheduler",
        page_icon="📅",
        layout="wide",
    )
    _show_brand_and_progress()
    st.title("Interview Scheduler")
    st.caption(
        "Create one-person interview assignments for Student and Adcom interviewers "
        "using availability, staffing needs, and assignment limits."
    )

    imported = _show_upload_step()
    if imported is None:
        st.info("Upload both availability files in Step 1 to begin.")
        return

    _show_file_checks(imported)
    import_has_errors = any(
        notice.severity == "error" for notice in imported.notices
    )
    counts = st.columns(4)
    counts[0].metric(
        "Student interviewers",
        sum(
            person.group.value == InterviewerGroup.STUDENT.value
            for person in imported.problem.interviewers
        ),
    )
    counts[1].metric(
        "Adcom interviewers",
        sum(
            person.group.value == InterviewerGroup.ADCOM.value
            for person in imported.problem.interviewers
        ),
    )
    counts[2].metric("Interview periods", len(imported.problem.slots))
    counts[3].metric(
        "Interview dates", len({slot.local_date for slot in imported.problem.slots})
    )

    people, slots = _show_review_step(imported)
    period_issues = _period_setup_issues(slots)
    _show_rules_step(
        imported,
        people,
        slots,
        import_has_errors=import_has_errors,
        period_setup_issues=period_issues,
    )
    _show_results(imported)


__all__ = ["main"]
