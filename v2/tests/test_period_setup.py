from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from interview_scheduler_v2 import admin_app
from interview_scheduler_v2.admin_app import (
    _frames_differ,
    _invalidate_result_after_edit,
    _people_frame,
    _period_setup_issues,
    _reviewed_problem_and_config,
    _slot_frame,
)
from interview_scheduler_v2.config import BackToBackPolicy, SchedulerConfig
from interview_scheduler_v2.domain import (
    Interviewer,
    InterviewerGroup,
    SchedulingProblem,
    Slot,
)
from interview_scheduler_v2.io import (
    CampaignImportResult,
    ParsedAvailability,
    ParsedSlot,
    ParsedSlotSet,
)


def prepared_campaign() -> CampaignImportResult:
    zone = ZoneInfo("America/New_York")
    first_start = datetime(2026, 2, 24, 8, tzinfo=zone)
    second_start = datetime(2026, 2, 24, 10, 30, tzinfo=zone)
    slots = (
        Slot("20260224-0800", first_start, first_start + timedelta(minutes=90), 0, 0),
        Slot("20260224-1030", second_start, second_start + timedelta(minutes=90), 0, 0),
    )
    person = Interviewer.create(
        name="Student One",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=[slot.id for slot in slots],
    )
    parsed_slots = ParsedSlotSet(
        [
            ParsedSlot(
                slot.id,
                slot.start,
                slot.end,
                0,
                0,
                slot.id,
            )
            for slot in slots
        ]
    )
    availability = ParsedAvailability("student", [], [person.name])
    return CampaignImportResult(
        problem=SchedulingProblem((person,), slots),
        notices=(),
        interviewer_ids={},
        student_availability=availability,
        adcom_availability=ParsedAvailability("adcom", [], []),
        slot_set=parsed_slots,
        periods_need_configuration=True,
    )


def test_period_setup_requires_every_capacity_and_has_no_separate_slot_target():
    frame = _slot_frame(prepared_campaign())
    assert "Target" not in frame.columns
    assert len(_period_setup_issues(frame)) == 1
    frame["Capacity"] = [0, 2]
    assert _period_setup_issues(frame) == []


def test_zero_capacity_period_is_excluded_from_solver_input_and_availability():
    imported = prepared_campaign()
    slots = _slot_frame(imported)
    slots["Capacity"] = [0, 2]
    defaults = SchedulerConfig().group_policies
    problem, _ = _reviewed_problem_and_config(
        imported,
        _people_frame(imported),
        slots,
        student_defaults=defaults[InterviewerGroup.STUDENT],
        adcom_defaults=defaults[InterviewerGroup.ADCOM],
        student_priority_weight=3,
        back_to_back=BackToBackPolicy.DISCOURAGED,
        maximum_consecutive=2,
        time_limit_seconds=5,
    )
    assert [slot.id for slot in problem.slots] == ["20260224-1030"]
    assert problem.slots[0].target == 2
    assert problem.locked_assignments == ()
    assert problem.interviewers[0].available_slot_ids == frozenset(
        {"20260224-1030"}
    )


def test_editor_comparison_ignores_blank_and_numeric_dtype_differences():
    first = pd.DataFrame({"Count": [1, None], "Name": ["One", "Two"]})
    second = pd.DataFrame({"Count": [1.0, float("nan")], "Name": ["One", "Two"]})
    assert not _frames_differ(first, second)
    second.loc[1, "Count"] = 2
    assert _frames_differ(first, second)


def test_changed_editor_data_clears_a_result_and_preserves_an_explanation(
    monkeypatch,
):
    messages: list[str] = []
    fake_streamlit = SimpleNamespace(
        session_state={
            "v2_result": object(),
            "v2_problem": object(),
            "v2_config": object(),
            "v2_validation": object(),
            "v2_report": b"report",
            "v2_simplified_report": b"simplified report",
            "v2_people": "keep",
        },
        toast=messages.append,
    )
    monkeypatch.setattr(admin_app, "st", fake_streamlit)

    cleared = _invalidate_result_after_edit(
        changed=True,
        reason="The number of interviews possible changed.",
    )

    assert cleared
    assert "v2_result" not in fake_streamlit.session_state
    assert "v2_problem" not in fake_streamlit.session_state
    assert "v2_report" not in fake_streamlit.session_state
    assert "v2_simplified_report" not in fake_streamlit.session_state
    assert fake_streamlit.session_state["v2_people"] == "keep"
    assert fake_streamlit.session_state["v2_stale_result_notice"] == (
        "The number of interviews possible changed."
    )
    assert messages == ["Previous scheduling result cleared."]


def test_initial_setup_edits_do_not_show_a_stale_result_warning(monkeypatch):
    messages: list[str] = []
    fake_streamlit = SimpleNamespace(
        session_state={"v2_people": "keep"},
        toast=messages.append,
    )
    monkeypatch.setattr(admin_app, "st", fake_streamlit)

    cleared = _invalidate_result_after_edit(
        changed=True,
        reason="The interviewer list changed.",
    )

    assert not cleared
    assert "v2_stale_result_notice" not in fake_streamlit.session_state
    assert messages == []
