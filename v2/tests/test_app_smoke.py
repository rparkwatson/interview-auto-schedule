from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest
import pytest

from interview_scheduler_v2 import (
    Interviewer,
    InterviewerGroup,
    SchedulerConfig,
    SchedulingProblem,
    Slot,
)
from interview_scheduler_v2 import admin_app
from interview_scheduler_v2.io import (
    CampaignImportResult,
    ParsedAvailability,
    ParsedSlot,
    ParsedSlotSet,
)
from interview_scheduler_v2.optimization import SolveResult, SolveStatus


def test_app_starts_at_guided_import_step_without_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    assert not app.exception
    assert app.title[0].value == "Interview Scheduler"
    assert any(
        "Upload both availability files in Step 1" in item.value
        for item in app.info
    )
    assert any(
        item.label == "Find interview periods and continue" for item in app.button
    )
    assert len(app.get("file_uploader")) == 2
    assert not any("exception" in item.label.lower() for item in app.selectbox)
    assert any(
        "schedule-journey-card" in item.value for item in app.markdown
    )
    assert any(
        'href="#schedule-availability"' in item.value for item in app.markdown
    )
    assert any(
        'id="schedule-availability"' in item.value for item in app.markdown
    )


def _prepared_campaign() -> CampaignImportResult:
    zone = ZoneInfo("America/New_York")
    start = datetime(2026, 2, 24, 8, tzinfo=zone)
    slot = Slot(
        "20260224-0800",
        start,
        start + timedelta(minutes=90),
        0,
        0,
    )
    person = Interviewer.create(
        name="Student One",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=[slot.id],
    )
    return CampaignImportResult(
        problem=SchedulingProblem((person,), (slot,)),
        notices=(),
        interviewer_ids={},
        student_availability=ParsedAvailability("student", [], [person.name]),
        adcom_availability=ParsedAvailability("adcom", [], []),
        slot_set=ParsedSlotSet(
            [ParsedSlot(slot.id, slot.start, slot.end, 0, 0, slot.id)]
        ),
        periods_need_configuration=True,
    )


def test_review_step_uses_one_period_count_and_has_no_preassignment_editor():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    imported = _prepared_campaign()
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    state = {
        "v2_import": imported,
        "v2_scenario": "UX check",
        "v2_campaign_year": 2026,
        "v2_timezone": "America/New_York",
        "v2_people": admin_app._people_frame(imported),
        "v2_slots": admin_app._slot_frame(imported),
        "v2_slot_editor_revision": 0,
        "v2_period_upload_revision": 0,
    }
    for key, value in state.items():
        app.session_state[key] = value
    app.run()

    assert not app.exception
    visible_text = " ".join(
        str(getattr(item, "value", ""))
        for group in (app.markdown, app.caption)
        for item in group
    )
    assert "Preassigned interviews" not in visible_text
    assert "Interviews to schedule" not in visible_text
    assert 'href="#schedule-interview-counts"' in visible_text
    assert 'href="#schedule-rules"' in visible_text
    assert 'href="#schedule-results"' not in visible_text
    assert 'id="schedule-interview-counts"' in visible_text
    assert 'id="schedule-rules"' in visible_text
    slot_tables = [
        item.value
        for item in app.get("dataframe")
        if "Capacity" in item.value.columns
    ]
    assert len(slot_tables) == 1
    assert "Target" not in slot_tables[0].columns


class _ResultColumn:
    def __init__(self) -> None:
        self.downloads: list[tuple[str, dict]] = []

    def metric(self, *args, **kwargs) -> None:
        return None

    def download_button(self, label: str, **kwargs) -> None:
        self.downloads.append((label, kwargs))


class _ResultStreamlit:
    def __init__(self) -> None:
        self.session_state = {
            "v2_problem": SchedulingProblem((), ()),
            "v2_config": SchedulerConfig(),
        }
        self.metric_columns = [_ResultColumn() for _ in range(4)]
        self.download_columns = [_ResultColumn() for _ in range(2)]

    def columns(self, count: int):
        return self.metric_columns if count == 4 else self.download_columns

    def tabs(self, labels):
        return [nullcontext() for _ in labels]

    def expander(self, *args, **kwargs):
        return nullcontext()

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def test_success_result_offers_full_and_simplified_downloads(monkeypatch):
    fake_streamlit = _ResultStreamlit()
    monkeypatch.setattr(admin_app, "st", fake_streamlit)
    result = SolveResult(status=SolveStatus.OPTIMAL, scenario="Winter 2026")

    admin_app._show_success(result, SimpleNamespace(notices=()))

    full_label, full_options = fake_streamlit.download_columns[0].downloads[0]
    simple_label, simple_options = fake_streamlit.download_columns[1].downloads[0]
    assert full_label == "Download full schedule workbook"
    assert simple_label == "Download simplified schedule"
    assert full_options["data"].startswith(b"PK")
    assert simple_options["data"].startswith(b"PK")
    assert full_options["file_name"].startswith("Winter_2026_interview_schedule_")
    assert simple_options["file_name"].startswith(
        "Winter_2026_simplified_schedule_"
    )
    assert not full_options["disabled"]
    assert not simple_options["disabled"]


def _app_with_result(*, historical_count=3):
    """A real Streamlit workflow with one reviewed, imported interview period."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=15).run()
    imported = _prepared_campaign()
    person = replace(imported.problem.interviewers[0], historical_prior_count=historical_count)
    imported = replace(imported, problem=replace(imported.problem, interviewers=(person,)))
    slots = admin_app._slot_frame(imported)
    slots["Capacity"] = [1]
    for key, value in {
        "v2_import": imported, "v2_scenario": "State test", "v2_campaign_year": 2026,
        "v2_people": admin_app._people_frame(imported), "v2_slots": slots,
        "v2_slot_editor_revision": 0, "v2_period_upload_revision": 0,
    }.items():
        app.session_state[key] = value
    app.run()
    next(item for item in app.button if item.label == "Create schedule using standard rules").click().run()
    assert not app.exception
    return app


def _assert_no_cached_result(app):
    assert not app.exception
    assert "v2_result" not in app.session_state
    assert "v2_report" not in app.session_state
    assert "v2_simplified_report" not in app.session_state
    assert not any("schedule" in item.label.lower() for item in app.get("download_button"))
    assert not any(item.label == "Try again with this exception" for item in app.button)


def test_custom_daily_limit_edit_clears_old_downloads_and_rebuilds_with_new_rule():
    app = _app_with_result()
    assert len(app.session_state["v2_result"].assignments) == 1
    app.checkbox(key="v2_recommended_rules").uncheck().run()
    _assert_no_cached_result(app)
    next(item for item in app.button if item.label == "Create schedule using standard rules").click().run()
    app.number_input(key="student_day_max").set_value(0).run()
    _assert_no_cached_result(app)
    assert any("scheduling rules changed" in item.value for item in app.warning)
    next(item for item in app.button if item.label == "Create schedule using standard rules").click().run()
    assert not app.exception
    assert app.session_state["v2_config"].group_policies[InterviewerGroup.STUDENT].max_per_day == 0
    assert not app.session_state["v2_result"].assignments


@pytest.mark.parametrize(("kind", "key", "value"), [
    ("number_input", "v2_time_limit", 35),
    ("number_input", "v2_max_consecutive", 3),
    ("selectbox", "v2_priority", "Treat Student and Adcom goals equally"),
    ("selectbox", "v2_consecutive_policy", "Never schedule consecutive interviews"),
    ("text_input", "v2_scenario_input", "Renamed campaign"),
])
def test_schedule_setting_edits_clear_downloads(kind, key, value):
    app = _app_with_result()
    getattr(app, kind)(key=key).set_value(value).run()
    _assert_no_cached_result(app)
    assert "v2_import" in app.session_state
    assert "v2_people" in app.session_state


def test_year_change_requires_reimport_and_hides_previous_results():
    app = _app_with_result()
    app.number_input(key="v2_year_input").set_value(2027).run()
    _assert_no_cached_result(app)
    assert "v2_import" not in app.session_state
    assert "v2_people" not in app.session_state
    assert any("interview year changed" in item.value for item in app.warning)


def test_rule_edit_after_infeasible_run_removes_stale_exception_retry():
    app = _app_with_result(historical_count=0)
    assert app.session_state["v2_result"].status == SolveStatus.INFEASIBLE
    assert any(item.label == "Try again with this exception" for item in app.button)
    app.checkbox(key="v2_recommended_rules").uncheck().run()
    _assert_no_cached_result(app)


def test_unrelated_rerun_keeps_schedule_downloads_and_run_identity():
    app = _app_with_result()
    result = app.session_state["v2_result"]
    full_report = app.session_state["v2_report"]
    simplified_report = app.session_state["v2_simplified_report"]
    labels = {"Download full schedule workbook", "Download simplified schedule"}

    def schedule_urls():
        # The input worksheet is regenerated, including its XLSX creation time.
        # Only the two solved reports must retain their bytes and download URLs.
        return {
            item.label: item.proto.url
            for item in app.get("download_button")
            if item.label in labels
        }

    urls = schedule_urls()
    assert set(urls) == labels
    assert all(urls.values())
    app.run()
    assert not app.exception
    assert app.session_state["v2_result"].settings["run_id"] == result.settings["run_id"]
    assert app.session_state["v2_report"] == full_report
    assert app.session_state["v2_simplified_report"] == simplified_report
    assert schedule_urls() == urls
