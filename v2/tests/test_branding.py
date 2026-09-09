from __future__ import annotations

import pytest

from interview_scheduler_v2.presentation.branding import (
    ACTION_RED_HOVER,
    BAY_BLUE,
    BRAND_BLUE,
    BRAND_CSS,
    BRAND_RED,
    MARINE_GRAY,
    MORNING_YELLOW,
    MUTED_TEXT,
    NIGHT_STREET,
    WHITE,
    build_schedule_journey,
    schedule_journey_html,
)


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(first: str, second: str) -> float:
    first_luminance = _relative_luminance(first)
    second_luminance = _relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize(
    ("foreground", "background"),
    (
        (WHITE, BRAND_BLUE),
        (WHITE, BRAND_RED),
        (WHITE, ACTION_RED_HOVER),
        (NIGHT_STREET, WHITE),
        (NIGHT_STREET, MARINE_GRAY),
        (MUTED_TEXT, WHITE),
        (BRAND_BLUE, BAY_BLUE),
        (BRAND_BLUE, MORNING_YELLOW),
    ),
)
def test_brand_text_pairs_meet_wcag_aa_normal_text_contrast(
    foreground: str,
    background: str,
):
    assert _contrast_ratio(foreground, background) >= 4.5


def test_schedule_journey_advances_through_meaningful_milestones():
    initial = build_schedule_journey(
        availability_ready=False,
        counts_ready=False,
        ready_to_run=False,
        run_attempted=False,
        schedule_ready=False,
        downloads_ready=False,
    )
    ready = build_schedule_journey(
        availability_ready=True,
        counts_ready=True,
        ready_to_run=True,
        run_attempted=False,
        schedule_ready=False,
        downloads_ready=False,
    )
    waiting_for_review = build_schedule_journey(
        availability_ready=True,
        counts_ready=True,
        ready_to_run=True,
        run_attempted=True,
        schedule_ready=True,
        downloads_ready=False,
    )
    complete = build_schedule_journey(
        availability_ready=True,
        counts_ready=True,
        ready_to_run=True,
        run_attempted=True,
        schedule_ready=True,
        downloads_ready=True,
    )

    assert initial.completed_count == 0
    assert initial.milestones[0].status == "current"
    assert [item.link_enabled for item in initial.milestones] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert ready.completed_count == 3
    assert ready.milestones[3].status == "current"
    assert [item.link_enabled for item in ready.milestones] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert waiting_for_review.completed_count == 4
    assert waiting_for_review.milestones[4].status == "current"
    assert all(item.link_enabled for item in waiting_for_review.milestones)
    assert complete.completed_count == 5
    assert complete.is_complete
    assert all(item.status == "complete" for item in complete.milestones)


def test_schedule_journey_marks_blockers_and_failed_runs_as_needing_attention():
    file_issue = build_schedule_journey(
        availability_ready=True,
        counts_ready=True,
        ready_to_run=False,
        run_attempted=False,
        schedule_ready=False,
        downloads_ready=False,
    )
    failed_run = build_schedule_journey(
        availability_ready=True,
        counts_ready=True,
        ready_to_run=True,
        run_attempted=True,
        schedule_ready=False,
        downloads_ready=False,
    )

    assert file_issue.milestones[2].status == "attention"
    assert failed_run.milestones[3].status == "attention"
    assert [item.link_enabled for item in failed_run.milestones] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert "attention" in schedule_journey_html(failed_run).lower()
    assert 'href="#schedule-results"' in schedule_journey_html(failed_run)
    assert 'href="#schedule-downloads"' not in schedule_journey_html(failed_run)


def test_schedule_journey_html_has_visible_and_screen_reader_status_cues():
    journey = build_schedule_journey(
        availability_ready=True,
        counts_ready=False,
        ready_to_run=False,
        run_attempted=False,
        schedule_ready=False,
        downloads_ready=False,
    )

    content = schedule_journey_html(journey)

    assert '<span class="schedule-visually-hidden" role="status"' in content
    assert '<section class="schedule-journey-card" role="status"' not in content
    assert 'aria-live="polite"' in content
    assert 'aria-label="Scheduling steps"' in content
    assert 'href="#schedule-availability"' in content
    assert 'href="#schedule-interview-counts"' in content
    assert 'href="#schedule-rules"' in content
    assert 'href="#schedule-results"' not in content
    assert 'href="#schedule-downloads"' not in content
    assert content.count('class="schedule-journey-link"') == 3
    assert content.count('class="schedule-journey-static"') == 2
    assert 'aria-current="step"' in content
    assert "Completed" in content
    assert "Current step" in content
    assert "Not started" in content
    assert "Schedule progress" in content
    assert "Route" not in content
    assert "Next stop" not in content
    assert "Wharton" not in content
    assert "logo" not in content.lower()
    assert '<span class="schedule-journey-calendar" aria-hidden="true"></span>' in content
    assert "animation:" not in BRAND_CSS
    assert "prefers-reduced-motion" in BRAND_CSS
    assert "min-height: 44px" in BRAND_CSS
    assert "min-width: 44px" in BRAND_CSS
    assert "scroll-margin-top: 5.5rem" in BRAND_CSS
    assert 'button[aria-label^="Help for"]::after' in BRAND_CSS
    assert (
        '[data-testid="stFileUploader"] button:not([aria-label^="Help for"])'
        in BRAND_CSS
    )
    assert '[data-testid="stFileUploader"] button {' not in BRAND_CSS
