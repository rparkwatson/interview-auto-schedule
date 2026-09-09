"""Accessible brand styling and the schedule-journey presentation model."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape


BRAND_BLUE = "#011F5B"
BRAND_RED = "#990000"
ACTION_RED_HOVER = "#750000"
PACIFIC_BLUE = "#026CB5"
BAY_BLUE = "#06AAFC"
MORNING_YELLOW = "#D7BC6A"
COLLEGE_GRAY = "#B2B0A7"
MARINE_GRAY = "#EEEDEA"
NIGHT_STREET = "#2D2C41"
MUTED_TEXT = "#4A485C"
WHITE = "#FFFFFF"

SCHEDULE_AVAILABILITY_SECTION = "schedule-availability"
SCHEDULE_COUNTS_SECTION = "schedule-interview-counts"
SCHEDULE_RULES_SECTION = "schedule-rules"
SCHEDULE_RESULTS_SECTION = "schedule-results"
SCHEDULE_DOWNLOADS_SECTION = "schedule-downloads"


BRAND_CSS = f"""
<style>
:root {{
  --scheduler-blue: {BRAND_BLUE};
  --scheduler-red: {BRAND_RED};
  --scheduler-red-hover: {ACTION_RED_HOVER};
  --scheduler-pacific: {PACIFIC_BLUE};
  --scheduler-bay: {BAY_BLUE};
  --scheduler-yellow: {MORNING_YELLOW};
  --scheduler-college-gray: {COLLEGE_GRAY};
  --scheduler-marine-gray: {MARINE_GRAY};
  --scheduler-night: {NIGHT_STREET};
  --scheduler-muted-text: {MUTED_TEXT};
  --scheduler-white: {WHITE};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
  font-family: Arial, Helvetica, sans-serif;
  color: var(--scheduler-night);
}}

.stApp {{
  background: linear-gradient(
    180deg,
    var(--scheduler-white) 0%,
    var(--scheduler-white) 72%,
    #F8F8F6 100%
  );
}}

[data-testid="stHeader"] {{
  background: rgba(255, 255, 255, 0.96);
  border-bottom: 1px solid var(--scheduler-marine-gray);
}}

h1, h2, h3, h4 {{
  font-family: Georgia, "Times New Roman", serif;
  color: var(--scheduler-blue);
}}

h1 {{
  border-bottom: 4px solid var(--scheduler-red);
  padding-bottom: 0.38rem;
}}

a {{
  color: var(--scheduler-pacific);
  text-decoration: underline;
  text-underline-offset: 0.16em;
}}

[data-testid="stMetric"] {{
  background: var(--scheduler-white);
  border: 1px solid var(--scheduler-college-gray);
  border-top: 4px solid var(--scheduler-blue);
  border-radius: 0.45rem;
  padding: 0.85rem 1rem;
}}

[data-testid="stMetricLabel"] {{
  color: var(--scheduler-night);
  font-weight: 700;
}}

[data-testid="stExpander"] {{
  background: var(--scheduler-white);
  border: 1px solid var(--scheduler-college-gray);
  border-left: 4px solid var(--scheduler-pacific);
  border-radius: 0.45rem;
}}

[data-testid="stFileUploaderDropzone"] {{
  background: #F8F8F6;
  border-color: var(--scheduler-college-gray);
}}

[data-baseweb="tab-list"] {{
  border-bottom-color: var(--scheduler-college-gray);
}}

[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--scheduler-blue);
  font-weight: 700;
}}

.stButton button,
.stDownloadButton button,
[data-testid="stFileUploader"] button:not([aria-label^="Help for"]) {{
  min-height: 44px;
  font-weight: 700;
}}

.stButton button[kind="primary"],
.stDownloadButton button[kind="primary"] {{
  background: var(--scheduler-red);
  border-color: var(--scheduler-red);
  color: var(--scheduler-white);
}}

.stButton button[kind="primary"]:hover,
.stDownloadButton button[kind="primary"]:hover {{
  background: var(--scheduler-red-hover);
  border-color: var(--scheduler-red-hover);
  color: var(--scheduler-white);
}}

[data-testid="stMainBlockContainer"] button[aria-label="Increment"],
[data-testid="stMainBlockContainer"] button[aria-label="Decrement"] {{
  min-height: 44px;
  min-width: 44px;
}}

[data-testid="stMainBlockContainer"] button[aria-label^="Help for"],
[data-testid="stMainBlockContainer"] a[aria-label="Link to heading"] {{
  overflow: visible;
  position: relative;
}}

[data-testid="stMainBlockContainer"] button[aria-label^="Help for"]::after,
[data-testid="stMainBlockContainer"] a[aria-label="Link to heading"]::after {{
  content: "";
  height: 44px;
  left: 50%;
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 44px;
}}

button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible,
[role="tab"]:focus-visible,
[role="checkbox"]:focus-visible {{
  outline: 3px solid var(--scheduler-yellow) !important;
  outline-offset: 2px !important;
}}

.schedule-section-anchor {{
  display: block;
  height: 0;
  scroll-margin-top: 5.5rem;
}}

[data-testid="stElementContainer"]:has(.schedule-section-anchor) {{
  margin-bottom: -1rem;
}}

.schedule-visually-hidden {{
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  overflow: hidden;
  position: absolute;
  white-space: nowrap;
  width: 1px;
}}

.schedule-journey-card {{
  background: var(--scheduler-white);
  border: 1px solid var(--scheduler-college-gray);
  border-top: 5px solid var(--scheduler-blue);
  border-radius: 0.6rem;
  box-shadow: 0 8px 24px rgba(1, 31, 91, 0.16);
  color: var(--scheduler-night);
  padding: 1rem 1rem 0.9rem;
}}

.schedule-journey-head {{
  align-items: center;
  display: flex;
  gap: 0.75rem;
  margin-bottom: 0.85rem;
}}

.schedule-journey-calendar {{
  background: var(--scheduler-white);
  border: 2px solid var(--scheduler-blue);
  border-radius: 0.3rem;
  box-shadow: inset 0 -1px 0 var(--scheduler-marine-gray);
  display: inline-block;
  flex: 0 0 2.35rem;
  height: 2.35rem;
  position: relative;
}}

.schedule-journey-calendar::before {{
  background: var(--scheduler-red);
  content: "";
  height: 0.52rem;
  left: 0;
  position: absolute;
  right: 0;
  top: 0.42rem;
}}

.schedule-journey-calendar::after {{
  background: var(--scheduler-blue);
  border-radius: 1rem;
  box-shadow: 1.05rem 0 0 var(--scheduler-blue);
  content: "";
  height: 0.5rem;
  left: 0.52rem;
  position: absolute;
  top: -0.25rem;
  width: 0.18rem;
}}

.schedule-journey-title {{
  color: var(--scheduler-blue);
  font-family: Georgia, "Times New Roman", serif;
  font-size: 1.08rem;
  font-weight: 700;
  line-height: 1.2;
  margin: 0;
}}

.schedule-journey-count {{
  color: var(--scheduler-night);
  font-size: 0.82rem;
  line-height: 1.35;
  margin: 0.18rem 0 0;
}}

.schedule-journey-list {{
  list-style: none;
  margin: 0;
  padding: 0;
  position: relative;
}}

.schedule-journey-list::before {{
  background: repeating-linear-gradient(
    to bottom,
    var(--scheduler-college-gray) 0,
    var(--scheduler-college-gray) 5px,
    transparent 5px,
    transparent 9px
  );
  content: "";
  left: 0.84rem;
  position: absolute;
  top: 1rem;
  bottom: 1rem;
  width: 2px;
}}

.schedule-journey-step {{
  align-items: center;
  display: grid;
  gap: 0.7rem;
  grid-template-columns: 1.7rem 1fr;
  min-height: 2.55rem;
  position: relative;
}}

.schedule-journey-link,
.schedule-journey-static {{
  align-self: stretch;
  border-radius: 0.3rem;
  color: var(--scheduler-night);
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 44px;
  padding: 0.14rem 0.22rem;
  text-decoration: none;
}}

.schedule-journey-link {{
  cursor: pointer;
}}

.schedule-journey-link .schedule-journey-label {{
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 0.14em;
}}

.schedule-journey-link:hover {{
  background: #F8F8F6;
  color: var(--scheduler-blue);
}}

.schedule-journey-link:focus-visible {{
  outline: 3px solid var(--scheduler-yellow) !important;
  outline-offset: 1px !important;
}}

.schedule-journey-marker {{
  align-items: center;
  background: var(--scheduler-white);
  border: 2px solid var(--scheduler-college-gray);
  border-radius: 50%;
  color: var(--scheduler-night);
  display: inline-flex;
  font-size: 0.75rem;
  font-weight: 700;
  height: 1.7rem;
  justify-content: center;
  line-height: 1;
  position: relative;
  width: 1.7rem;
  z-index: 1;
}}

.schedule-journey-step.is-complete .schedule-journey-marker {{
  background: var(--scheduler-blue);
  border-color: var(--scheduler-blue);
  color: var(--scheduler-white);
}}

.schedule-journey-step.is-current .schedule-journey-marker {{
  border: 4px solid var(--scheduler-red);
  color: var(--scheduler-blue);
}}

.schedule-journey-step.needs-attention .schedule-journey-marker {{
  background: var(--scheduler-yellow);
  border-color: var(--scheduler-red);
  color: var(--scheduler-blue);
}}

.schedule-journey-label {{
  color: var(--scheduler-night);
  display: block;
  font-size: 0.86rem;
  font-weight: 700;
  line-height: 1.25;
}}

.schedule-journey-status {{
  color: var(--scheduler-muted-text);
  display: block;
  font-size: 0.75rem;
  line-height: 1.25;
  margin-top: 0.08rem;
}}

.schedule-journey-step.is-current .schedule-journey-status,
.schedule-journey-step.needs-attention .schedule-journey-status {{
  color: var(--scheduler-red);
  font-weight: 700;
}}

.schedule-journey-next {{
  background: var(--scheduler-marine-gray);
  border-left: 4px solid var(--scheduler-red);
  color: var(--scheduler-blue);
  font-size: 0.8rem;
  font-weight: 700;
  line-height: 1.35;
  margin-top: 0.8rem;
  padding: 0.6rem 0.7rem;
}}

@media (min-width: 1280px) {{
  [data-testid="stMainBlockContainer"],
  .main .block-container {{
    padding-right: 18rem;
  }}

  .schedule-journey-card {{
    position: fixed;
    right: 1.15rem;
    top: 4.75rem;
    width: 15.5rem;
    z-index: 999;
  }}
}}

@media (max-width: 1279px) {{
  .schedule-journey-card {{
    margin: 0 0 1rem;
    position: relative;
    width: auto;
  }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }}
}}
</style>
"""


@dataclass(frozen=True, slots=True)
class JourneyMilestone:
    label: str
    status: str
    anchor_id: str
    link_enabled: bool


@dataclass(frozen=True, slots=True)
class ScheduleJourney:
    milestones: tuple[JourneyMilestone, ...]
    completed_count: int
    current_message: str
    is_complete: bool


def build_schedule_journey(
    *,
    availability_ready: bool,
    counts_ready: bool,
    ready_to_run: bool,
    run_attempted: bool,
    schedule_ready: bool,
    downloads_ready: bool,
) -> ScheduleJourney:
    """Build the five-step progress state from meaningful scheduling milestones."""

    definitions = (
        ("Availability files checked", SCHEDULE_AVAILABILITY_SECTION),
        ("Interview counts entered", SCHEDULE_COUNTS_SECTION),
        ("Ready to create schedule", SCHEDULE_RULES_SECTION),
        ("Schedule created", SCHEDULE_RESULTS_SECTION),
        ("Files ready to download", SCHEDULE_DOWNLOADS_SECTION),
    )
    labels = tuple(label for label, _ in definitions)
    completed = [False] * len(labels)
    completed[0] = availability_ready
    completed[1] = availability_ready and counts_ready
    completed[2] = availability_ready and counts_ready and ready_to_run
    completed[3] = completed[2] and schedule_ready
    completed[4] = completed[3] and downloads_ready

    if completed[4]:
        statuses = ["complete"] * len(labels)
        message = "Your schedule files are ready to download"
        current_index = None
    else:
        statuses = ["complete" if item else "waiting" for item in completed]
        if not completed[0]:
            current_index = 0
            message = "Upload both availability files to begin"
        elif not completed[1]:
            current_index = 1
            message = "Enter the number of interviews for each period"
        elif not completed[2]:
            current_index = 2
            statuses[current_index] = "attention"
            message = "Correct the availability file issues"
        elif run_attempted and not schedule_ready:
            current_index = 3
            statuses[current_index] = "attention"
            message = "Review the schedule issues, then try again"
        elif not completed[3]:
            current_index = 3
            message = "Create the schedule"
        else:
            current_index = 4
            message = "Review the exceptions to enable the downloads"
        if statuses[current_index] != "attention":
            statuses[current_index] = "current"

    link_enabled = (
        True,
        availability_ready,
        availability_ready,
        run_attempted,
        schedule_ready,
    )
    milestones = tuple(
        JourneyMilestone(
            label=label,
            status=status,
            anchor_id=anchor_id,
            link_enabled=enabled,
        )
        for (label, anchor_id), status, enabled in zip(
            definitions,
            statuses,
            link_enabled,
        )
    )
    return ScheduleJourney(
        milestones=milestones,
        completed_count=sum(completed),
        current_message=message,
        is_complete=completed[4],
    )


def schedule_journey_html(journey: ScheduleJourney) -> str:
    """Render an accessible, non-animated milestone trail for Streamlit."""

    total = len(journey.milestones)
    if journey.is_complete:
        accessible_summary = f"All {total} scheduling milestones are complete."
    else:
        accessible_summary = (
            f"{journey.completed_count} of {total} scheduling milestones are complete. "
            f"{journey.current_message}."
        )
    status_labels = {
        "complete": "Completed",
        "current": "Current step",
        "attention": "Needs attention",
        "waiting": "Not started",
    }
    steps: list[str] = []
    for index, milestone in enumerate(journey.milestones, start=1):
        marker = (
            "&#10003;"
            if milestone.status == "complete"
            else "!"
            if milestone.status == "attention"
            else str(index)
        )
        class_name = (
            "needs-attention"
            if milestone.status == "attention"
            else f"is-{milestone.status}"
        )
        status_label = status_labels[milestone.status]
        if milestone.link_enabled:
            current_attribute = (
                ' aria-current="step"'
                if milestone.status in {"current", "attention"}
                else ""
            )
            step_content = (
                f'<a class="schedule-journey-link" href="#{escape(milestone.anchor_id)}" '
                f'aria-label="Go to {escape(milestone.label)}. {status_label}."'
                f"{current_attribute}>"
                f'<span class="schedule-journey-label">{escape(milestone.label)}</span>'
                f'<span class="schedule-journey-status">{status_label}</span>'
                "</a>"
            )
        else:
            step_content = (
                '<span class="schedule-journey-static">'
                f'<span class="schedule-journey-label">{escape(milestone.label)}</span>'
                f'<span class="schedule-journey-status">{status_label}</span>'
                "</span>"
            )
        steps.append(
            f'<li class="schedule-journey-step {class_name}">'
            f'<span class="schedule-journey-marker" aria-hidden="true">{marker}</span>'
            f"{step_content}"
            '</li>'
        )
    count_text = (
        "All steps completed"
        if journey.is_complete
        else f"{journey.completed_count} of {total} steps completed"
    )
    return (
        '<section class="schedule-journey-card" '
        'aria-labelledby="schedule-progress-title">'
        '<span class="schedule-visually-hidden" role="status" aria-live="polite" '
        f'aria-atomic="true">{escape(accessible_summary)}</span>'
        '<div class="schedule-journey-head">'
        '<span class="schedule-journey-calendar" aria-hidden="true"></span>'
        '<div>'
        '<p class="schedule-journey-title" id="schedule-progress-title">'
        'Schedule progress</p>'
        f'<p class="schedule-journey-count">{escape(count_text)}</p>'
        '</div>'
        '</div>'
        '<ol class="schedule-journey-list" aria-label="Scheduling steps">'
        f'{"".join(steps)}'
        '</ol>'
        f'<div class="schedule-journey-next">{escape(journey.current_message)}</div>'
        '</section>'
    )


__all__ = [
    "ACTION_RED_HOVER",
    "BAY_BLUE",
    "BRAND_BLUE",
    "BRAND_CSS",
    "BRAND_RED",
    "COLLEGE_GRAY",
    "MARINE_GRAY",
    "MORNING_YELLOW",
    "MUTED_TEXT",
    "NIGHT_STREET",
    "PACIFIC_BLUE",
    "SCHEDULE_AVAILABILITY_SECTION",
    "SCHEDULE_COUNTS_SECTION",
    "SCHEDULE_DOWNLOADS_SECTION",
    "SCHEDULE_RESULTS_SECTION",
    "SCHEDULE_RULES_SECTION",
    "ScheduleJourney",
    "WHITE",
    "build_schedule_journey",
    "schedule_journey_html",
]
