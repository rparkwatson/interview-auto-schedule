# Interview Scheduler v2

This directory contains the independent, single-assignment scheduler. It does
not import or mutate the legacy application. The prioritized delivery and
cutover plan is in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).
The current technical verification record is in
[ACCEPTANCE_RESULTS.md](ACCEPTANCE_RESULTS.md).
The staged GitHub and Streamlit release checks are in
[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md).

## Scheduling contract

- Groups are **Student Interviewer** and **Adcom Interviewer**.
- Every interviewer assignment consumes one shared slot-capacity unit.
- Each period has one `Interviews possible` count. It is both the desired staffing
  count and the maximum shared capacity.
- Availability, total maximums, and daily maximums are hard.
- Group minimums and individual cumulative minimums are hard in strict mode and
  may be relaxed with named shortfalls in advisory mode.
- Consecutive listed slots are back-to-back. Back-to-back assignments are
  discouraged and runs longer than two assignments are prohibited by default.
- All datetimes use `America/New_York` unless a campaign selects another IANA
  timezone.

## Default policies

| Group | Minimum | Target | Maximum | Maximum/day |
|---|---:|---:|---:|---:|
| Student Interviewer | 3 | 4 | 5 | 2 |
| Adcom Interviewer | 4 | 4 | 6 | 2 |

Historical counts contribute to cumulative totals. The administrative workflow
does not collect preassigned interviews.

## Local development

From the repository root, so local execution uses the same root Streamlit
configuration and working directory as the hosted app:

```powershell
python -m venv v2\.venv
.\v2\.venv\Scripts\python -m pip install -r v2\requirements-dev.txt
.\v2\.venv\Scripts\python -m pytest v2\tests
.\v2\.venv\Scripts\streamlit run v2\app.py
```

Python 3.11 or newer is required. The production app currently uses Python
3.11, and the automated GitHub release check validates v2 against that version.
`requirements.txt` contains only packages required by the hosted app;
`requirements-dev.txt` adds the automated test tools used during development.

## Administrative workflow

1. Upload the Student and Adcom availability files. The Student workbook's date
   headings and time worksheets create the complete candidate-period list.
2. Enter interview counts directly or download the generated interview-period
   worksheet, complete its yellow cells in Excel, and upload it again.
3. Review grouped file-check messages, people, and optional group preferences.
4. Confirm the recommended rules or open the optional settings that need to change.
5. Create a schedule using standard rules first. Exception choices appear only
   when the standard rules cannot produce a schedule.
6. Review staffing readiness, unfilled seats, and any named interviewers affected
   by an approved exception before downloading either the full schedule workbook
   or the separate simplified schedule.

`Interviews possible` is required for each candidate period and is the number the
scheduler will try to fill. An explicit zero means the period will not be offered
and removes it from the scheduling model.
If any Step 2 scheduling information changes after a result is created, the app
clears that result automatically and keeps a warning visible until the schedule
is created again.

Technical IDs, diagnostic references, and processing details remain available
inside collapsed sections for troubleshooting and audit work.

## Visual design and schedule progress

The administrative interface uses a blue-dominant palette with red reserved for
primary actions and emphasis, supported by neutral gray, white, light blue, and
gold. Arial and Georgia provide broadly available, readable font fallbacks. No
institutional name, logo, or identifying mark appears in the visual treatment.

A five-step `Schedule progress` card remains visible while a desktop user scrolls:

1. Availability files checked
2. Interview counts entered
3. Ready to create schedule
4. Schedule created
5. Files ready to download

The progress display is derived from current Streamlit session data rather than clicks or
page position. A failed run and blocking file checks display `Needs attention`.
Completed, current, waiting, and attention states use visible text and symbols in
addition to color. Each step becomes a keyboard-accessible in-page link when its
section is available; future steps remain plain status text until their content
exists. On narrower displays the card returns to normal page flow so it cannot
cover fields or actions.

Normal-size text color pairs meet a minimum 4.5:1 contrast ratio. Primary
controls, help buttons, and number-input steppers provide at least 44-by-44-pixel
click targets, keyboard focus remains visible, and the status card includes a
polite screen-reader announcement. The design does not use looping animation and
honors reduced-motion preferences.

## Architecture

- `interview_scheduler_v2/domain.py` — immutable scheduling inputs
- `interview_scheduler_v2/io/` — workbook parsing, period-template generation,
  reconciliation, and campaign assembly
- `interview_scheduler_v2/validation.py` — structured preflight diagnostics
- `interview_scheduler_v2/optimization/` — CP-SAT model and result contract
- `interview_scheduler_v2/presentation/` — plain-language administrative messages
- `interview_scheduler_v2/presentation/branding.py` — accessible palette,
  interface styling, and schedule-progress states
- `interview_scheduler_v2/reporting/` — nine-sheet full report and separate
  one-sheet simplified schedule, both generated in memory
- `interview_scheduler_v2/admin_app.py` — guided single-user Streamlit workflow
- `app.py` — minimal Streamlit entry point
- `tests/` — unit, integration, solver, and export contract tests

The v2 application accepts separate Student and Adcom availability workbooks.
It derives candidate dates and times from the Student workbook and creates the
fillable interview-period worksheet entirely in memory. The worksheet protects
the period structure, provides one yellow count cell per period, validates
whole-number counts, includes Student and Adcom
availability totals for context, and is accepted only by the availability setup
that generated it. Final schedule reports are also generated in memory. The
simplified schedule has one row per interview period, formatted date and start
time columns, and one interviewer per capacity-based `Group A`, `Group B`, and
subsequent column. Rows are ordered chronologically, blank Group cells identify
unfilled interviews, and gray cells identify groups outside a period's capacity.

The lower-level import adapter still supports the earlier `Date_Time` /
`Max_Slot` slot-workbook contract for programmatic compatibility, but the
administrative Streamlit workflow no longer requests that third source file.
