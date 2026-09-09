# V2 baseline acceptance results

Run date: 2026-09-08 (America/New_York)

These checks validate the independent baseline before user acceptance. They do
not authorize production cutover.

## Automated suite

- Result: **62 passed**
- Coverage: domain invariants, stable IDs, validation, source-role detection,
  Student-derived period detection (including zero-availability periods), roster
  preservation, 30-minute reconciliation, single-count period setup, protected
  Excel template round trips, stale/edited template rejection, zero-capacity period
  exclusion, campaign assembly, shared capacity, Student weighting, cumulative
  history, locks, strict/advisory behavior, independent minimum/maximum
  relaxation, consecutive limits, optional group targets, overlapping-slot
  protection, reproducibility, required workbook sheets/columns, simplified
  schedule ordering/capacity/filename/alignment, brand-palette contrast,
  schedule-progress state and accessibility semantics, and Streamlit startup.

## GitHub review-release preparation

Review date: 2026-09-09 (America/New_York)

- The production dependency file contains only packages needed to run the app;
  test tooling is isolated in `requirements-dev.txt`.
- A GitHub workflow is prepared to run the full suite and a repository-root
  Streamlit health check on Linux with Python 3.11, matching the current hosted
  production runtime. Its first authoritative run is pending publication of the
  review branch.
- Local verification remains green on Python 3.12.10: **62 passed**, dependency
  consistency passed, and the repository-root health endpoint returned `ok`.
- The root `.gitignore` excludes credentials, local environments, uploaded Excel
  source files, and locally generated Excel schedules.
- The duplicate v2-local Streamlit configuration was removed so local review and
  Community Cloud both use the single configuration at repository root.
- No commit, push, pull request, or hosted-app setting change was made during
  this preparation step.

Command from the repository root:

```powershell
$env:PYTHONPATH = "v2"
python -m pytest v2/tests -q
```

## Supplied Winter 2026 examples

| Measure | Result |
|---|---:|
| Interviewers | 28 (20 Student, 8 Adcom) |
| Student-derived interview periods | 40 across 8 dates |
| Matched availability records | 720 |
| Source notices | 9 warnings |
| Strict solve status | Optimal |
| Assignments | 142 |
| Interviewers below minimum | 0 |
| Discouraged back-to-back pairs | 0 |
| Slot-target deficit | 98 |
| Solver wall time | 0.69 seconds |
| In-memory workbook generated | Yes, all 9 required sheets |

The nine source warnings are the missing separate Adcom roster plus eight
explicit 10:00 AM → 10:30 AM reconciliations, one for each date. Credential and
example/helper sheets were not parsed.

The target deficit is expected and is now explained in preflight: the 40 slots
request 240 assignments, while availability plus cumulative/daily maximums cap
the supplied 28-person roster at 142 new assignments. This is a target shortfall,
not a violation of minimums, maximums, capacity, or spacing.

## Typical-scale synthetic gate

Configuration: 65 interviewers (40 Student, 25 Adcom), 150 slots, 9,750 binary
assignment decisions, deterministic 80% availability pattern, shared capacity
and target of three per slot, default group policies, 30-second solver budget,
single search worker.

| Measure | Result |
|---|---:|
| Solve status | Optimal |
| Assignments | 350 |
| Interviewers below minimum | 0 |
| Discouraged back-to-back pairs | 0 |
| Slot-target deficit | 100 |
| Solver wall time | 10.66 seconds |

The target deficit is again bounded by interviewer maximums: requested coverage
is 450 while aggregate cumulative maxima allow 350 assignments.

## Remaining acceptance work

- User review of the guided workflow and terminology.
- Manual spot-check of assignments for representative Student and Adcom names.
- A real case with historical counts and dated locks.
- An approved strict-infeasible/advisory-relaxation example.
- Downstream owner review of all workbook sheets.
- Parallel run during one production campaign before changing the deployment
  entry point.

## Administrative UX verification

Run date: 2026-09-08 (America/New_York)

- Automated result: **62 passed**.
- The supplied Student and Adcom workbooks were uploaded through the local
  Streamlit UI; no separate date/time-slot upload was requested.
- The Student workbook produced all 40 candidate periods, including headers with
  no Student availability, and displayed Student/Adcom availability counts beside
  each period.
- Both count-entry paths were exercised: direct bulk entry and download, complete,
  and reupload of the protected interview-period worksheet. Each path now asks for
  only `Interviews possible`, which is used as both desired staffing and shared capacity.
- Step 2 no longer displays a preassigned-interview editor or a separate
  `Interviews to schedule` field.
- The setup kept schedule creation disabled while counts were blank and enabled it
  after all 40 periods were completed. Explicit zero-capacity periods are removed
  from the solver input and interviewer availability before adjacency is computed.
- Step 2 edits made after a scheduling attempt clear the stale result and report,
  show an immediate notification, and keep a plain-language warning visible until
  a new schedule is created. Initial setup edits do not display a stale-result warning.
- Nine raw import notices were presented as two grouped file-check messages.
- The supplied files produced an optimal standard schedule with 142 assignments,
  0 people below minimum, and a plain-language explanation of the 98 unfilled seats.
- An intentionally infeasible rule set confirmed that exception choices remain
  hidden until the standard attempt fails.
- Exception reruns require explicit acknowledgment, display the affected people
  by name and group, and keep workbook download disabled until review is confirmed.
- Successful results provide separate full and simplified workbook downloads built
  from the same in-memory result. The simplified workbook was data-inspected and
  rendered to confirm chronological rows, `MM/DD` dates, readable 12-hour times,
  one name per Group column, blank unfilled groups, gray unavailable groups, and
  no formula errors or clipped content.
- The opening, file-check, derived-period, direct-entry, Excel round-trip, rules,
  standard-result, and exception-result screens were verified. The updated screens
  were visually reviewed in a local browser with no console errors.
- The branded interface was reviewed at desktop width. The five-step progress card
  remained at the same viewport position while the page scrolled, and its current,
  complete, waiting, and attention states were verified in automated tests. Its
  wording is plain English and its calendar icon contains no number or date.
- Available progress steps render as keyboard-accessible links to stable workflow
  anchors. Future steps remain non-link status text until their sections exist. A
  browser check confirmed the link updates the page fragment and reaches the
  availability section without a rerun or console error.
- A browser-level audit found no visible normal-size text below 4.5:1 contrast.
  All application buttons, help controls, and number-input steppers rendered with
  at least 44-by-44-pixel targets. Expanded help targets no longer change field or
  upload-row alignment. The progress card uses text and symbols alongside color,
  provides a polite accessible status, has no looping animation, and returns to
  normal document flow below the desktop breakpoint.

Production cutover remains subject to the business acceptance items above.
