# Interview Scheduler v2 implementation plan

## Product decision

V2 is an independent application with no paired-assignment model and no runtime
dependency on the legacy scheduler. It will be accepted alongside v1, then
become the production entry point in a separate cutover change.

Canonical terms are **Student Interviewer** and **Adcom Interviewer**. Every
assignment is one interviewer in one authoritative slot and consumes exactly
one unit from the capacity shared by both groups.

## Prioritized delivery order

| Order | Priority | Change | Impact | Completion / acceptance gate |
|---:|:---:|---|---|---|
| 1 | P0 | Canonical domain, source contract, ET-aware slots, stable IDs | Removes paired concepts at the foundation and prevents ambiguous joins by name | Implemented; workbook examples import as 28 people, 40 slots, and 720 availability records |
| 2 | P0 | Structured validation before optimization | Stops malformed targets, IDs, dates, availability, locks, and impossible local limits from becoming unexplained solver failures | Implemented; every issue has code, severity, constraint family, path, and relaxation metadata |
| 3 | P0 | Shared-capacity CP-SAT scheduler | Provides the core scheduling outcome with cumulative history, dated locks, total/daily limits, slot targets, group targets, and spacing | Implemented; strict example campaign solves optimally with no minimum shortfalls |
| 4 | P0 | Explicit strict/advisory relaxation | Makes exceptions a deliberate operator choice and identifies each person below a minimum or above a maximum | Implemented; minimum-only, maximum-only, and combined modes produce named warnings |
| 5 | P1 | Guided and accessible Streamlit workflow | Reduces operator error with plain-language file checks, standard-first scheduling, progressive settings, readiness-focused results, accessible brand styling, and persistent schedule progress | Administrative redesign implemented and locally verified; business acceptance is the gate |
| 6 | P1 | In-memory schedule workbooks | Gives operations both a complete audit file and a simple period-by-group handoff without shared-file collisions | Implemented with the nine-sheet full report, a separate simplified schedule, and scenario-specific filenames |
| 7 | P1 | Automated and example-based acceptance tests | Protects source-role detection, single assignments, capacity, history, locks, spacing, relaxations, reproducibility, and exports | Implemented baseline; add user-discovered edge cases during acceptance |
| 8 | P1 | Parallel acceptance and production cutover | Preserves a safe rollback point while v2 results are compared with manually reviewed expectations | Pending business acceptance |
| 9 | P2 | Advanced scenario comparison and repair workflow | Speeds policy tuning and makes late availability/lock changes less disruptive | Post-acceptance enhancement |

## Work packages and impact

### 1. Inputs and identity (P0)

- Derive the authoritative candidate dates, start times, and end times from the
  Student availability workbook, including periods with no entered Student
  availability. Collect one shared interview count afterward in the app or
  through a protected Excel worksheet generated for the current setup.
- Require `Interviews possible` for every candidate period and use it as both
  the desired staffing count and maximum shared capacity. Interpret an explicit
  zero as an excluded period.
- Validate a completed period worksheet against a fingerprint of the current
  candidate list so stale files and edited period structures cannot be applied.
- Parse only role-specific Student time sheets and the explicit Adcom
  availability sheet. Never inspect example, helper, ignore, Zoom-link, or
  credential sheets as scheduling data.
- Reconcile only a unique same-day start within 30 minutes. Record each
  reconciliation; never silently modify source times.
- Generate deterministic `STU-############` and `ADC-############` IDs from
  normalized names. Preserve explicit IDs when a future roster provides them.
- Preserve Student roster members with zero availability. Warn that the example
  Adcom source has no separate roster and allow missing members to be added in
  the People editor.

Impact: imports become predictable, names no longer act as fragile primary
keys, and the 10:00/10:30 Adcom discrepancy is visible to the operator.

### 2. Constraints and validation (P0)

- Apply the supplied Student defaults (3/4/5 total, 2/day) and Adcom defaults
  (4/4/6 total, 2/day), with zero minimum on an active day.
- Allow per-person overrides without changing group defaults.
- Include historical counts in cumulative minimum, target, and maximum totals.
- Count locks against availability, slot capacity, cumulative totals, daily
  totals, adjacency, and the consecutive-run limit.
- Define adjacency as consecutive listed slots on the same date. Discourage
  adjacent assignments and prohibit a run longer than two by default.
- Warn when authoritative slot intervals overlap and prohibit assigning one
  interviewer to two overlapping intervals.

Impact: infeasibility is diagnosed before scheduling where possible, and every
rule has one consistent interpretation in validation, optimization, UI, and
exports.

### 3. Optimization and scheduling enhancements (P0)

The solver uses ordered objectives so a later aesthetic choice cannot trade
away an earlier operational requirement:

1. minimize explicitly permitted relaxation cost;
2. minimize total and group slot-target deviation;
3. minimize the worst individual target shortfall in each group;
4. minimize weighted total individual target shortfall, with adjustable Student
   priority;
5. minimize discouraged back-to-back pairs, assignments above individual
   targets, and lower-preference assignments.

The default seeded, single-worker CP-SAT search makes remaining ties
reproducible without adding a large final tie-break objective.

Maximums are ceilings, not quotas. The Student priority control therefore
weights Student minimum and target attainment; it does not reward pushing a
Student interviewer to their maximum.

Impact: schedules prioritize operational coverage, then fair target attainment,
then quality, while remaining reproducible with the default single-worker
configuration.

### 4. UX workflow (P1)

1. **Check availability** — upload the Student and Adcom files, name the schedule,
   confirm the interview year, derive periods from the Student file, and review
   grouped checks written in administrative language.
2. **Set interview counts** — enter counts directly, apply one count in bulk, or
   download a protected worksheet and upload the completed copy. Show Student and
   Adcom availability totals beside every period for context.
3. **Review information** — edit prior counts, optionally override individual
   limits, and set optional group preferences. Clear any existing result when
   these inputs change and explain that the schedule must be created again.
4. **Confirm rules** — show recommended group rules by default, with priority,
   consecutive-interview behavior, custom limits, and processing time available
   only where they are useful.
5. **Create with standard rules** — always attempt all minimums and maximums as
   requirements before offering an exception.
6. **Handle a failed attempt** — explain the conflict in plain language and show
   minimum/maximum exception choices only after failure and explicit acknowledgment.
7. **Review readiness** — lead with filled periods, unfilled seats, people below
   minimum, and a named table of everyone affected by an exception. Keep raw IDs,
   codes, and processing details in collapsed technical sections.
8. **Download** — require an additional review acknowledgment for exception
   schedules, then offer both the full schedule workbook and a separate simplified
   period-by-group schedule. Generate both entirely in memory from the same result.

The interface uses a blue-dominant, red-accented institutional palette without
logos or identifying references. It uses approved system font fallbacks, tested
normal-text contrast, visible keyboard focus, large click targets, and status cues
that do not rely on color alone. A fixed desktop `Schedule progress` card tracks five
data-backed milestones from availability through download readiness. It identifies
blocking checks and failed runs as needing attention, announces changes politely
to assistive technology, and provides keyboard-accessible links to each workflow
section as that section becomes available. It avoids decorative motion and moves
into normal page flow on narrower screens to prevent overlap.

Impact: the operator can see what changed, what was relaxed, and who is affected
before distributing a schedule.

### 5. Reporting and audit (P1)

The workbook contains, in order:

1. `Assignments`
2. `Schedule_By_Slot`
3. `Interviewer_Summary`
4. `Student_only Schedule_by_slot`
5. `Adcom_only Schedule_by_slot`
6. `Group_Summary`
7. `Slot_Summary`
8. `Constraint_Diagnostics`
9. `Run_Settings`

Every assignment carries Scenario, Slot ID, Start, End, Interviewer ID,
Interviewer Name, Group, Locked, and Preference. Run settings record policies,
relaxation mode, seed, solver limit, objective values, and source notices.

A separate simplified workbook contains one `Schedule` sheet. It has one row per
interview period, sorted by date and start time, with `Slot Date`, `Start Time`,
and capacity-based `Group A`, `Group B`, and subsequent columns. Each assigned
interviewer appears once in a Group cell. Blank available Group cells show
unfilled interviews, while gray cells show groups outside that period's capacity.
The simplified workbook is built directly from the same result used by the full
report and is governed by the same exception-review download safeguard.

Impact: each exported schedule is self-describing and can be audited without
access to Streamlit session state.

## Acceptance plan

### Automated gates

- Source-role and roster detection tests pass.
- No assignment exceeds shared slot capacity.
- No interviewer is assigned outside availability unless a future policy
  explicitly changes that rule.
- Historical counts and locks affect all relevant totals.
- Strict mode never returns a schedule that violates a hard limit.
- Advisory results name every minimum shortfall and maximum overage.
- No run exceeds two consecutive listed slots.
- Same seed/input/settings produce the same assignment set.
- All nine workbook sheets and required assignment columns exist.
- The simplified workbook remains chronologically sorted, represents every slot's
  capacity, and contains exactly the interviewer names assigned in the full report.

### Business acceptance scenarios

- Run the supplied Winter 2026 examples and review every source warning.
- Compare slot coverage, group mix, and named totals against an independently
  reviewed manual expectation.
- Exercise at least one strict infeasible case, one minimum-only relaxation, one
  maximum relaxation caused by a lock/history conflict, and one optional group
  target case.
- Validate the typical operating size of roughly 65 people and 150 slots within
  the agreed interactive time limit.
- Confirm workbook usability with the downstream owners of each summary sheet.

## Cutover and rollback

1. Freeze the accepted v2 input/output contract and tag an acceptance build.
2. Run v1 and v2 in parallel for at least one real campaign; v2 remains the
   proposed result and receives human approval before distribution.
3. Change the deployment entry point to `v2/app.py` in a dedicated release
   change. Do not delete v1 during cutover.
4. Retain the previous deployment artifact and configuration for immediate
   rollback through the first completed production campaign.
5. After the retention window, archive v1 and its paired terminology rather than
   mixing compatibility logic into v2.

## Post-acceptance enhancements (P2)

- Side-by-side scenario comparison with changed assignments highlighted.
- Repair mode that minimizes changes after a late lock or availability update.
- Capacity/constraint sensitivity analysis explaining which small policy change
  would recover the most unmet target coverage.
- Optional persistent roster registry for IDs that survive display-name changes.
- Rich availability preferences (1–5) and preference-source templates.
- Downloadable infeasibility packet for operational escalation.
