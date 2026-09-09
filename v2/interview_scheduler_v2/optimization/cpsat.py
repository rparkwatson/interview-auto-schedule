"""OR-Tools CP-SAT model for shared-capacity, single-person assignments."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Callable, Iterable

from ortools.sat.python import cp_model

from ..config import (
    DEFAULT_CONFIG,
    BackToBackPolicy,
    RelaxationMode,
    SchedulerConfig,
)
from ..domain import InterviewerGroup, SchedulingProblem
from ..validation import Severity, ValidationIssue, validate_problem
from .results import (
    Assignment,
    ConstraintDiagnostic,
    InterviewerSummary,
    SlotSummary,
    SolveResult,
    SolveStatus,
)


_MODELABLE_FEASIBILITY_ERRORS = {
    "MIN_TOTAL_INFEASIBLE",
    "AGGREGATE_MINIMUM_EXCEEDS_CAPACITY",
    "PRIOR_EXCEEDS_MAX_TOTAL",
    "LOCKS_EXCEED_MAX_TOTAL",
    "LOCKS_EXCEED_MAX_PER_DAY",
    "LOCKS_EXCEED_SLOT_CAPACITY",
    "LOCKS_EXCEED_MAX_CONSECUTIVE",
    "LOCK_OUTSIDE_AVAILABILITY",
}


@dataclass(slots=True)
class _Artifacts:
    x: dict[tuple[str, str], cp_model.IntVar]
    minimum_shortfall: dict[str, cp_model.IntVar]
    maximum_overage: dict[str, cp_model.IntVar]
    daily_overage: dict[tuple[str, date], cp_model.IntVar]
    active_day_shortfall: dict[tuple[str, date], cp_model.IntVar]
    person_target_shortfall: dict[str, cp_model.IntVar]
    person_target_excess: dict[str, cp_model.IntVar]
    slot_target_deficit: dict[str, cp_model.IntVar]
    slot_target_excess: dict[str, cp_model.IntVar]
    group_target_deficit: dict[tuple[str, InterviewerGroup], cp_model.IntVar]
    back_to_back: dict[tuple[str, str, str], cp_model.IntVar]


def _linear_sum(values: Iterable[cp_model.LinearExpr | cp_model.IntVar | int]):
    items = list(values)
    return cp_model.LinearExpr.Sum(items) if items else 0


def _diagnostic_from_validation(issue: ValidationIssue) -> ConstraintDiagnostic:
    return ConstraintDiagnostic(
        severity=issue.severity.value,
        code=issue.code,
        message=issue.message,
        constraint=issue.family.value,
        interviewer_id=issue.context.get("interviewer_id"),
        slot_id=issue.context.get("slot_id"),
        expected=issue.context.get("expected"),
        actual=issue.context.get("actual"),
    )


def _status(value: int, all_stages_optimal: bool) -> SolveStatus:
    if value == cp_model.OPTIMAL and all_stages_optimal:
        return SolveStatus.OPTIMAL
    if value in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveStatus.FEASIBLE
    if value == cp_model.INFEASIBLE:
        return SolveStatus.INFEASIBLE
    if value == cp_model.MODEL_INVALID:
        return SolveStatus.INVALID
    return SolveStatus.UNKNOWN


def solve(
    problem: SchedulingProblem,
    *,
    scenario: str,
    config: SchedulerConfig | None = None,
    relaxation_mode: RelaxationMode = RelaxationMode.STRICT,
) -> SolveResult:
    """Solve one scenario with auditable strict/advisory relaxation modes."""

    started = perf_counter()
    cfg = config or DEFAULT_CONFIG
    relaxation_mode = RelaxationMode(relaxation_mode)
    scenario = str(scenario).strip() or "Scenario"
    validation = validate_problem(problem, cfg)
    fatal_errors = tuple(
        issue
        for issue in validation.errors
        if issue.code not in _MODELABLE_FEASIBILITY_ERRORS
    )
    if fatal_errors:
        return SolveResult(
            status=SolveStatus.INVALID,
            scenario=scenario,
            diagnostics=tuple(_diagnostic_from_validation(i) for i in validation.issues),
            wall_time_seconds=perf_counter() - started,
            message="Inputs contain validation errors that cannot be scheduled.",
            settings=_settings(cfg, relaxation_mode),
        )

    model = cp_model.CpModel()
    people = tuple(problem.interviewers)
    slots = tuple(problem.slots)
    people_by_id = {person.id: person for person in people}
    slots_by_id = {slot.id: slot for slot in slots}
    locked_keys = {
        (lock.interviewer_id, lock.slot_id)
        for lock in problem.locked_assignments
    }

    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for person in people:
        for slot in slots:
            variable = model.NewBoolVar(f"x__{person.id}__{slot.id}")
            x[(person.id, slot.id)] = variable
            if slot.id not in person.available_slot_ids:
                model.Add(variable == 0)
            if (person.id, slot.id) in locked_keys:
                model.Add(variable == 1)

    for slot in slots:
        model.Add(
            _linear_sum(x[(person.id, slot.id)] for person in people)
            <= slot.capacity
        )

    for person in people:
        for left, right in problem.overlaps:
            model.Add(x[(person.id, left)] + x[(person.id, right)] <= 1)

    days: dict[date, list[str]] = defaultdict(list)
    for slot in slots:
        days[slot.local_date].append(slot.id)

    minimum_shortfall: dict[str, cp_model.IntVar] = {}
    maximum_overage: dict[str, cp_model.IntVar] = {}
    daily_overage: dict[tuple[str, date], cp_model.IntVar] = {}
    active_day_shortfall: dict[tuple[str, date], cp_model.IntVar] = {}
    person_target_shortfall: dict[str, cp_model.IntVar] = {}
    person_target_excess: dict[str, cp_model.IntVar] = {}
    violation_terms: list[cp_model.LinearExpr] = []
    person_target_terms: list[cp_model.LinearExpr] = []
    person_excess_terms: list[cp_model.LinearExpr] = []

    relax_minimums = relaxation_mode in {
        RelaxationMode.MINIMUMS,
        RelaxationMode.MINIMUMS_AND_MAXIMUMS,
    }
    relax_maximums = relaxation_mode in {
        RelaxationMode.MAXIMUMS,
        RelaxationMode.MINIMUMS_AND_MAXIMUMS,
    }

    for person in people:
        policy = cfg.policy_for(person.id, person.group)
        total_new = _linear_sum(x[(person.id, slot.id)] for slot in slots)
        cumulative = person.historical_prior_count + total_new
        priority = (
            cfg.student_priority_weight
            if person.group is InterviewerGroup.STUDENT
            else 1
        )

        if relax_minimums:
            shortfall = model.NewIntVar(
                0,
                policy.min_total,
                f"minimum_shortfall__{person.id}",
            )
            model.Add(cumulative + shortfall >= policy.min_total)
            minimum_shortfall[person.id] = shortfall
            violation_terms.append(priority * shortfall)
        else:
            model.Add(cumulative >= policy.min_total)

        if relax_maximums:
            overage = model.NewIntVar(
                0,
                person.historical_prior_count + len(slots),
                f"maximum_overage__{person.id}",
            )
            model.Add(cumulative <= policy.max_total + overage)
            maximum_overage[person.id] = overage
            violation_terms.append(10 * overage)
        else:
            model.Add(cumulative <= policy.max_total)

        target_shortfall = model.NewIntVar(
            0,
            policy.target_total,
            f"target_shortfall__{person.id}",
        )
        model.Add(cumulative + target_shortfall >= policy.target_total)
        person_target_shortfall[person.id] = target_shortfall
        person_target_terms.append(priority * target_shortfall)

        target_excess = model.NewIntVar(
            0,
            person.historical_prior_count + len(slots),
            f"target_excess__{person.id}",
        )
        model.Add(target_excess >= cumulative - policy.target_total)
        person_target_excess[person.id] = target_excess
        person_excess_terms.append(target_excess)

        for assignment_day, slot_ids in days.items():
            daily_total = _linear_sum(
                x[(person.id, slot_id)] for slot_id in slot_ids
            )
            if relax_maximums:
                daily_over = model.NewIntVar(
                    0,
                    len(slot_ids),
                    f"daily_overage__{person.id}__{assignment_day.isoformat()}",
                )
                model.Add(daily_total <= policy.max_per_day + daily_over)
                daily_overage[(person.id, assignment_day)] = daily_over
                violation_terms.append(5 * daily_over)
            else:
                model.Add(daily_total <= policy.max_per_day)

            if policy.min_per_active_day > 0:
                active = model.NewBoolVar(
                    f"active_day__{person.id}__{assignment_day.isoformat()}"
                )
                model.Add(daily_total >= active)
                model.Add(daily_total <= len(slot_ids) * active)
                if relax_minimums:
                    day_short = model.NewIntVar(
                        0,
                        policy.min_per_active_day,
                        f"active_day_shortfall__{person.id}__{assignment_day.isoformat()}",
                    )
                    model.Add(
                        daily_total + day_short
                        >= policy.min_per_active_day * active
                    )
                    active_day_shortfall[(person.id, assignment_day)] = day_short
                    violation_terms.append(priority * day_short)
                else:
                    model.Add(daily_total >= policy.min_per_active_day * active)

    # Minimize the worst target shortfall in each group before minimizing the
    # total shortfall. This prevents deterministic tie-breaking from repeatedly
    # favoring the first names in the roster.
    fairness_terms: list[cp_model.LinearExpr] = []
    for group in InterviewerGroup:
        group_people = [person for person in people if person.group is group]
        if not group_people:
            continue
        upper_bound = max(
            cfg.policy_for(person.id, person.group).target_total
            for person in group_people
        )
        worst = model.NewIntVar(0, upper_bound, f"worst_target_shortfall__{group.value}")
        for person in group_people:
            model.Add(worst >= person_target_shortfall[person.id])
        priority = cfg.student_priority_weight if group is InterviewerGroup.STUDENT else 1
        fairness_terms.append(priority * worst)

    # A run is based on consecutive listed slots, exactly as presented by the
    # authoritative slot source, and never crosses a date boundary.
    for person in people:
        for assignment_day, slot_ids in days.items():
            window_size = cfg.max_consecutive_slots + 1
            for offset in range(0, len(slot_ids) - window_size + 1):
                window = slot_ids[offset : offset + window_size]
                model.Add(
                    _linear_sum(x[(person.id, slot_id)] for slot_id in window)
                    <= cfg.max_consecutive_slots
                )

    back_to_back: dict[tuple[str, str, str], cp_model.IntVar] = {}
    spacing_terms: list[cp_model.LinearExpr] = []
    if cfg.back_to_back is not BackToBackPolicy.OFF:
        for person in people:
            for left, right in problem.adjacency:
                if cfg.back_to_back is BackToBackPolicy.HARD:
                    model.Add(x[(person.id, left)] + x[(person.id, right)] <= 1)
                else:
                    pair = model.NewBoolVar(
                        f"back_to_back__{person.id}__{left}__{right}"
                    )
                    model.Add(pair >= x[(person.id, left)] + x[(person.id, right)] - 1)
                    model.Add(pair <= x[(person.id, left)])
                    model.Add(pair <= x[(person.id, right)])
                    back_to_back[(person.id, left, right)] = pair
                    spacing_terms.append(pair)

    slot_target_deficit: dict[str, cp_model.IntVar] = {}
    slot_target_excess: dict[str, cp_model.IntVar] = {}
    group_target_deficit: dict[tuple[str, InterviewerGroup], cp_model.IntVar] = {}
    coverage_terms: list[cp_model.LinearExpr] = []
    for slot in slots:
        target = slot.target if slot.target is not None else slot.capacity
        assigned = _linear_sum(x[(person.id, slot.id)] for person in people)
        deficit = model.NewIntVar(0, target, f"slot_deficit__{slot.id}")
        excess = model.NewIntVar(0, slot.capacity, f"slot_excess__{slot.id}")
        model.Add(assigned + deficit >= target)
        model.Add(excess >= assigned - target)
        slot_target_deficit[slot.id] = deficit
        slot_target_excess[slot.id] = excess
        coverage_terms.extend((10 * deficit, 10 * excess))

        for group, group_target in slot.group_targets.items():
            group_assigned = _linear_sum(
                x[(person.id, slot.id)]
                for person in people
                if person.group is group
            )
            group_deficit = model.NewIntVar(
                0,
                group_target,
                f"group_deficit__{slot.id}__{group.value}",
            )
            model.Add(group_assigned + group_deficit >= group_target)
            group_target_deficit[(slot.id, group)] = group_deficit
            coverage_terms.append(5 * group_deficit)

    preference_terms: list[cp_model.LinearExpr] = []
    for person in people:
        for slot in slots:
            preference_penalty = 5 - person.preference_for(slot.id)
            if preference_penalty:
                preference_terms.append(
                    preference_penalty * x[(person.id, slot.id)]
                )

    artifacts = _Artifacts(
        x=x,
        minimum_shortfall=minimum_shortfall,
        maximum_overage=maximum_overage,
        daily_overage=daily_overage,
        active_day_shortfall=active_day_shortfall,
        person_target_shortfall=person_target_shortfall,
        person_target_excess=person_target_excess,
        slot_target_deficit=slot_target_deficit,
        slot_target_excess=slot_target_excess,
        group_target_deficit=group_target_deficit,
        back_to_back=back_to_back,
    )

    stages: list[tuple[str, cp_model.LinearExpr]] = []
    if violation_terms:
        stages.append(("relaxation_cost", _linear_sum(violation_terms)))
    if coverage_terms:
        stages.append(("slot_target_deviation", _linear_sum(coverage_terms)))
    if fairness_terms:
        stages.append(("worst_individual_target_shortfall", _linear_sum(fairness_terms)))
    if person_target_terms:
        stages.append(("individual_target_shortfall", _linear_sum(person_target_terms)))
    quality_terms = (
        [100 * item for item in spacing_terms]
        + [10 * item for item in person_excess_terms]
        + preference_terms
    )
    if quality_terms:
        stages.append(("spacing_and_preference", _linear_sum(quality_terms)))

    solver: cp_model.CpSolver | None = None
    best_values: dict[int, int] | None = None
    last_status = cp_model.UNKNOWN
    all_stages_optimal = True
    objective_metrics: dict[str, int | float] = {}
    time_per_stage = max(0.25, cfg.time_limit_seconds / max(1, len(stages)))

    for stage_name, expression in stages:
        model.Minimize(expression)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_per_stage
        solver.parameters.random_seed = cfg.random_seed
        solver.parameters.num_search_workers = cfg.num_search_workers
        last_status = solver.Solve(model)
        if last_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            status = _status(last_status, False)
            if best_values is not None and status is SolveStatus.UNKNOWN:
                objective_metrics["quality_stage_timeout_fallback"] = 1
                return _build_result(
                    problem=problem,
                    scenario=scenario,
                    cfg=cfg,
                    relaxation_mode=relaxation_mode,
                    validation_issues=validation.issues,
                    artifacts=artifacts,
                    value_of=lambda variable: best_values[variable.Index()],
                    status=SolveStatus.FEASIBLE,
                    objective_metrics=objective_metrics,
                    wall_time_seconds=perf_counter() - started,
                    people_by_id=people_by_id,
                    slots_by_id=slots_by_id,
                    locked_keys=locked_keys,
                    fallback_stage=stage_name,
                )
            diagnostics = tuple(
                _diagnostic_from_validation(issue) for issue in validation.issues
            )
            return SolveResult(
                status=status,
                scenario=scenario,
                diagnostics=diagnostics,
                objective_metrics=objective_metrics,
                settings=_settings(cfg, relaxation_mode),
                wall_time_seconds=perf_counter() - started,
                message=(
                    "No schedule satisfies the selected hard constraints. "
                    "Use an advisory relaxation only after reviewing diagnostics."
                    if status is SolveStatus.INFEASIBLE
                    else "The solver did not produce a schedule within the limit."
                ),
            )
        optimum = int(round(solver.ObjectiveValue()))
        objective_metrics[stage_name] = optimum
        all_stages_optimal = all_stages_optimal and last_status == cp_model.OPTIMAL
        best_values = {
            index: int(
                solver.Value(model.GetIntVarFromProtoIndex(index))
            )
            for index in range(len(model.Proto().variables))
        }
        model.Add(expression == optimum)
        model.ClearHints()
        for index, value in best_values.items():
            model.AddHint(model.GetIntVarFromProtoIndex(index), value)

    if solver is None:
        solver = cp_model.CpSolver()
        last_status = solver.Solve(model)

    return _build_result(
        problem=problem,
        scenario=scenario,
        cfg=cfg,
        relaxation_mode=relaxation_mode,
        validation_issues=validation.issues,
        artifacts=artifacts,
        value_of=solver.Value,
        status=_status(last_status, all_stages_optimal),
        objective_metrics=objective_metrics,
        wall_time_seconds=perf_counter() - started,
        people_by_id=people_by_id,
        slots_by_id=slots_by_id,
        locked_keys=locked_keys,
    )


def _settings(
    cfg: SchedulerConfig,
    relaxation_mode: RelaxationMode,
) -> dict[str, int | float | str]:
    return {
        "relaxation_mode": relaxation_mode.value,
        "student_priority_weight": cfg.student_priority_weight,
        "back_to_back_policy": cfg.back_to_back.value,
        "maximum_consecutive_slots": cfg.max_consecutive_slots,
        "time_limit_seconds": cfg.time_limit_seconds,
        "random_seed": cfg.random_seed,
        "search_workers": cfg.num_search_workers,
        "capacity_units_per_assignment": cfg.capacity_units_per_assignment,
    }


def _build_result(
    *,
    problem: SchedulingProblem,
    scenario: str,
    cfg: SchedulerConfig,
    relaxation_mode: RelaxationMode,
    validation_issues: tuple[ValidationIssue, ...],
    artifacts: _Artifacts,
    value_of: Callable[[cp_model.IntVar], int],
    status: SolveStatus,
    objective_metrics: dict[str, int | float],
    wall_time_seconds: float,
    people_by_id: dict,
    slots_by_id: dict,
    locked_keys: set[tuple[str, str]],
    fallback_stage: str | None = None,
) -> SolveResult:
    assignments: list[Assignment] = []
    assigned_keys: set[tuple[str, str]] = set()
    for person in problem.interviewers:
        for slot in problem.slots:
            key = (person.id, slot.id)
            if value_of(artifacts.x[key]) != 1:
                continue
            assigned_keys.add(key)
            assignments.append(
                Assignment(
                    scenario=scenario,
                    slot_id=slot.id,
                    start=slot.start,
                    end=slot.end,
                    interviewer_id=person.id,
                    interviewer_name=person.name,
                    group=person.group,
                    locked=key in locked_keys,
                    preference=person.preference_for(slot.id),
                )
            )
    assignments.sort(key=lambda item: (item.start, item.group.value, item.interviewer_name.casefold()))

    diagnostics = [
        _diagnostic_from_validation(issue)
        for issue in validation_issues
        if issue.severity is Severity.WARNING
    ]
    if status is SolveStatus.FEASIBLE:
        diagnostics.append(
            ConstraintDiagnostic(
                severity="info",
                code="SOLUTION_NOT_PROVEN_OPTIMAL",
                message=(
                    f"The prior feasible schedule was retained when quality stage "
                    f"{fallback_stage!r} reached its time slice."
                    if fallback_stage
                    else "A feasible schedule was returned before optimality was proven."
                ),
                constraint="solver_time_limit",
            )
        )
    adjacency = set(problem.adjacency)
    interviewer_summaries: list[InterviewerSummary] = []
    for person in problem.interviewers:
        policy = cfg.policy_for(person.id, person.group)
        person_slots = {
            slot_id
            for interviewer_id, slot_id in assigned_keys
            if interviewer_id == person.id
        }
        day_counts = Counter(
            slots_by_id[slot_id].local_date for slot_id in person_slots
        )
        new_count = len(person_slots)
        cumulative = person.historical_prior_count + new_count
        minimum_shortfall = max(0, policy.min_total - cumulative)
        target_shortfall = max(0, policy.target_total - cumulative)
        maximum_overage = max(0, cumulative - policy.max_total)
        back_to_back_pairs = sum(
            1
            for left, right in adjacency
            if left in person_slots and right in person_slots
        )
        summary = InterviewerSummary(
            interviewer_id=person.id,
            interviewer_name=person.name,
            group=person.group,
            historical_prior_count=person.historical_prior_count,
            new_assignments=new_count,
            cumulative_total=cumulative,
            minimum=policy.min_total,
            target=policy.target_total,
            maximum=policy.max_total,
            max_per_day=policy.max_per_day,
            minimum_shortfall=minimum_shortfall,
            target_shortfall=target_shortfall,
            maximum_overage=maximum_overage,
            active_days=len(day_counts),
            maximum_assigned_on_day=max(day_counts.values(), default=0),
            back_to_back_pairs=back_to_back_pairs,
        )
        interviewer_summaries.append(summary)
        if minimum_shortfall:
            diagnostics.append(
                ConstraintDiagnostic(
                    severity="warning",
                    code="MINIMUM_RELAXED",
                    message=(
                        f"{person.name} is {minimum_shortfall} assignment(s) below "
                        f"the cumulative minimum of {policy.min_total}."
                    ),
                    constraint="minimum_total",
                    interviewer_id=person.id,
                    interviewer_name=person.name,
                    group=person.group,
                    expected=policy.min_total,
                    actual=cumulative,
                )
            )
        if maximum_overage:
            diagnostics.append(
                ConstraintDiagnostic(
                    severity="warning",
                    code="MAXIMUM_RELAXED",
                    message=(
                        f"{person.name} is {maximum_overage} assignment(s) above "
                        f"the cumulative maximum of {policy.max_total}."
                    ),
                    constraint="maximum_total",
                    interviewer_id=person.id,
                    interviewer_name=person.name,
                    group=person.group,
                    expected=policy.max_total,
                    actual=cumulative,
                )
            )
        for assignment_day, count in day_counts.items():
            if 0 < count < policy.min_per_active_day:
                diagnostics.append(
                    ConstraintDiagnostic(
                        severity="warning",
                        code="ACTIVE_DAY_MINIMUM_RELAXED",
                        message=(
                            f"{person.name} has {count} assignment(s) on "
                            f"{assignment_day.isoformat()}, below the active-day "
                            f"minimum of {policy.min_per_active_day}."
                        ),
                        constraint="minimum_per_active_day",
                        interviewer_id=person.id,
                        interviewer_name=person.name,
                        group=person.group,
                        assignment_date=assignment_day,
                        expected=policy.min_per_active_day,
                        actual=count,
                    )
                )
            if count > policy.max_per_day:
                diagnostics.append(
                    ConstraintDiagnostic(
                        severity="warning",
                        code="DAILY_MAXIMUM_RELAXED",
                        message=(
                            f"{person.name} has {count} assignments on "
                            f"{assignment_day.isoformat()}, above the maximum of "
                            f"{policy.max_per_day}."
                        ),
                        constraint="maximum_per_day",
                        interviewer_id=person.id,
                        interviewer_name=person.name,
                        group=person.group,
                        assignment_date=assignment_day,
                        expected=policy.max_per_day,
                        actual=count,
                    )
                )
        if back_to_back_pairs:
            diagnostics.append(
                ConstraintDiagnostic(
                    severity="info",
                    code="BACK_TO_BACK_ASSIGNED",
                    message=(
                        f"{person.name} has {back_to_back_pairs} discouraged "
                        "back-to-back assignment pair(s)."
                    ),
                    constraint="back_to_back",
                    interviewer_id=person.id,
                    interviewer_name=person.name,
                    group=person.group,
                    expected=0,
                    actual=back_to_back_pairs,
                )
            )

    slot_summaries: list[SlotSummary] = []
    for slot in problem.slots:
        assigned_people = [
            people_by_id[person_id]
            for person_id, slot_id in assigned_keys
            if slot_id == slot.id
        ]
        student_count = sum(
            person.group is InterviewerGroup.STUDENT for person in assigned_people
        )
        adcom_count = sum(
            person.group is InterviewerGroup.ADCOM for person in assigned_people
        )
        assigned_count = len(assigned_people)
        target = slot.target if slot.target is not None else slot.capacity
        target_deficit = max(0, target - assigned_count)
        student_target = slot.group_targets.get(InterviewerGroup.STUDENT)
        adcom_target = slot.group_targets.get(InterviewerGroup.ADCOM)
        slot_summaries.append(
            SlotSummary(
                slot_id=slot.id,
                start=slot.start,
                end=slot.end,
                target=target,
                capacity=slot.capacity,
                assigned=assigned_count,
                target_deficit=target_deficit,
                remaining_capacity=slot.capacity - assigned_count,
                student_assigned=student_count,
                adcom_assigned=adcom_count,
                student_target=student_target,
                adcom_target=adcom_target,
            )
        )
        if target_deficit:
            diagnostics.append(
                ConstraintDiagnostic(
                    severity="warning",
                    code="SLOT_TARGET_UNMET",
                    message=(
                        f"Slot {slot.id} is {target_deficit} assignment(s) below "
                        f"its target of {target}."
                    ),
                    constraint="slot_target",
                    slot_id=slot.id,
                    assignment_date=slot.local_date,
                    expected=target,
                    actual=assigned_count,
                )
            )
        for group, group_target in slot.group_targets.items():
            actual = student_count if group is InterviewerGroup.STUDENT else adcom_count
            if actual < group_target:
                diagnostics.append(
                    ConstraintDiagnostic(
                        severity="warning",
                        code="GROUP_SLOT_TARGET_UNMET",
                        message=(
                            f"Slot {slot.id} is {group_target - actual} "
                            f"{group.label} assignment(s) below its group target."
                        ),
                        constraint="group_slot_target",
                        group=group,
                        slot_id=slot.id,
                        assignment_date=slot.local_date,
                        expected=group_target,
                        actual=actual,
                    )
                )

    message = (
        "Optimal schedule generated."
        if status is SolveStatus.OPTIMAL
        else "A feasible schedule was generated within the configured time limit."
    )
    return SolveResult(
        status=status,
        scenario=scenario,
        assignments=tuple(assignments),
        interviewer_summaries=tuple(interviewer_summaries),
        slot_summaries=tuple(slot_summaries),
        diagnostics=tuple(diagnostics),
        objective_metrics=objective_metrics,
        settings=_settings(cfg, relaxation_mode),
        wall_time_seconds=wall_time_seconds,
        message=message,
    )
