"""Structured validation for v2 inputs and future relaxation workflows."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .config import BackToBackPolicy, DEFAULT_CONFIG, GroupPolicy, SchedulerConfig
from .domain import InterviewerGroup, SchedulingProblem
from .identifiers import normalize_name


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ConstraintFamily(str, Enum):
    IDENTITY = "identity"
    SLOT_TIME = "slot_time"
    CAPACITY = "capacity"
    TARGET = "target"
    AVAILABILITY = "availability"
    TOTAL_LIMIT = "total_limit"
    DAILY_LIMIT = "daily_limit"
    LOCK = "lock"
    CONSECUTIVE = "consecutive"
    CONFIG = "config"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: Severity
    family: ConstraintFamily
    message: str
    path: str
    relaxable: bool = False
    relaxation_key: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", Severity(self.severity))
        object.__setattr__(self, "family", ConstraintFamily(self.family))
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.WARNING)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def by_code(self, code: str) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.code == code)

    def raise_for_errors(self) -> None:
        if self.errors:
            raise InputValidationError(self)


class InputValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(f"Invalid scheduling problem: {len(report.errors)} error(s)")


def _aware(value: Any) -> bool:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _validate_policy(policy: GroupPolicy, path: str, add: Any) -> None:
    values = (
        policy.min_total,
        policy.target_total,
        policy.max_total,
        policy.max_per_day,
        policy.min_per_active_day,
    )
    if any(value < 0 for value in values):
        add("NEGATIVE_GROUP_POLICY", Severity.ERROR, ConstraintFamily.CONFIG,
            "Group policy values cannot be negative", path)
    if not policy.min_total <= policy.target_total <= policy.max_total:
        add("INVALID_TOTAL_POLICY_ORDER", Severity.ERROR, ConstraintFamily.CONFIG,
            "Expected min_total <= target_total <= max_total", path)
    if policy.min_per_active_day > policy.max_per_day:
        add("INVALID_DAILY_POLICY_ORDER", Severity.ERROR, ConstraintFamily.CONFIG,
            "min_per_active_day cannot exceed max_per_day", path)


def validate_problem(
    problem: SchedulingProblem, config: SchedulerConfig | None = None
) -> ValidationReport:
    cfg = config or DEFAULT_CONFIG
    issues: list[ValidationIssue] = []

    def add(
        code: str,
        severity: Severity,
        family: ConstraintFamily,
        message: str,
        path: str,
        *,
        relaxable: bool = False,
        relaxation_key: str | None = None,
        **context: Any,
    ) -> None:
        issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                family=family,
                message=message,
                path=path,
                relaxable=relaxable,
                relaxation_key=relaxation_key,
                context=context,
            )
        )

    for group in InterviewerGroup:
        policy = cfg.group_policies.get(group)
        if policy is None:
            add("MISSING_GROUP_POLICY", Severity.ERROR, ConstraintFamily.CONFIG,
                f"Missing policy for {group.value}", f"config.group_policies.{group.value}")
        else:
            _validate_policy(policy, f"config.group_policies.{group.value}", add)

    if not problem.interviewers:
        add(
            "NO_INTERVIEWERS",
            Severity.ERROR,
            ConstraintFamily.IDENTITY,
            "At least one interviewer is required",
            "interviewers",
        )
    if not problem.slots:
        add(
            "NO_SLOTS",
            Severity.ERROR,
            ConstraintFamily.IDENTITY,
            "At least one authoritative slot is required",
            "slots",
        )

    for interviewer_id, policy in cfg.person_policies.items():
        _validate_policy(
            policy,
            f"config.person_policies.{interviewer_id}",
            add,
        )

    interviewer_ids: dict[str, list[int]] = defaultdict(list)
    normalized_names: dict[str, list[int]] = defaultdict(list)
    possible_new_by_group: Counter[InterviewerGroup] = Counter()
    for index, person in enumerate(problem.interviewers):
        path = f"interviewers[{index}]"
        interviewer_ids[person.id].append(index)
        normalized = normalize_name(person.name)
        normalized_names[normalized].append(index)
        if not person.id:
            add("BLANK_INTERVIEWER_ID", Severity.ERROR, ConstraintFamily.IDENTITY,
                "Interviewer ID cannot be blank", f"{path}.id")
        if not normalized:
            add("BLANK_INTERVIEWER_NAME", Severity.ERROR, ConstraintFamily.IDENTITY,
                "Interviewer name cannot be blank", f"{path}.name")
        if person.historical_prior_count < 0:
            add("NEGATIVE_PRIOR_COUNT", Severity.ERROR, ConstraintFamily.TOTAL_LIMIT,
                "Historical prior count cannot be negative", f"{path}.historical_prior_count")

    for value, indexes in interviewer_ids.items():
        if value and len(indexes) > 1:
            add("DUPLICATE_INTERVIEWER_ID", Severity.ERROR, ConstraintFamily.IDENTITY,
                f"Interviewer ID {value!r} is used more than once", "interviewers",
                indexes=indexes, interviewer_id=value)

    for normalized, indexes in normalized_names.items():
        if not normalized or len(indexes) < 2:
            continue
        groups = {problem.interviewers[index].group for index in indexes}
        code = "CROSS_GROUP_NAME_COLLISION" if len(groups) > 1 else "NORMALIZED_NAME_COLLISION"
        severity = Severity.WARNING if len(groups) > 1 else Severity.ERROR
        add(code, severity, ConstraintFamily.IDENTITY,
            f"Normalized interviewer name {normalized!r} is not unique", "interviewers",
            indexes=indexes, normalized_name=normalized,
            groups=sorted(group.value for group in groups))

    slot_ids: dict[str, list[int]] = defaultdict(list)
    for index, slot in enumerate(problem.slots):
        path = f"slots[{index}]"
        slot_ids[slot.id].append(index)
        if not slot.id:
            add("BLANK_SLOT_ID", Severity.ERROR, ConstraintFamily.IDENTITY,
                "Slot ID cannot be blank", f"{path}.id")
        start_aware, end_aware = _aware(slot.start), _aware(slot.end)
        if not start_aware or not end_aware:
            add("NAIVE_SLOT_DATETIME", Severity.ERROR, ConstraintFamily.SLOT_TIME,
                "Slot start and end must both be timezone-aware", path)
        elif slot.end <= slot.start:
            add("INVALID_SLOT_INTERVAL", Severity.ERROR, ConstraintFamily.SLOT_TIME,
                "Slot end must be after slot start", path)
        if slot.capacity < 0:
            add("NEGATIVE_SLOT_CAPACITY", Severity.ERROR, ConstraintFamily.CAPACITY,
                "Shared slot capacity cannot be negative", f"{path}.capacity")
        if slot.target is not None and not 0 <= slot.target <= slot.capacity:
            add("INVALID_SLOT_TARGET", Severity.ERROR, ConstraintFamily.TARGET,
                "Slot target must be between zero and shared capacity", f"{path}.target",
                relaxable=True, relaxation_key="slot_target")
        for group, target in slot.group_targets.items():
            if target < 0 or target > slot.capacity:
                add("INVALID_GROUP_SLOT_TARGET", Severity.ERROR, ConstraintFamily.TARGET,
                    "Group target must be between zero and shared capacity",
                    f"{path}.group_targets.{group.value}", relaxable=True,
                    relaxation_key="group_slot_target")
        group_target_sum = sum(slot.group_targets.values())
        target_ceiling = slot.target if slot.target is not None else slot.capacity
        if group_target_sum > target_ceiling:
            add("GROUP_TARGETS_EXCEED_SLOT_TARGET", Severity.ERROR, ConstraintFamily.TARGET,
                "Group targets cannot sum above the total slot target/capacity", path,
                relaxable=True, relaxation_key="group_slot_targets",
                group_target_sum=group_target_sum, target_ceiling=target_ceiling)

    for value, indexes in slot_ids.items():
        if value and len(indexes) > 1:
            add("DUPLICATE_SLOT_ID", Severity.ERROR, ConstraintFamily.IDENTITY,
                f"Slot ID {value!r} is used more than once", "slots",
                indexes=indexes, slot_id=value)

    for left, right in problem.overlaps:
        add(
            "OVERLAPPING_SLOT_INTERVALS",
            Severity.WARNING,
            ConstraintFamily.SLOT_TIME,
            "Authoritative slot intervals overlap; an interviewer cannot be assigned to both",
            "slots",
            left_slot_id=left,
            right_slot_id=right,
        )

    known_slots = {slot.id: slot for slot in problem.slots}
    known_people = {person.id: person for person in problem.interviewers}
    for interviewer_id in cfg.person_policies:
        if interviewer_id not in known_people:
            add(
                "UNKNOWN_PERSON_POLICY",
                Severity.ERROR,
                ConstraintFamily.CONFIG,
                "Per-person policy references an unknown interviewer ID",
                f"config.person_policies.{interviewer_id}",
            )

    def policy_for(person: Any) -> GroupPolicy | None:
        if person.id in cfg.person_policies:
            return cfg.person_policies[person.id]
        return cfg.group_policies.get(person.group)

    def policy_key(person: Any, field_name: str) -> str:
        if person.id in cfg.person_policies:
            return f"person_policies.{person.id}.{field_name}"
        return f"group_policies.{person.group.value}.{field_name}"

    for index, person in enumerate(problem.interviewers):
        unknown = sorted(person.available_slot_ids - known_slots.keys())
        if unknown:
            add("UNKNOWN_AVAILABILITY_SLOT", Severity.ERROR, ConstraintFamily.AVAILABILITY,
                "Availability references unknown slots", f"interviewers[{index}].available_slot_ids",
                unknown_slot_ids=unknown)

        unknown_preferences = sorted(person.preference_by_slot.keys() - known_slots.keys())
        if unknown_preferences:
            add(
                "UNKNOWN_PREFERENCE_SLOT",
                Severity.ERROR,
                ConstraintFamily.AVAILABILITY,
                "Preferences reference unknown slots",
                f"interviewers[{index}].preference_by_slot",
                unknown_slot_ids=unknown_preferences,
            )
        outside_availability = sorted(
            person.preference_by_slot.keys() - person.available_slot_ids
        )
        if outside_availability:
            add(
                "PREFERENCE_OUTSIDE_AVAILABILITY",
                Severity.ERROR,
                ConstraintFamily.AVAILABILITY,
                "Preferences may only be set for available slots",
                f"interviewers[{index}].preference_by_slot",
                slot_ids=outside_availability,
            )
        invalid_scores = {
            slot_id: score
            for slot_id, score in person.preference_by_slot.items()
            if score < 1 or score > 5
        }
        if invalid_scores:
            add(
                "INVALID_PREFERENCE_SCORE",
                Severity.ERROR,
                ConstraintFamily.AVAILABILITY,
                "Preference scores must be integers from 1 through 5",
                f"interviewers[{index}].preference_by_slot",
                invalid_scores=invalid_scores,
            )

        policy = policy_for(person)
        if policy is None:
            continue
        if person.historical_prior_count > policy.max_total:
            add("PRIOR_EXCEEDS_MAX_TOTAL", Severity.ERROR, ConstraintFamily.TOTAL_LIMIT,
                "Historical prior count already exceeds the hard maximum total",
                f"interviewers[{index}].historical_prior_count", relaxable=True,
                relaxation_key=policy_key(person, "max_total"))
        day_counts: Counter[Any] = Counter(
            known_slots[slot_id].local_date
            for slot_id in person.available_slot_ids
            if slot_id in known_slots
        )
        daily_bound = sum(min(count, policy.max_per_day) for count in day_counts.values())
        total_bound = max(0, policy.max_total - person.historical_prior_count)
        possible_new = min(daily_bound, total_bound)
        possible_new_by_group[person.group] += possible_new
        required_new = max(0, policy.min_total - person.historical_prior_count)
        if possible_new < required_new:
            add("MIN_TOTAL_INFEASIBLE", Severity.ERROR, ConstraintFamily.TOTAL_LIMIT,
                "Availability and hard limits cannot satisfy this interviewer's minimum total",
                f"interviewers[{index}]", relaxable=True,
                relaxation_key=policy_key(person, "min_total"),
                required_new=required_new, possible_new=possible_new)

    lock_keys: Counter[tuple[str, str]] = Counter()
    locks_by_slot: dict[str, list[Any]] = defaultdict(list)
    locks_by_person_day: dict[tuple[str, Any], set[str]] = defaultdict(set)
    valid_locks_by_person: Counter[str] = Counter()
    for index, lock in enumerate(problem.locked_assignments):
        path = f"locked_assignments[{index}]"
        lock_keys[(lock.interviewer_id, lock.slot_id)] += 1
        person = known_people.get(lock.interviewer_id)
        slot = known_slots.get(lock.slot_id)
        if person is None:
            add("UNKNOWN_LOCKED_INTERVIEWER", Severity.ERROR, ConstraintFamily.LOCK,
                "Locked assignment references an unknown interviewer", f"{path}.interviewer_id")
        if slot is None:
            add("UNKNOWN_LOCKED_SLOT", Severity.ERROR, ConstraintFamily.LOCK,
                "Locked assignment references an unknown slot", f"{path}.slot_id")
        if person is None or slot is None:
            continue
        locks_by_slot[slot.id].append(lock)
        locks_by_person_day[(person.id, slot.local_date)].add(slot.id)
        valid_locks_by_person[person.id] += 1
        if lock.assignment_date != slot.local_date:
            add("LOCK_DATE_MISMATCH", Severity.ERROR, ConstraintFamily.LOCK,
                "Locked assignment date must match the slot's local date",
                f"{path}.assignment_date")
        if slot.id not in person.available_slot_ids:
            add("LOCK_OUTSIDE_AVAILABILITY", Severity.ERROR, ConstraintFamily.AVAILABILITY,
                "Locked assignment is outside interviewer availability", path)

    for key, count in lock_keys.items():
        if count > 1:
            add("DUPLICATE_LOCKED_ASSIGNMENT", Severity.ERROR, ConstraintFamily.LOCK,
                "The same interviewer/slot lock appears more than once", "locked_assignments",
                interviewer_id=key[0], slot_id=key[1], count=count)

    for slot_id, locks in locks_by_slot.items():
        slot = known_slots[slot_id]
        if len(locks) > slot.capacity:
            add("LOCKS_EXCEED_SLOT_CAPACITY", Severity.ERROR, ConstraintFamily.CAPACITY,
                "Locked assignments exceed shared slot capacity", f"slots.{slot_id}",
                locked_count=len(locks), capacity=slot.capacity)
        if slot.target is not None and len(locks) > slot.target:
            add("LOCKS_EXCEED_SLOT_TARGET", Severity.WARNING, ConstraintFamily.TARGET,
                "Locked assignments already exceed the soft slot target", f"slots.{slot_id}")

    for person_id, lock_count in valid_locks_by_person.items():
        person = known_people[person_id]
        policy = policy_for(person)
        if policy and person.historical_prior_count + lock_count > policy.max_total:
            add("LOCKS_EXCEED_MAX_TOTAL", Severity.ERROR, ConstraintFamily.TOTAL_LIMIT,
                "Historical and locked assignments exceed the hard maximum total",
                f"interviewers.{person_id}", relaxable=True,
                relaxation_key=policy_key(person, "max_total"))

    adjacency = set(problem.adjacency)
    overlaps = set(problem.overlaps)
    ordered_slot_ids_by_day: dict[Any, list[str]] = defaultdict(list)
    for slot in problem.slots:
        ordered_slot_ids_by_day[slot.local_date].append(slot.id)

    for (person_id, day), locked_slot_ids in locks_by_person_day.items():
        person = known_people[person_id]
        policy = policy_for(person)
        if policy and len(locked_slot_ids) > policy.max_per_day:
            add("LOCKS_EXCEED_MAX_PER_DAY", Severity.ERROR, ConstraintFamily.DAILY_LIMIT,
                "Locked assignments exceed the hard daily maximum",
                f"interviewers.{person_id}.days.{day.isoformat()}", relaxable=True,
                relaxation_key=policy_key(person, "max_per_day"))

        ordered = ordered_slot_ids_by_day[day]
        run = 0
        longest = 0
        for slot_id in ordered:
            if slot_id in locked_slot_ids:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        if longest > cfg.max_consecutive_slots:
            add("LOCKS_EXCEED_MAX_CONSECUTIVE", Severity.ERROR, ConstraintFamily.CONSECUTIVE,
                "Locked assignments exceed the hard consecutive-slot limit",
                f"interviewers.{person_id}.days.{day.isoformat()}",
                longest_run=longest)

        for left, right in overlaps:
            if left in locked_slot_ids and right in locked_slot_ids:
                add(
                    "LOCKED_ASSIGNMENTS_OVERLAP",
                    Severity.ERROR,
                    ConstraintFamily.SLOT_TIME,
                    "The same interviewer has locked assignments in overlapping slots",
                    f"interviewers.{person_id}.days.{day.isoformat()}",
                    left_slot_id=left,
                    right_slot_id=right,
                )

        if cfg.back_to_back is BackToBackPolicy.DISCOURAGED:
            count = sum(
                1 for left, right in adjacency
                if left in locked_slot_ids and right in locked_slot_ids
            )
            if count:
                add("LOCKED_BACK_TO_BACK", Severity.WARNING, ConstraintFamily.CONSECUTIVE,
                    "Locked assignments contain discouraged back-to-back slots",
                    f"interviewers.{person_id}.days.{day.isoformat()}",
                    back_to_back_count=count)

    required_new_total = 0
    for person in problem.interviewers:
        policy = policy_for(person)
        if policy:
            required_new_total += max(0, policy.min_total - person.historical_prior_count)
    total_capacity = sum(max(0, slot.capacity) for slot in problem.slots)
    if required_new_total > total_capacity:
        add("AGGREGATE_MINIMUM_EXCEEDS_CAPACITY", Severity.ERROR, ConstraintFamily.CAPACITY,
            "Aggregate hard minimum demand exceeds shared capacity", "problem",
            relaxable=True, relaxation_key="minimum_totals_or_capacity",
            required_new_total=required_new_total, total_capacity=total_capacity)

    total_target_demand = sum(
        slot.target if slot.target is not None else slot.capacity
        for slot in problem.slots
    )
    assignable_upper_bound = sum(possible_new_by_group.values())
    if total_target_demand > assignable_upper_bound:
        add(
            "TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM",
            Severity.WARNING,
            ConstraintFamily.TARGET,
            f"Slot target demand {total_target_demand} exceeds the aggregate assignment upper bound {assignable_upper_bound}; at least {total_target_demand - assignable_upper_bound} target assignment(s) are unattainable under current availability and cumulative/daily maximums",
            "problem",
            target_demand=total_target_demand,
            assignable_upper_bound=assignable_upper_bound,
            unavoidable_target_deficit=total_target_demand - assignable_upper_bound,
        )

    for group in InterviewerGroup:
        group_target_demand = sum(
            slot.group_targets.get(group, 0) for slot in problem.slots
        )
        group_upper_bound = possible_new_by_group[group]
        if group_target_demand > group_upper_bound:
            add(
                "GROUP_TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM",
                Severity.WARNING,
                ConstraintFamily.TARGET,
                f"{group.label} slot target demand {group_target_demand} exceeds the group's aggregate assignment upper bound {group_upper_bound}",
                "problem",
                group=group.value,
                target_demand=group_target_demand,
                assignable_upper_bound=group_upper_bound,
                unavoidable_target_deficit=group_target_demand - group_upper_bound,
            )

    return ValidationReport(tuple(issues))
