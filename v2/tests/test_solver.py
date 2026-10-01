from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from interview_scheduler_v2 import (
    BackToBackPolicy,
    GroupPolicy,
    Interviewer,
    InterviewerGroup,
    LockedAssignment,
    RelaxationMode,
    SchedulerConfig,
    SchedulingProblem,
    Slot,
)
from interview_scheduler_v2.optimization import SolveStatus, solve


ET = ZoneInfo("America/New_York")


def make_slot(
    slot_id: str,
    hour: int,
    *,
    day: int = 1,
    capacity: int = 1,
    target: int | None = None,
    group_targets=None,
) -> Slot:
    start = datetime(2026, 3, day, hour, tzinfo=ET)
    return Slot(
        slot_id,
        start,
        start + timedelta(minutes=90),
        capacity,
        capacity if target is None else target,
        group_targets or {},
    )


def config(
    *,
    student: GroupPolicy = GroupPolicy(0, 1, 3, 2),
    adcom: GroupPolicy = GroupPolicy(0, 1, 3, 2),
    student_weight: int = 3,
) -> SchedulerConfig:
    return SchedulerConfig(
        group_policies={
            InterviewerGroup.STUDENT: student,
            InterviewerGroup.ADCOM: adcom,
        },
        student_priority_weight=student_weight,
        time_limit_seconds=5,
        num_search_workers=1,
    )


def test_one_assignment_consumes_shared_capacity_and_student_weight_breaks_tie():
    slot = make_slot("s1", 8)
    student = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1"],
    )
    adcom = Interviewer.create(
        name="Adcom",
        group=InterviewerGroup.ADCOM,
        available_slot_ids=["s1"],
    )
    result = solve(
        SchedulingProblem((student, adcom), (slot,)),
        scenario="priority",
        config=config(student_weight=5),
    )
    assert result.succeeded
    assert len(result.assignments) == 1
    assert result.assignments[0].group is InterviewerGroup.STUDENT


def test_historical_count_and_dated_lock_count_toward_total_and_daily_limits():
    slots = (make_slot("s1", 8), make_slot("s2", 10))
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1", "s2"],
        historical_prior_count=2,
    )
    lock = LockedAssignment(person.id, "s2", slots[1].local_date)
    result = solve(
        SchedulingProblem((person,), slots, (lock,)),
        scenario="locks",
        config=config(student=GroupPolicy(3, 3, 3, 1)),
    )
    assert result.succeeded
    assert [(item.slot_id, item.locked) for item in result.assignments] == [("s2", True)]
    assert result.interviewer_summaries[0].cumulative_total == 3


def test_relaxed_result_names_person_below_minimum():
    slot = make_slot("s1", 8, capacity=0, target=0)
    person = Interviewer.create(
        name="Unavailable Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=[],
    )
    problem = SchedulingProblem((person,), (slot,))
    policy = config(student=GroupPolicy(1, 1, 2, 1))
    strict = solve(problem, scenario="strict", config=policy)
    assert strict.status is SolveStatus.INFEASIBLE

    relaxed = solve(
        problem,
        scenario="relaxed",
        config=policy,
        relaxation_mode=RelaxationMode.MINIMUMS,
    )
    assert relaxed.succeeded
    assert relaxed.minimum_shortfalls[0].interviewer_name == "Unavailable Student"
    assert any(item.code == "MINIMUM_RELAXED" for item in relaxed.diagnostics)


def test_maximum_can_be_relaxed_independently_for_historical_overage():
    slot = make_slot("s1", 8, capacity=0, target=0)
    person = Interviewer.create(
        name="Prior Overage",
        group=InterviewerGroup.ADCOM,
        available_slot_ids=[],
        historical_prior_count=2,
    )
    problem = SchedulingProblem((person,), (slot,))
    policy = config(adcom=GroupPolicy(0, 0, 1, 1))
    assert solve(problem, scenario="strict", config=policy).status is SolveStatus.INFEASIBLE

    relaxed = solve(
        problem,
        scenario="maximum advisory",
        config=policy,
        relaxation_mode=RelaxationMode.MAXIMUMS,
    )
    assert relaxed.succeeded
    assert relaxed.interviewer_summaries[0].maximum_overage == 1
    assert any(item.code == "MAXIMUM_RELAXED" for item in relaxed.diagnostics)


def test_maximum_consecutive_two_is_hard_while_back_to_back_is_discouraged():
    slots = tuple(make_slot(f"s{index}", 8 + index * 2) for index in range(3))
    first = Interviewer.create(
        name="First",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=[slot.id for slot in slots],
    )
    second = Interviewer.create(
        name="Second",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=[slot.id for slot in slots],
    )
    result = solve(
        SchedulingProblem((first, second), slots),
        scenario="spacing",
        config=config(student=GroupPolicy(0, 2, 3, 3)),
    )
    assert result.succeeded
    assigned_by_person = {}
    for assignment in result.assignments:
        assigned_by_person.setdefault(assignment.interviewer_id, []).append(assignment.slot_id)
    assert all(len(slot_ids) <= 2 for slot_ids in assigned_by_person.values())
    assert sum(item.back_to_back_pairs for item in result.interviewer_summaries) == 0


def test_optional_group_targets_share_one_capacity_pool():
    slot = make_slot(
        "s1",
        8,
        capacity=2,
        target=2,
        group_targets={
            InterviewerGroup.STUDENT: 1,
            InterviewerGroup.ADCOM: 1,
        },
    )
    student = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1"],
    )
    adcom = Interviewer.create(
        name="Adcom",
        group=InterviewerGroup.ADCOM,
        available_slot_ids=["s1"],
    )
    result = solve(
        SchedulingProblem((student, adcom), (slot,)),
        scenario="targets",
        config=config(),
    )
    assert result.succeeded
    assert result.slot_summaries[0].student_assigned == 1
    assert result.slot_summaries[0].adcom_assigned == 1


def test_one_interviewer_cannot_be_double_booked_in_overlapping_slots():
    slots = (make_slot("s1", 8), make_slot("s2", 9))
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1", "s2"],
    )
    result = solve(
        SchedulingProblem((person,), slots),
        scenario="overlap",
        config=config(student=GroupPolicy(0, 2, 2, 2)),
    )
    assert result.succeeded
    assert len(result.assignments) == 1
    assert sum(item.target_deficit for item in result.slot_summaries) == 1


def test_seeded_single_worker_runs_are_reproducible():
    slots = (make_slot("s1", 8), make_slot("s2", 10))
    people = tuple(
        Interviewer.create(
            name=f"Student {index}",
            group=InterviewerGroup.STUDENT,
            available_slot_ids=[slot.id for slot in slots],
        )
        for index in range(3)
    )
    problem = SchedulingProblem(people, slots)
    first = solve(problem, scenario="stable", config=config())
    second = solve(problem, scenario="stable", config=config())
    assert [
        (item.interviewer_id, item.slot_id) for item in first.assignments
    ] == [
        (item.interviewer_id, item.slot_id) for item in second.assignments
    ]


def test_lock_outside_availability_is_invalid_in_every_relaxation_mode():
    slot = make_slot("s1", 8)
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=[],
    )
    lock = LockedAssignment(person.id, "s1", slot.local_date)
    problem = SchedulingProblem((person,), (slot,), (lock,))
    for mode in RelaxationMode:
        result = solve(
            problem,
            scenario="impossible lock",
            config=config(),
            relaxation_mode=mode,
        )
        assert result.status is SolveStatus.INVALID
        assert any(
            item.code == "LOCK_OUTSIDE_AVAILABILITY"
            for item in result.diagnostics
        )


def _two_adjacent_slots() -> tuple[Slot, Slot]:
    return (make_slot("s1", 8), make_slot("s2", 10))


def test_hard_back_to_back_policy_forbids_adjacent_assignments():
    slots = _two_adjacent_slots()
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1", "s2"],
    )
    hard = SchedulerConfig(
        group_policies={
            InterviewerGroup.STUDENT: GroupPolicy(0, 2, 2, 2),
            InterviewerGroup.ADCOM: GroupPolicy(0, 1, 3, 2),
        },
        back_to_back=BackToBackPolicy.HARD,
        time_limit_seconds=5,
    )
    result = solve(SchedulingProblem((person,), slots), scenario="hard", config=hard)
    assert result.succeeded
    assert len(result.assignments) == 1


def test_off_back_to_back_policy_allows_adjacent_assignments():
    slots = _two_adjacent_slots()
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1", "s2"],
    )
    off = SchedulerConfig(
        group_policies={
            InterviewerGroup.STUDENT: GroupPolicy(0, 2, 2, 2),
            InterviewerGroup.ADCOM: GroupPolicy(0, 1, 3, 2),
        },
        back_to_back=BackToBackPolicy.OFF,
        time_limit_seconds=5,
    )
    result = solve(SchedulingProblem((person,), slots), scenario="off", config=off)
    assert result.succeeded
    assert len(result.assignments) == 2


def test_person_policy_override_caps_one_person_below_the_group_default():
    slots = _two_adjacent_slots()
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1", "s2"],
    )
    overridden = SchedulerConfig(
        group_policies={
            InterviewerGroup.STUDENT: GroupPolicy(0, 2, 2, 2),
            InterviewerGroup.ADCOM: GroupPolicy(0, 1, 3, 2),
        },
        person_policies={person.id: GroupPolicy(0, 1, 1, 1)},
        back_to_back=BackToBackPolicy.OFF,
        time_limit_seconds=5,
    )
    result = solve(
        SchedulingProblem((person,), slots),
        scenario="override",
        config=overridden,
    )
    assert result.succeeded
    assert len(result.assignments) == 1


def test_min_per_active_day_forces_all_or_nothing_on_a_day():
    slots = _two_adjacent_slots()
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1", "s2"],
    )
    result = solve(
        SchedulingProblem((person,), slots),
        scenario="active day",
        config=config(student=GroupPolicy(0, 1, 2, 2, min_per_active_day=2)),
    )
    assert result.succeeded
    assert len(result.assignments) in (0, 2)


def test_objective_weights_are_recorded_in_run_settings():
    slot = make_slot("s1", 8)
    person = Interviewer.create(
        name="Student",
        group=InterviewerGroup.STUDENT,
        available_slot_ids=["s1"],
    )
    result = solve(
        SchedulingProblem((person,), (slot,)),
        scenario="weights",
        config=config(),
    )
    assert result.settings["back_to_back_penalty"] == 100
    assert result.settings["over_target_penalty"] == 10
    assert result.settings["slot_target_deviation_weight"] == 10
    assert result.settings["group_target_deficit_weight"] == 5
    assert result.settings["maximum_overage_weight"] == 10
    assert result.settings["daily_overage_weight"] == 5
