from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone

from interview_scheduler_v2 import (
    CAPACITY_UNITS_PER_ASSIGNMENT,
    DEFAULT_CONFIG,
    BackToBackPolicy,
    GroupPolicy,
    Interviewer,
    InterviewerGroup,
    LockedAssignment,
    SchedulerConfig,
    SchedulingProblem,
    Slot,
    generated_interviewer_id,
    normalize_name,
    validate_problem,
)


UTC = timezone.utc


def slot(slot_id: str, hour: int, *, day: int = 1, capacity: int = 2, **kwargs) -> Slot:
    start = datetime(2026, 9, day, hour, tzinfo=UTC)
    return Slot(
        id=slot_id,
        start=start,
        end=start + timedelta(hours=1),
        capacity=capacity,
        **kwargs,
    )


class IdentifierTests(unittest.TestCase):
    def test_generated_ids_are_normalized_deterministic_group_prefixed_numeric(self):
        first = generated_interviewer_id(InterviewerGroup.STUDENT, "  José   Smith ")
        second = generated_interviewer_id("student", "JOSÉ SMITH")
        self.assertEqual(first, second)
        prefix, digits = first.split("-", 1)
        self.assertEqual(prefix, "STU")
        self.assertTrue(digits.isdigit())
        self.assertNotEqual(first, generated_interviewer_id("adcom", "José Smith"))
        self.assertEqual(normalize_name("  A\t B "), "a b")

    def test_explicit_id_is_preserved(self):
        person = Interviewer.create(
            name="Alex Example", group=InterviewerGroup.ADCOM, explicit_id=" ADC-42 "
        )
        self.assertEqual(person.id, "ADC-42")


class DomainTests(unittest.TestCase):
    def test_required_defaults_and_one_unit_invariant(self):
        student = DEFAULT_CONFIG.group_policies[InterviewerGroup.STUDENT]
        adcom = DEFAULT_CONFIG.group_policies[InterviewerGroup.ADCOM]
        self.assertEqual(student, GroupPolicy(3, 4, 5, 2, 0))
        self.assertEqual(adcom, GroupPolicy(4, 4, 6, 2, 0))
        self.assertEqual(DEFAULT_CONFIG.back_to_back, BackToBackPolicy.DISCOURAGED)
        self.assertEqual(DEFAULT_CONFIG.max_consecutive_slots, 2)
        self.assertEqual(CAPACITY_UNITS_PER_ASSIGNMENT, 1)
        with self.assertRaises(ValueError):
            SchedulerConfig(capacity_units_per_assignment=2)

    def test_models_are_immutable_and_slot_targets_are_immutable(self):
        current = slot(
            "s1", 9, target=2,
            group_targets={InterviewerGroup.STUDENT: 1, InterviewerGroup.ADCOM: 1},
        )
        with self.assertRaises(FrozenInstanceError):
            current.capacity = 3
        with self.assertRaises(TypeError):
            current.group_targets[InterviewerGroup.STUDENT] = 2

    def test_adjacency_uses_consecutive_listed_slots_only_within_day(self):
        problem = SchedulingProblem(
            interviewers=(),
            slots=(slot("late", 15), slot("early", 9), slot("next", 9, day=2)),
        )
        self.assertEqual(problem.adjacency, (("late", "early"),))


class ValidationTests(unittest.TestCase):
    def valid_problem(self) -> SchedulingProblem:
        slots = tuple(
            slot(f"s{i}", 8 + ((i - 1) % 2), day=1 + ((i - 1) // 2))
            for i in range(1, 5)
        )
        student = Interviewer.create(
            name="Student One",
            group=InterviewerGroup.STUDENT,
            available_slot_ids=[item.id for item in slots],
        )
        adcom = Interviewer.create(
            name="Adcom One",
            group=InterviewerGroup.ADCOM,
            available_slot_ids=[item.id for item in slots],
        )
        return SchedulingProblem((student, adcom), slots)

    def test_valid_problem_has_no_errors(self):
        report = validate_problem(self.valid_problem())
        self.assertTrue(report.is_valid, report.issues)

    def test_timezone_capacity_and_target_validation(self):
        naive = Slot(
            id="bad",
            start=datetime(2026, 9, 1, 9),
            end=datetime(2026, 9, 1, 10),
            capacity=1,
            target=2,
            group_targets={InterviewerGroup.STUDENT: 2},
        )
        report = validate_problem(SchedulingProblem((), (naive,)))
        self.assertTrue(report.by_code("NAIVE_SLOT_DATETIME"))
        self.assertTrue(report.by_code("INVALID_SLOT_TARGET"))
        self.assertTrue(report.by_code("INVALID_GROUP_SLOT_TARGET"))

    def test_duplicate_id_and_normalized_name_collisions(self):
        current_slot = slot("s1", 9, capacity=10)
        people = (
            Interviewer("same", " Alex  Doe ", InterviewerGroup.STUDENT, frozenset({"s1"}), 3),
            Interviewer("same", "Different", InterviewerGroup.STUDENT, frozenset({"s1"}), 3),
            Interviewer("third", "alex doe", InterviewerGroup.ADCOM, frozenset({"s1"}), 4),
        )
        report = validate_problem(SchedulingProblem(people, (current_slot,)))
        self.assertTrue(report.by_code("DUPLICATE_INTERVIEWER_ID"))
        self.assertTrue(report.by_code("CROSS_GROUP_NAME_COLLISION"))

    def test_availability_and_minimum_feasibility(self):
        person = Interviewer.create(
            name="No Availability", group=InterviewerGroup.STUDENT,
            available_slot_ids=["missing"], historical_prior_count=0,
        )
        report = validate_problem(SchedulingProblem((person,), (slot("s1", 9),)))
        self.assertTrue(report.by_code("UNKNOWN_AVAILABILITY_SLOT"))
        issue = report.by_code("MIN_TOTAL_INFEASIBLE")[0]
        self.assertTrue(issue.relaxable)
        self.assertEqual(issue.family.value, "total_limit")

    def test_dated_locks_capacity_daily_consecutive_and_b2b(self):
        slots = tuple(slot(f"s{i}", 8 + i, capacity=1) for i in range(1, 4))
        first = Interviewer.create(
            name="Student One", group=InterviewerGroup.STUDENT,
            available_slot_ids=[item.id for item in slots], historical_prior_count=3,
        )
        second = Interviewer.create(
            name="Student Two", group=InterviewerGroup.STUDENT,
            available_slot_ids=[item.id for item in slots], historical_prior_count=3,
        )
        locks = (
            LockedAssignment(first.id, "s1", date(2026, 9, 1)),
            LockedAssignment(first.id, "s2", date(2026, 9, 1)),
            LockedAssignment(first.id, "s3", date(2026, 9, 1)),
            LockedAssignment(second.id, "s1", date(2026, 9, 2)),
        )
        report = validate_problem(SchedulingProblem((first, second), slots, locks))
        self.assertTrue(report.by_code("LOCKS_EXCEED_SLOT_CAPACITY"))
        self.assertTrue(report.by_code("LOCK_DATE_MISMATCH"))
        self.assertTrue(report.by_code("LOCKS_EXCEED_MAX_PER_DAY"))
        self.assertTrue(report.by_code("LOCKS_EXCEED_MAX_CONSECUTIVE"))
        self.assertEqual(len(report.by_code("LOCKED_BACK_TO_BACK")), 1)

    def test_prior_above_hard_max_is_structured_relaxable_error(self):
        person = Interviewer.create(
            name="Over Max", group=InterviewerGroup.STUDENT,
            available_slot_ids=["s1"], historical_prior_count=6,
        )
        report = validate_problem(SchedulingProblem((person,), (slot("s1", 9),)))
        issue = report.by_code("PRIOR_EXCEEDS_MAX_TOTAL")[0]
        self.assertTrue(issue.relaxable)
        self.assertEqual(issue.relaxation_key, "group_policies.student.max_total")


if __name__ == "__main__":
    unittest.main()
