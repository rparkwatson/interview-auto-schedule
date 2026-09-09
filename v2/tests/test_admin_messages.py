from datetime import datetime
from zoneinfo import ZoneInfo

from interview_scheduler_v2.io.source_parsers import ImportNotice
from interview_scheduler_v2.optimization import ConstraintDiagnostic
from interview_scheduler_v2.presentation import (
    AdminMessageLevel,
    format_interview_period,
    present_diagnostics,
    present_import_notices,
    present_validation_issues,
)
from interview_scheduler_v2.validation import (
    ConstraintFamily,
    Severity,
    ValidationIssue,
)


def test_file_reconciliation_notices_are_grouped_in_plain_language():
    notices = tuple(
        ImportNotice(
            "info",
            "slot_time_reconciled",
            f"Reconciled source time {index}.",
            f"Sheet {index}",
        )
        for index in range(8)
    )

    messages = present_import_notices(notices)

    assert len(messages) == 1
    assert messages[0].affected_count == 8
    assert messages[0].level is AdminMessageLevel.ADJUSTMENT
    assert messages[0].title == "Availability times were matched automatically"
    assert "8 availability times" in messages[0].summary
    assert "slot_time_reconciled" not in messages[0].summary
    assert messages[0].technical_codes == ("slot_time_reconciled",)


def test_target_demand_warning_explains_requested_possible_and_unfilled_counts():
    issue = ValidationIssue(
        code="TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM",
        severity=Severity.WARNING,
        family=ConstraintFamily.TARGET,
        message="Technical target-demand warning.",
        path="problem",
        context={
            "target_demand": 150,
            "assignable_upper_bound": 142,
            "unavoidable_target_deficit": 8,
        },
    )

    message = present_validation_issues((issue,))[0]

    assert message.title == "Not all requested interview seats can be filled"
    assert "requested 150" in message.summary
    assert "allow up to 142" in message.summary
    assert "at least 8 seats" in message.summary


def test_relaxed_minimum_diagnostics_identify_the_size_of_the_exception():
    diagnostics = (
        ConstraintDiagnostic(
            severity="warning",
            code="MINIMUM_RELAXED",
            message="Alex is below minimum.",
            constraint="minimum_total",
            interviewer_name="Alex",
            expected=3,
            actual=2,
        ),
        ConstraintDiagnostic(
            severity="warning",
            code="MINIMUM_RELAXED",
            message="Blair is below minimum.",
            constraint="minimum_total",
            interviewer_name="Blair",
            expected=4,
            actual=2,
        ),
    )

    message = present_diagnostics(diagnostics)[0]

    assert message.level is AdminMessageLevel.REVIEW
    assert message.affected_count == 2
    assert "2 interviewers" in message.summary
    assert "3 assignments in total" in message.summary
    assert message.details == (
        "Alex is below minimum.",
        "Blair is below minimum.",
    )


def test_result_target_warning_keeps_technical_copy_out_of_the_summary():
    diagnostic = ConstraintDiagnostic(
        severity="warning",
        code="TARGET_DEMAND_EXCEEDS_ASSIGNABLE_MAXIMUM",
        message=(
            "Slot target demand 240 exceeds the aggregate assignment upper bound "
            "142; at least 98 target assignment(s) are unattainable under current "
            "availability and cumulative/daily maximums"
        ),
        constraint="target",
    )

    message = present_diagnostics((diagnostic,))[0]

    assert message.title == "Not all requested interview seats can be filled"
    assert "requested 240" in message.summary
    assert "allow up to 142" in message.summary
    assert "98 seats" in message.summary
    assert "aggregate assignment upper bound" not in message.summary
    assert message.details == (diagnostic.message,)


def test_standard_failure_explains_unreachable_minimum_without_solver_terms():
    diagnostic = ConstraintDiagnostic(
        severity="error",
        code="MIN_TOTAL_INFEASIBLE",
        message=(
            "Availability and hard limits cannot satisfy this interviewer's "
            "minimum total"
        ),
        constraint="total_limit",
    )

    message = present_diagnostics((diagnostic,))[0]

    assert message.title == "Some interviewers cannot reach their required minimum"
    assert "Current availability and limits" in message.summary
    assert "hard limits" not in message.summary
    assert message.level is AdminMessageLevel.BLOCKING


def test_interview_period_uses_an_administrative_date_and_time_label():
    eastern = ZoneInfo("America/New_York")
    start = datetime(2026, 2, 24, 8, 0, tzinfo=eastern)
    end = datetime(2026, 2, 24, 9, 30, tzinfo=eastern)

    assert format_interview_period(start, end) == "Tue, Feb 24 · 8:00 AM–9:30 AM"
