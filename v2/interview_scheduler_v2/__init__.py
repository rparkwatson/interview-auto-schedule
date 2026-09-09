"""Independent v2 interview-scheduler foundation."""

from .config import (
    DEFAULT_CONFIG,
    BackToBackPolicy,
    GroupPolicy,
    RelaxationMode,
    SchedulerConfig,
    default_group_policies,
)
from .domain import (
    CAPACITY_UNITS_PER_ASSIGNMENT,
    Interviewer,
    InterviewerGroup,
    LockedAssignment,
    SchedulingProblem,
    Slot,
    consecutive_listed_slot_pairs,
    overlapping_slot_pairs,
)
from .identifiers import generated_interviewer_id, interviewer_id, normalize_name
from .validation import (
    ConstraintFamily,
    InputValidationError,
    Severity,
    ValidationIssue,
    ValidationReport,
    validate_problem,
)

__all__ = [
    "BackToBackPolicy",
    "CAPACITY_UNITS_PER_ASSIGNMENT",
    "ConstraintFamily",
    "DEFAULT_CONFIG",
    "GroupPolicy",
    "InputValidationError",
    "Interviewer",
    "InterviewerGroup",
    "LockedAssignment",
    "RelaxationMode",
    "SchedulerConfig",
    "SchedulingProblem",
    "Severity",
    "Slot",
    "ValidationIssue",
    "ValidationReport",
    "consecutive_listed_slot_pairs",
    "default_group_policies",
    "generated_interviewer_id",
    "interviewer_id",
    "normalize_name",
    "overlapping_slot_pairs",
    "validate_problem",
]
