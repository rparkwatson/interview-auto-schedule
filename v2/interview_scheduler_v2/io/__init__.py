"""Input and output adapters for Interview Scheduler v2."""

from .campaign import (
    CampaignImportResult,
    import_campaign,
    prepare_campaign_from_availability,
)
from .period_template import (
    PeriodTemplateError,
    build_interview_period_template,
    parse_completed_interview_period_template,
    period_fingerprint,
    period_template_filename,
)
from .source_parsers import (
    ImportNotice,
    ParsedAvailability,
    ParsedAvailabilityEntry,
    ParsedSlot,
    ParsedSlotSet,
    ParsedStudentSchedule,
    parse_adcom_availability,
    parse_student_availability,
    parse_student_schedule,
    parse_time_slot_workbook,
    reconcile_availability,
)

__all__ = [
    "CampaignImportResult",
    "ImportNotice",
    "ParsedAvailability",
    "ParsedAvailabilityEntry",
    "ParsedSlot",
    "ParsedSlotSet",
    "ParsedStudentSchedule",
    "PeriodTemplateError",
    "build_interview_period_template",
    "import_campaign",
    "parse_completed_interview_period_template",
    "parse_adcom_availability",
    "parse_student_availability",
    "parse_student_schedule",
    "parse_time_slot_workbook",
    "period_fingerprint",
    "period_template_filename",
    "prepare_campaign_from_availability",
    "reconcile_availability",
]
