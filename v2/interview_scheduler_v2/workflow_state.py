"""Workflow state transitions, independent of Streamlit and widget rendering."""

from collections.abc import MutableMapping
from typing import Any


OUTPUT_KEYS = (
    "v2_problem", "v2_config", "v2_validation", "v2_result",
    "v2_report", "v2_simplified_report", "v2_download_reviewed",
    "v2_exception_confirmed",
)


def clear_schedule_outputs(state: MutableMapping[str, Any]) -> None:
    for key in OUTPUT_KEYS:
        state.pop(key, None)


def invalidate_result(state: MutableMapping[str, Any], *, reason: str) -> bool:
    had_result = state.get("v2_result") is not None
    clear_schedule_outputs(state)
    if had_result:
        state["v2_stale_result_notice"] = reason
    return had_result


def invalidate_import(state: MutableMapping[str, Any]) -> None:
    """A changed source/year requires a fresh import and fresh review tables."""

    had_import = state.get("v2_import") is not None
    clear_schedule_outputs(state)
    for key in ("v2_import", "v2_people", "v2_slots", "v2_source_hashes"):
        state.pop(key, None)
    state.pop("v2_stale_result_notice", None)
    if had_import:
        state["v2_source_notice"] = (
            "The availability files or interview year changed. Check the files "
            "again, then review the interview counts and create a new schedule."
        )


def reset_review(state: MutableMapping[str, Any]) -> None:
    """Give each successful import new widget identities so edits cannot replay."""

    clear_schedule_outputs(state)
    state["v2_review_revision"] = state.get("v2_review_revision", 0) + 1
    state["v2_slot_editor_revision"] = 0
    state["v2_period_upload_revision"] = 0
    state.pop("v2_stale_result_notice", None)
    state.pop("v2_source_notice", None)
