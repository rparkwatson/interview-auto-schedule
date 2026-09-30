from interview_scheduler_v2.workflow_state import (
    OUTPUT_KEYS, invalidate_import, invalidate_result, reset_review,
)


def test_import_change_clears_all_dependent_state_and_explains_next_step():
    state = dict.fromkeys(OUTPUT_KEYS, "previous run")
    state.update(v2_import="old import", v2_people="people", v2_slots="slots",
                 v2_source_hashes={"hash": "old"}, student_file="new upload")
    invalidate_import(state)
    assert not (set(OUTPUT_KEYS) & state.keys())
    assert not ({"v2_import", "v2_people", "v2_slots", "v2_source_hashes"} & state.keys())
    assert state["student_file"] == "new upload"
    assert "Check the files again" in state["v2_source_notice"]


def test_rule_edit_clears_exception_acknowledgments_but_keeps_review():
    state = dict.fromkeys(OUTPUT_KEYS, "previous run")
    state["v2_people"] = "review"
    assert invalidate_result(state, reason="Rules changed.")
    assert not (set(OUTPUT_KEYS) & state.keys())
    assert state["v2_people"] == "review"
    assert state["v2_stale_result_notice"] == "Rules changed."


def test_new_import_uses_fresh_editor_identifiers():
    state = {"v2_review_revision": 7, "v2_slot_editor_revision": 4,
             "v2_source_notice": "changed", "v2_stale_result_notice": "stale"}
    reset_review(state)
    assert state["v2_review_revision"] == 8
    assert state["v2_slot_editor_revision"] == 0
    assert "v2_source_notice" not in state
    assert "v2_stale_result_notice" not in state
