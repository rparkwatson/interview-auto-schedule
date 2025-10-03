from __future__ import annotations
import json
from datetime import datetime, timedelta
import itertools
import io
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Domain models (minimal stubs shown here; replace with your actual imports)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Slot:
    id: str
    start: datetime
    end: datetime
    day_key: str
    adjacent_forward: frozenset[str]

@dataclass(frozen=True)
class Interviewer:
    id: str
    name: str
    kind: str  # "Regular" | "Senior"

@dataclass
class Inputs:
    interviewers: list[Interviewer]
    slots: list[Slot]
    max_pairs_per_slot: dict[str, int]

@dataclass
class Settings:
    time_limit_s: float = 120.0
    threads: int = 0
    back_to_back_mode: str = "soft"  # "soft" | "hard" | "off"
    observer_extra_per_slot: int = 0
    w_pairs: int = 1_000_000
    w_fill: int = 1_000
    w_b2b: int = 1
    adjacency_grace_min: int = 0
    scarcity_bonus: int = 0
    w_fill_adcom: int = 0
    day_caps: dict[str, int] | None = None

# ----------------------------------------------------------------------------
# External hooks (replace with your real project functions)
# ----------------------------------------------------------------------------

def read_inputs_from_legacy(workbook, *, year: int, slot_minutes: int, defaults: dict) -> Inputs:
    """Placeholder legacy parser: swap with the real importer."""
    # In real code, parse workbook and construct domain objects.
    now = datetime(year, 10, 1, 9, 0)
    slots = []
    for d in range(3):
        for h in range(6):
            s = now + timedelta(days=d, minutes=h*slot_minutes)
            e = s + timedelta(minutes=slot_minutes)
            slots.append(Slot(
                id=f"{d}-{h}", start=s, end=e, day_key=s.strftime("%Y-%m-%d"), adjacent_forward=frozenset()
            ))
    ints = [Interviewer(id=f"r{i}", name=f"Reg {i}", kind="Regular") for i in range(20)]
    ints += [Interviewer(id=f"s{i}", name=f"Sen {i}", kind="Senior") for i in range(5)]
    cap = {sl.id: 10 for sl in slots}
    return Inputs(interviewers=ints, slots=slots, max_pairs_per_slot=cap)


def build_adjacency(slots: Iterable[Slot], grace_min: int) -> dict[str, tuple[str, ...]]:
    """Placeholder adjacency builder."""
    return {s.id: tuple() for s in slots}


def greedy_seed(inputs: Inputs):
    return {}


def solve_weighted(inputs: Inputs, cfg: Settings, *, hint=None) -> dict:
    """Placeholder CP-SAT result; replace with your solver."""
    # Create a fake optimal solution using up to capacity
    slot_ids = [s.id for s in inputs.slots]
    pairs = {t: min(2, inputs.max_pairs_per_slot.get(t, 0)) for t in slot_ids}
    adcom = {t: min(1, max(0, inputs.max_pairs_per_slot.get(t, 0) - pairs[t])) for t in slot_ids}
    return {
        "status": "OPTIMAL",
        "assign": {},  # not used by the dashboard when pairs/adcom_singles present
        "pairs": pairs,
        "adcom_singles": adcom,
        "objective": 1234,
    }


def make_excel_report(inputs: Inputs, assign: dict, *, path: str) -> str:
    with open(path, "wb") as f:
        f.write(b"EXCEL_PLACEHOLDER")
    return path

# ---------------------------
# Utilities
# ---------------------------

def _sanity_check_slot_durations(inputs_obj, expected_minutes: int):
    """Warn if any slots have a duration different from expected minutes."""
    try:
        exp = pd.Timedelta(minutes=int(expected_minutes))
        mismatches = [s for s in getattr(inputs_obj, "slots", []) if (s.end - s.start) != exp]
        if mismatches:
            st.warning(f"{len(mismatches)} slot(s) differ from expected {int(expected_minutes)}-minute duration. Check workbook/reader.")
    except Exception:
        # Non-fatal; only a UI warning
        pass


def _mark_dirty():
    """Mark results as stale due to a setting change."""
    st.session_state["needs_rerun"] = True


def _on_upload_change():
    """When a new file is uploaded, mark stale and clear last results + slider defaults."""
    st.session_state["needs_rerun"] = True
    st.session_state.pop("last_results", None)
    # Reset slider ranges so wide defaults reapply for the new workbook
    for k in [
        "reg_max_daily_range","reg_max_total_range","reg_min_total_range",
        "sen_max_daily_range","sen_max_total_range","sen_min_total_range",
    ]:
        st.session_state.pop(k, None)
    # Optional: also clear run history on new upload
    # st.session_state.pop("run_history", None)


def _build_range(lo: int, hi: int, step: int) -> list[int]:
    """Inclusive integer range with guards."""
    lo = int(lo); hi = int(hi); step = max(1, int(step))
    if hi < lo:
        lo, hi = hi, lo
    return list(range(lo, hi + 1, step))


def _init_range_state(key: str, lo: int, hi: int, center: int, *, width: int, step: int) -> None:
    """Initialize a slider range around a center value only once."""
    if key in st.session_state:
        return
    w = int(width)
    lo2 = max(lo, center - w//2)
    hi2 = min(hi, center + w//2)
    # Snap to step grid
    def snap(x):
        base = lo
        k = round((x - base)/step)
        return int(base + k*step)
    st.session_state[key] = (snap(lo2), snap(hi2))


def _compute_rooms_metrics(inputs_local: Inputs, res_local: dict, assign_local: dict) -> tuple[int, int, int]:
    """Compute (rooms_filled, reg_pairs, capacity) from result/assignments."""
    slot_ids_local = [s.id for s in inputs_local.slots]
    cap_map_local = inputs_local.max_pairs_per_slot

    pairs_local = res_local.get("pairs", {})
    adcom_local = res_local.get("adcom_singles")

    if pairs_local and adcom_local is not None:
        rooms_filled = int(sum(int(pairs_local.get(t, 0)) + int(adcom_local.get(t, 0)) for t in slot_ids_local))
        reg_pairs = int(sum(int(pairs_local.get(t, 0)) for t in slot_ids_local))
    else:
        # fallback from raw assign
        iv_by_id_local = {iv.id: iv for iv in inputs_local.interviewers}
        reg_people_by_t_local = {t: 0 for t in slot_ids_local}
        adcom_people_by_t_local = {t: 0 for t in slot_ids_local}
        for (i, t), v in assign_local.items():
            if not v or t not in cap_map_local or i not in iv_by_id_local:
                continue
            kind = iv_by_id_local[i].kind
            if kind == "Regular":
                reg_people_by_t_local[t] += 1
            elif kind == "Senior":
                adcom_people_by_t_local[t] += 1
        reg_pairs = sum(reg_people_by_t_local[t] // 2 for t in slot_ids_local)
        rooms_filled = int(reg_pairs + sum(adcom_people_by_t_local[t] for t in slot_ids_local))

    capacity = int(sum(int(cap_map_local.get(t, 0)) for t in slot_ids_local))
    return rooms_filled, reg_pairs, capacity


def _arrow_safe_scan_df(df: pd.DataFrame, max_rows: int | None = 2000) -> pd.DataFrame:
    """
    Prepare the auto-scan DataFrame for st.dataframe() to avoid Arrow overflows and casting errors.
    - Coerces numeric columns
    - Uses Int64 only when values are integer-like; otherwise keeps float64
    - Replaces ±inf with NaN
    - Optionally limits to top `max_rows`
    - Truncates long Status cells
    """
    if df is None or df.empty:
        return df

    safe = df.copy()

    if max_rows is not None and len(safe) > max_rows:
        safe = safe.head(max_rows)

    INT_CANDIDATES = [
        "Scenario #", "Rooms Filled", "Reg Pairs", "Capacity",
        "reg_max/day", "reg_max_total", "reg_min_total",
        "adcom_max/day", "adcom_max_total", "adcom_min_total",
    ]
    FLOAT_COLS = ["Percent Filled", "Objective"]

    # Normalize floats
    for c in FLOAT_COLS:
        if c in safe.columns:
            s = pd.to_numeric(safe[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan)
            safe[c] = s

    # Helper: all non-null values are whole numbers (with tolerance)
    def _int_like(series: pd.Series) -> bool:
        vals = pd.to_numeric(series, errors="coerce")
        vals = vals[vals.notna()]
        if vals.empty:
            return True
        return bool(np.all(np.isfinite(vals) & np.isclose(vals, np.round(vals)))))

    # Integer candidates: only cast to Int64 when safe; otherwise keep as float
    for c in INT_CANDIDATES:
        if c in safe.columns:
            vals = pd.to_numeric(safe[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if _int_like(vals):
                r = vals.round(0)
                mask = r.isna().to_numpy()
                int_vals = r.fillna(0).astype("int64").to_numpy()
                safe[c] = pd.Series(pd.arrays.IntegerArray(int_vals, mask), dtype="Int64")
            else:
                safe[c] = vals  # remains float64 with NaN allowed

    if "Status" in safe.columns:
        safe["Status"] = safe["Status"].astype(str).str.slice(0, 240)

    return safe

# ---------------------------
# App init
# ---------------------------
st.set_page_config(page_title="Scheduler", layout="wide")
st.title("🎯 Interview Scheduler")
if "needs_rerun" not in st.session_state:
    st.session_state["needs_rerun"] = False
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None

# ---------------------------
# Sidebar (collapsible groups)
# ---------------------------
with st.sidebar:
    # 1) Assignment Limits
    with st.expander("Assignment Limits", expanded=False):
        st.markdown("**Select Group Constraints**")
        reg_max_daily = st.number_input("Regular MAX per day", 0, 10, 6, key="reg_max_daily", on_change=_mark_dirty)
        reg_max_total = st.number_input("Regular MAX total", 0, 50, 10, key="reg_max_total", on_change=_mark_dirty)
        reg_min_total = st.number_input("Regular MIN total", 0, 50, 0, key="reg_min_total", on_change=_mark_dirty)

        sen_max_daily = st.number_input("Adcom MAX per day", 0, 10, 5, key="sen_max_daily", on_change=_mark_dirty)
        sen_max_total = st.number_input("Adcom MAX total", 0, 50, 5, key="sen_max_total", on_change=_mark_dirty)
        sen_min_total = st.number_input("Adcom MIN total", 0, 50, 0, key="sen_min_total", on_change=_mark_dirty)

    # 2) Settings
    with st.expander("Settings", expanded=False):
        st.markdown("**Solver runtime & adjacency**")
        time_limit = st.number_input("Time limit (s)", 10, 1800, 120, key="time_limit", on_change=_mark_dirty)
        threads = st.number_input("Threads (0=auto)", 0, 64, 0, key="threads", on_change=_mark_dirty)
        b2b_mode = st.selectbox("Back-to-back", ["soft", "hard", "off"], index=0, key="b2b_mode", on_change=_mark_dirty)
        observer_extra = st.number_input("Observer extra per slot", 0, 10, 0, key="observer_extra", on_change=_mark_dirty)
        adjacency_grace = st.number_input("Adjacency grace (min)", 0, 30, 0, key="adjacency_grace", on_change=_mark_dirty)

        with st.expander("Objective Weights", expanded=False):
            w_pairs = st.number_input("Weight: pairs", 1000, 5_000_000, 1_000_000, step=1000, key="w_pairs", on_change=_mark_dirty)
            w_fill = st.number_input("Weight: fill (Regular rooms)", 0, 5_000_000, 1_000, step=100, key="w_fill", on_change=_mark_dirty)
            w_b2b = st.number_input("Weight: back-to-back comfort", 0, 5_000_000, 1, step=1, key="w_b2b", on_change=_mark_dirty)
            scarcity_bonus = st.number_input("Scarcity bonus", 0, 1000, 0, key="scarcity_bonus", on_change=_mark_dirty)
            w_fill_adcom = st.number_input("Weight: fill (Adcom rooms)", 0, 5_000_000, 0, step=100, key="w_fill_adcom", on_change=_mark_dirty)

        with st.expander("Day Capacities (optional)", expanded=False):
            st.caption('JSON mapping of day -> cap, e.g. {"2025-10-01": 120}')
            day_caps_text = st.text_area("Day caps JSON", value="", key="day_caps_text", on_change=_mark_dirty)

        with st.expander("Legacy parsing options", expanded=False):
            year = st.number_input("Calendar year", 2000, 2100, 2025, key="year", on_change=_mark_dirty)
            slot_minutes = st.number_input("Slot length (minutes)", 5, 240, 120, key="slot_minutes", on_change=_mark_dirty)

    # 3) Auto-scan Defaults (SLIDER-BASED with wider first-run defaults)
    with st.expander("Auto-scan defaults (experimental)", expanded=False):
        st.caption(
            "Pick ranges (inclusive). The solver will try every combination and rank by "
            "Filled rooms, Regular pairs, then Objective."
        )

        # One step control for all sliders (quantization)
        granularity = st.number_input("Granularity (step)", 1, 50, 1, key="scan_step")

        # Wider defaults ON FIRST RENDER (and after upload reset).
        # Use ±3 for per-day, ±10 for totals/min totals (full width 6/20).
        w_day = 6
        w_total = 20

        # Initialize defaults once (no-op if keys already set)
        _init_range_state("reg_max_daily_range", 0, 10, int(reg_max_daily), width=w_day,  step=int(granularity))
        _init_range_state("reg_max_total_range", 0, 50, int(reg_max_total), width=w_total, step=int(granularity))
        _init_range_state("reg_min_total_range", 0, 50, int(reg_min_total), width=w_total, step=int(granularity))

        _init_range_state("sen_max_daily_range", 0, 10, int(sen_max_daily), width=w_day,  step=int(granularity))
        _init_range_state("sen_max_total_range", 0, 50, int(sen_max_total), width=w_total, step=int(granularity))
        _init_range_state("sen_min_total_range", 0, 50, int(sen_min_total), width=w_total, step=int(granularity))

        # Render sliders (values come from session_state, and will persist)
        st.markdown("**Regulars**")
        reg_max_daily_min, reg_max_daily_max = st.slider(
            "Regular max/day", 0, 10, st.session_state["reg_max_daily_range"],
            step=int(granularity), key="reg_max_daily_range", help="Drag handles to set the inclusive min/max."
        )
        reg_max_total_min, reg_max_total_max = st.slider(
            "Regular max total", 0, 50, st.session_state["reg_max_total_range"],
            step=int(granularity), key="reg_max_total_range"
        )
        reg_min_total_min, reg_min_total_max = st.slider(
            "Regular min total", 0, 50, st.session_state["reg_min_total_range"],
            step=int(granularity), key="reg_min_total_range"
        )

        st.markdown("**Adcom**")
        sen_max_daily_min, sen_max_daily_max = st.slider(
            "Adcom max/day", 0, 10, st.session_state["sen_max_daily_range"],
            step=int(granularity), key="sen_max_daily_range"
        )
        sen_max_total_min, sen_max_total_max = st.slider(
            "Adcom max total", 0, 50, st.session_state["sen_max_total_range"],
            step=int(granularity), key="sen_max_total_range"
        )
        sen_min_total_min, sen_min_total_max = st.slider(
            "Adcom min total", 0, 50, st.session_state["sen_min_total_range"],
            step=int(granularity), key="sen_min_total_range"
        )

        # Step values for the ranges
        reg_max_daily_step = st.number_input("Step: Regular max/day", 1, 10, int(granularity))
        reg_max_total_step = st.number_input("Step: Regular max total", 1, 50, int(granularity))
        reg_min_total_step = st.number_input("Step: Regular min total", 1, 50, int(granularity))

        sen_max_daily_step = st.number_input("Step: Adcom max/day", 1, 10, int(granularity))
        sen_max_total_step = st.number_input("Step: Adcom max total", 1, 50, int(granularity))
        sen_min_total_step = st.number_input("Step: Adcom min total", 1, 50, int(granularity))

        max_scenarios_warn = st.number_input("Warn if scenarios >", 10, 10_000, 500)

        run_autoscan = st.button("Run auto-scan", type="secondary")

# ---------- Main content (file upload + parsing) ----------
up = st.file_uploader("Upload legacy workbook (.xlsx)", type=["xlsx"], on_change=_on_upload_change)
if not up:
    st.info("Upload the workbook to begin.")
    st.stop()

# Quick sniff: legacy vs new format
try:
    xls_preview = pd.ExcelFile(up)
    sheets = set(xls_preview.sheet_names)
    use_legacy = {"Max_Pairs_Per_Slot", "Master_Availability_Sheet", "Adcom_Availability"}.issubset(sheets)
except Exception as e:
    st.error(f"Cannot open workbook: {e}")
    st.stop()

if not use_legacy:
    st.error("Workbook doesn't match the legacy format. Please provide the original workbook.")
    st.stop()

st.success("Detected legacy workbook format ✔️")

# Parse legacy with the limits from the top expander
try:
    inputs = read_inputs_from_legacy(
        up, year=int(year), slot_minutes=int(slot_minutes),
        defaults={
            "reg_max_daily": int(reg_max_daily),
            "reg_max_total": int(reg_max_total),
            "reg_min_total": int(reg_min_total),
            "senior_max_daily": int(sen_max_daily),
            "senior_max_total": int(sen_max_total),
            "senior_min_total": int(sen_min_total),
        }
    )
    _sanity_check_slot_durations(inputs, expected_minutes=int(slot_minutes))
except Exception as e:
    st.error(f"Failed to parse legacy workbook: {e}")
    st.stop()

# Build adjacency
nexts = build_adjacency(inputs.slots, grace_min=cfg.adjacency_grace_min)
slots2 = [
    Slot(
        id=s.id, start=s.start, end=s.end, day_key=s.day_key,
        adjacent_forward=frozenset(nexts.get(s.id, tuple()))
    )
    for s in inputs.slots
]
inputs_view = Inputs(
    interviewers=inputs.interviewers,
    slots=slots2,
    max_pairs_per_slot=inputs.max_pairs_per_slot
)

# Compose Settings
cfg = Settings(
    time_limit_s=float(time_limit),
    threads=int(threads),
    back_to_back_mode=str(b2b_mode),
    observer_extra_per_slot=int(observer_extra),
    w_pairs=int(w_pairs),
    w_fill=int(w_fill),
    w_b2b=int(w_b2b),
    adjacency_grace_min=int(adjacency_grace),
    scarcity_bonus=int(scarcity_bonus),
    w_fill_adcom=int(w_fill_adcom),
    day_caps=(json.loads(day_caps_text) if day_caps_text.strip() else None),
)

# --- Preview ---
st.markdown("### 2) Data preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Interviewers", len(inputs.interviewers))
with c2:
    st.metric("Slots", len(inputs.slots))
with c3:
    st.metric("Slots with capacity", sum(1 for v in inputs.max_pairs_per_slot.values() if v > 0))

with st.expander("People"):
    st.dataframe(pd.DataFrame([{ "id": iv.id, "name": iv.name, "kind": iv.kind } for iv in inputs.interviewers]))

with st.expander("Slots"):
    st.dataframe(pd.DataFrame([{
        "slot_id": s.id, "start": s.start, "end": s.end,
        "day": s.day_key, "cap_pairs": inputs.max_pairs_per_slot.get(s.id, 0),
        "adjacent": list(s.adjacent_forward)
    } for s in inputs.slots]))

# =========================
#  Auto-scan (grid search)
# =========================
if run_autoscan:
    # Build grids from ranges
    reg_max_daily_grid = _build_range(reg_max_daily_min, reg_max_daily_max, reg_max_daily_step)
    reg_max_total_grid = _build_range(reg_max_total_min, reg_max_total_max, reg_max_total_step)
    reg_min_total_grid = _build_range(reg_min_total_min, reg_min_total_max, reg_min_total_step)

    sen_max_daily_grid = _build_range(sen_max_daily_min, sen_max_daily_max, sen_max_daily_step)
    sen_max_total_grid = _build_range(sen_max_total_min, sen_max_total_max, sen_max_total_step)
    sen_min_total_grid = _build_range(sen_min_total_min, sen_min_total_max, sen_min_total_step)

    grid = list(itertools.product(
        reg_max_daily_grid, reg_max_total_grid, reg_min_total_grid,
        sen_max_daily_grid, sen_max_total_grid, sen_min_total_grid
    ))

    st.info(f"Trying {len(grid)} scenario(s) …")
    if len(grid) > max_scenarios_warn:
        st.warning("This may take a while. Consider narrowing the ranges.")

    # Cache the upload bytes so we can re-parse cheaply
    try:
        file_bytes = up.getvalue()
    except Exception:
        file_bytes = None

    results_rows = []
    best = None  # ((pct_filled, reg_pairs, objective), scenario index, row dict)

    prog = st.progress(0.0)
    for idx, (r_md, r_mt, r_mn, s_md, s_mt, s_mn) in enumerate(grid, start=1):
        # Re-parse with these defaults
        wb = io.BytesIO(file_bytes) if file_bytes is not None else up
        try:
            inputs_i = read_inputs_from_legacy(
                wb, year=int(year), slot_minutes=int(slot_minutes),
                defaults={
                    "reg_max_daily": int(r_md),
                    "reg_max_total": int(r_mt),
                    "reg_min_total": int(r_mn),
                    "senior_max_daily": int(s_md),
                    "senior_max_total": int(s_mt),
                    "senior_min_total": int(s_mn),
                }
            )
            _sanity_check_slot_durations(inputs_i, expected_minutes=int(slot_minutes))
        except Exception as e:
            results_rows.append({
                "Scenario #": idx,
                "Status": f"PARSE FAIL: {e}",
                "Rooms Filled": None, "Reg Pairs": None, "Capacity": None, "Percent Filled": None,
                "Objective": None,
                "reg_max/day": r_md, "reg_max_total": r_mt, "reg_min_total": r_mn,
                "adcom_max/day": s_md, "adcom_max_total": s_mt, "adcom_min_total": s_mn,
            })
            prog.progress(idx/len(grid))
            continue

        # Build adjacency (again, fast)
        nexts_i = build_adjacency(inputs_i.slots, grace_min=cfg.adjacency_grace_min)
        slots2_i = [
            Slot(
                id=s.id, start=s.start, end=s.end, day_key=s.day_key,
                adjacent_forward=frozenset(nexts_i.get(s.id, tuple()))
            )
            for s in inputs_i.slots
        ]
        inputs_i_view = Inputs(
            interviewers=inputs_i.interviewers,
            slots=slots2_i,
            max_pairs_per_slot=inputs_i.max_pairs_per_slot
        )

        # Shortened time limit per scenario to keep scans practical
        cfg_scan = Settings(
            time_limit_s=float(time_limit),
            threads=cfg.threads,
            back_to_back_mode=cfg.back_to_back_mode,
            observer_extra_per_slot=cfg.observer_extra_per_slot,
            w_pairs=cfg.w_pairs,
            w_fill=cfg.w_fill,
            w_b2b=cfg.w_b2b,
            adjacency_grace_min=cfg.adjacency_grace_min,
            scarcity_bonus=cfg.scarcity_bonus,
            w_fill_adcom=cfg.w_fill_adcom,
            day_caps=getattr(cfg, "day_caps", None),
        )

        # Seed + solve
        try:
            hint_i = greedy_seed(inputs_i_view)
            res_i = solve_weighted(inputs_i_view, cfg_scan, hint=hint_i)
        except Exception as e:
            results_rows.append({
                "Scenario #": idx,
                "Status": f"SOLVE FAIL: {e}",
                "Rooms Filled": None, "Reg Pairs": None, "Capacity": None, "Percent Filled": None,
                "Objective": None,
                "reg_max/day": r_md, "reg_max_total": r_mt, "reg_min_total": r_mn,
                "adcom_max/day": s_md, "adcom_max_total": s_mt, "adcom_min_total": s_mn,
            })
            prog.progress(idx/len(grid))
            continue

        assign_i = res_i.get("assign", {})
        rooms_filled, reg_pairs, capacity = _compute_rooms_metrics(inputs_i, res_i, assign_i)
        pct = None if not capacity else 100.0 * rooms_filled / capacity
        objective = float(res_i.get("objective", 0.0))
        status = res_i.get("status", "UNKNOWN")

        # --- Guard non-OPTIMAL scenarios and clamp ---
        if status != "OPTIMAL":
            rooms_filled = 0
            reg_pairs = 0
            pct = 0.0 if capacity else 0.0
            objective = float("nan")
        else:
            cap_i = int(capacity or 0)
            rooms_filled = int(max(0, min(int(rooms_filled), cap_i)))
            reg_pairs    = int(max(0, min(int(reg_pairs),    cap_i)))
            pct = 0.0 if cap_i == 0 else 100.0 * rooms_filled / cap_i

        row = {
            "Scenario #": idx,
            "Status": status,
            "Rooms Filled": rooms_filled,
            "Reg Pairs": reg_pairs,
            "Capacity": capacity,
            "Percent Filled": round(pct, 2) if pct is not None else None,
            "Objective": objective,
            "reg_max/day": r_md, "reg_max_total": r_mt, "reg_min_total": r_mn,
            "adcom_max/day": s_md, "adcom_max_total": s_mt, "adcom_min_total": s_mn,
        }
        results_rows.append(row)

        # Track best (lexicographic: Percent Filled, reg_pairs, objective)
        key = (
            (pct if pct is not None else -1.0),
            (reg_pairs or -1),
            (objective or -1),
        )
        if (best is None) or (key > best[0]):
            best = (key, idx, row)

        prog.progress(idx/len(grid))

    # Show results table (TOP 50 by % filled, then reg pairs, then objective)
    if results_rows:
        df_scan = pd.DataFrame(results_rows)
        # Coerce numeric columns for robust sorting
        df_scan["Percent Filled"] = pd.to_numeric(df_scan["Percent Filled"], errors="coerce")
        df_scan["Reg Pairs"] = pd.to_numeric(df_scan["Reg Pairs"], errors="coerce")
        df_scan["Objective"] = pd.to_numeric(df_scan["Objective"], errors="coerce")

        df_scan_sorted = df_scan.sort_values(
            by=["Percent Filled", "Reg Pairs", "Objective"],
            ascending=[False, False, False],
            na_position="last",
        )

        TOP_N = 50
        df_scan_top = df_scan_sorted.head(TOP_N)

        st.markdown(f"### Auto-scan results — Top {min(TOP_N, len(df_scan_sorted))} of {len(df_scan_sorted)} by % filled")

        # Clean a copy for display (avoid Arrow overflow). Top 50 is small; no row cap needed.
        df_scan_view = _arrow_safe_scan_df(df_scan_top, max_rows=None)
        st.dataframe(df_scan_view, use_container_width=True)

        # CSV download of ALL scenarios
        csv_bytes = df_scan_sorted.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download all scenarios (CSV)", csv_bytes, file_name="autoscan_results.csv", mime="text/csv")

        # Best scenario banner
        if best is not None:
            best_row = best[2]
            st.success(
                f"Best scenario → % Filled: {best_row['Percent Filled']} | "
                f"Rooms: {best_row['Rooms Filled']}/{best_row['Capacity']} • "
                f"Reg Pairs: {best_row['Reg Pairs']} | Objective: {best_row['Objective']}"
            )
            # Button to apply the best defaults back into the sidebar
            if st.button("Apply best defaults to sidebar"):
                st.session_state["reg_max_daily"] = int(best_row["reg_max/day"])
                st.session_state["reg_max_total"] = int(best_row["reg_max_total"])
                st.session_state["reg_min_total"] = int(best_row["reg_min_total"])
                st.session_state["sen_max_daily"] = int(best_row["adcom_max/day"])
                st.session_state["sen_max_total"] = int(best_row["adcom_max_total"])
                st.session_state["sen_min_total"] = int(best_row["adcom_min_total"])
                st.session_state["needs_rerun"] = True
                st.info("Applied best defaults. Click **Run scheduler** to solve with these settings.")

        # --- Run any of the top scenarios (skip rows with missing defaults) ---
        st.markdown("#### Run any of the top scenarios")
        for _, row in df_scan_top.iterrows():
            required_keys = [
                "reg_max/day","reg_max_total","reg_min_total",
                "adcom_max/day","adcom_max_total","adcom_min_total",
            ]
            # Show line without a Run button if any defaults are missing/not-a-number
            if any(pd.isna(row[k]) for k in required_keys):
                scen_label = int(row["Scenario #"]) if pd.notna(row["Scenario #"]) else "—"
                st.write(
                    f"**Scenario #{scen_label}** — Status: {row['Status']} • "
                    f"Rooms: {row['Rooms Filled']} • Reg Pairs: {row['Reg Pairs']} • "
                    f"Obj: {row['Objective']} • % Filled: {row['Percent Filled']}"
                )
                continue

            scn_id = int(row["Scenario #"]) if pd.notna(row["Scenario #"]) else 0
            with st.expander(f"Scenario #{scn_id}"):
                st.write(
                    f"Status: {row['Status']} • Rooms: {row['Rooms Filled']} "
                    f"/ {row['Capacity']} • Reg Pairs: {row['Reg Pairs']} • Obj: {row['Objective']}"
                )
                if st.button("Run", key=f"run_scn_{scn_id}"):
                    wb = io.BytesIO(file_bytes) if file_bytes is not None else up
                    try:
                        inputs_run = read_inputs_from_legacy(
                            wb, year=int(year), slot_minutes=int(slot_minutes),
                            defaults={
                                "reg_max_daily": int(row["reg_max/day"]),
                                "reg_max_total": int(row["reg_max_total"]),
                                "reg_min_total": int(row["reg_min_total"]),
                                "senior_max_daily": int(row["adcom_max/day"]),
                                "senior_max_total": int(row["adcom_max_total"]),
                                "senior_min_total": int(row["adcom_min_total"]),
                            }
                        )
                        _sanity_check_slot_durations(inputs_run, expected_minutes=int(slot_minutes))
                        nexts_run = build_adjacency(inputs_run.slots, grace_min=cfg.adjacency_grace_min)
                        slots2_run = [
                            Slot(
                                id=s.id, start=s.start, end=s.end, day_key=s.day_key,
                                adjacent_forward=frozenset(nexts_run.get(s.id, tuple()))
                            ) for s in inputs_run.slots
                        ]
                        inputs_for_res = Inputs(
                            interviewers=inputs_run.interviewers,
                            slots=slots2_run,
                            max_pairs_per_slot=inputs_run.max_pairs_per_slot,
                        )

                        # Use current sidebar Settings
                        cfg_now = Settings(
                            time_limit_s=float(time_limit),
                            threads=cfg.threads,
                            back_to_back_mode=cfg.back_to_back_mode,
                            observer_extra_per_slot=cfg.observer_extra_per_slot,
                            w_pairs=cfg.w_pairs,
                            w_fill=cfg.w_fill,
                            w_b2b=cfg.w_b2b,
                            adjacency_grace_min=cfg.adjacency_grace_min,
                            scarcity_bonus=cfg.scarcity_bonus,
                            w_fill_adcom=cfg.w_fill_adcom,
                            day_caps=getattr(cfg, "day_caps", None),
                        )

                        with st.spinner("Solving with CP-SAT…"):
                            hint_run = greedy_seed(inputs_for_res)
                            res_run = solve_weighted(inputs_for_res, cfg_now, hint=hint_run)

                        st.session_state["last_results"] = {"res": res_run, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        st.session_state["needs_rerun"] = False
                        st.session_state["inputs_for_results"] = inputs_for_res

                        # Reflect chosen defaults to controls for convenience
                        st.session_state["reg_max_daily"] = int(row["reg_max/day"])
                        st.session_state["reg_max_total"] = int(row["reg_max_total"])
                        st.session_state["reg_min_total"] = int(row["reg_min_total"])
                        st.session_state["sen_max_daily"] = int(row["adcom_max/day"])
                        st.session_state["sen_max_total"] = int(row["adcom_max_total"])
                        st.session_state["sen_min_total"] = int(row["adcom_min_total"])

                        st.success(f"Scenario #{scn_id} solved. Refreshing…")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Failed to run scenario #{scn_id}: {e}")

# -----------------------------------
# 3) Solve (persistent results view)
# -----------------------------------
st.markdown("### 3) Solve")

# Run button FIRST so we can clear stale state in the same render
run_clicked = st.button("Run scheduler", type="primary")
if run_clicked:
    with st.spinner("Solving with CP-SAT…"):
        hint = greedy_seed(inputs)
        res = solve_weighted(inputs, cfg, hint=hint)
    st.session_state["last_results"] = {"res": res, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    st.session_state["needs_rerun"] = False
    st.session_state["inputs_for_results"] = inputs  # ensure results correspond to these inputs

# Decide what to display: current (just run) or last results
current_res = (st.session_state.get("last_results") or {}).get("res")
if not current_res:
    st.info("Click **Run scheduler** to produce a schedule.")
    st.stop()

# ---------- Results/Dashboard (built from persisted results) ----------
res = current_res
assign = res["assign"]
pairs = res.get("pairs", {})                 # Regular pairs per Date_Time
adcom_singles = res.get("adcom_singles")     # Adcom singles per Date_Time (may be None on older solver builds)
b2b = res.get("b2b", {})  # may be unused depending on solver version

# Use the inputs that correspond to the current results (persisted at run time)
inputs_view = st.session_state.get("inputs_for_results", inputs_view)

iv_by_id = {iv.id: iv for iv in inputs_view.interviewers}
slot_ids = [s.id for s in inputs_view.slots]
cap_map = inputs_view.max_pairs_per_slot

# If not OPTIMAL, zero out derived metrics to avoid bogus counts
if res.get("status") != "OPTIMAL":
    rooms_used_by_t = {t: 0 for t in slot_ids}
    total_rooms_used = 0
    total_capacity = int(sum(int(cap_map.get(t, 0)) for t in slot_ids))
    pct_filled = 0.0 if total_capacity == 0 else 0.0
else:

# Rooms used per Date_Time (pairs + adcom singles)
    rooms_used_by_t: dict[str, int] = {}
    if adcom_singles is not None:
        for t in slot_ids:
            rooms_used_by_t[t] = int(pairs.get(t, 0)) + int(adcom_singles.get(t, 0))
        # Clamp used rooms by slot capacity
        for t in slot_ids:
            rooms_used_by_t[t] = max(0, min(int(rooms_used_by_t[t]), int(cap_map.get(t, 0))))
    else:
        # Fallback: compute from raw assignments; filter to valid ids/slots
        valid_ids = set(iv_by_id.keys())
        valid_slots = set(slot_ids)
        reg_people_by_t = {t: 0 for t in slot_ids}
        adcom_people_by_t = {t: 0 for t in slot_ids}
        for (i, t), v in assign.items():
            if not v or (i not in valid_ids) or (t not in valid_slots):
                continue
            if iv_by_id[i].kind == "Regular":
                reg_people_by_t[t] += 1
            elif iv_by_id[i].kind == "Senior":
                adcom_people_by_t[t] += 1
        for t in slot_ids:
            rooms_used_by_t[t] = (reg_people_by_t[t] // 2) + adcom_people_by_t[t]
        # Clamp used rooms by slot capacity
        for t in slot_ids:
            rooms_used_by_t[t] = max(0, min(int(rooms_used_by_t[t]), int(cap_map.get(t, 0))))

    total_rooms_used = int(sum(rooms_used_by_t.values()))
    total_capacity = int(sum(int(cap_map.get(t, 0)) for t in slot_ids))
    pct_filled = 0.0 if total_capacity == 0 else 100.0 * total_rooms_used / total_capacity

# --- Regular vs Adcom dashboard breakdown ---
if adcom_singles is not None and res.get("status") == "OPTIMAL":
    reg_rooms_used = int(sum(pairs.get(t, 0) for t in slot_ids))
    adcom_rooms_used = int(sum(adcom_singles.get(t, 0) for t in slot_ids))
else:
    reg_rooms_used = int(sum(min(rooms_used_by_t.get(t, 0), int(cap_map.get(t, 0))) for t in slot_ids))
    adcom_rooms_used = 0  # unknown split without explicit singles; treat as zero for display

used_total = reg_rooms_used + adcom_rooms_used
reg_share_used = 0.0 if used_total == 0 else 100.0 * reg_rooms_used / used_total
adcom_share_used = 100.0 - reg_share_used

reg_pct_capacity = 0.0 if total_capacity == 0 else 100.0 * reg_rooms_used / total_capacity
adcom_pct_capacity = 0.0 if total_capacity == 0 else 100.0 * adcom_rooms_used / total_capacity

c1, c2 = st.columns(2)
with c1:
    st.metric("Regular (rooms)", f"{reg_rooms_used}/{total_capacity}",
              help="Rooms occupied by Regular pairs (2 people per room).")
    st.progress(min(max(reg_pct_capacity/100.0, 0.0), 1.0),
                text=f"{reg_share_used:.1f}% of used • {reg_pct_capacity:.1f}% of capacity")
with c2:
    st.metric("Adcom (rooms)", f"{adcom_rooms_used}/{total_capacity}")
    st.progress(min(max(adcom_pct_capacity/100.0, 0.0), 1.0),
                text=f"{adcom_share_used:.1f}% of used • {adcom_pct_capacity:.1f}% of capacity")

# ----- Export buttons -----
try:
    excel_path = f"/tmp/schedule_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    out_path = make_excel_report(inputs_view, assign, path=excel_path)
    with open(out_path, "rb") as fh:
        excel_bytes = fh.read()
    st.success("✅ Excel report is ready.")
    st.download_button(
        "⬇️ Download Excel Report (.xlsx)",
        excel_bytes,
        file_name="schedule_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"dl_xlsx_{st.session_state['last_results']['ts']}",
    )
except Exception as e:
    st.error(f"Failed to prepare Excel report: {e}")
