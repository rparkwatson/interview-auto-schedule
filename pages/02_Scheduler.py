from __future__ import annotations
import json
from datetime import datetime
import collections
import pandas as pd
import numpy as np
import streamlit as st
import io
import itertools
import altair as alt

from scheduler.config import Settings
from scheduler.io.read import read_inputs_from_legacy
from scheduler.io.write import make_excel_report
from scheduler.preprocess.calendar import build_adjacency
from scheduler.heuristics.seed import greedy_seed
from scheduler.solvers.cpsat import solve_weighted
from scheduler.domain import Slot, Inputs as Inp

# ---------------------------
# Utilities
# ---------------------------
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

def _init_range_state(state_key: str, min_v: int, max_v: int, seed: int, width: int, step: int):
    """
    Initialize st.session_state[state_key] to a wider (lo, hi) range centered on seed.
    - width is the full span, in the same units as the slider.
    - snaps lo/hi to the step and clamps to [min_v, max_v].
    Does nothing if the key already exists (i.e., user already interacted).
    """
    if state_key in st.session_state:
        return
    step = max(1, int(step))
    seed = int(seed)
    min_v, max_v = int(min_v), int(max_v)
    width = max(step, int(width))
    half = max(step, width // 2)

    lo = max(min_v, seed - half)
    hi = min(max_v, seed + half)

    # snap to step grid
    def _snap(x):
        return min(max_v, max(min_v, ((x - min_v) // step) * step + min_v))
    lo = _snap(lo)
    hi = _snap(hi)
    if lo > hi:
        hi = min(max_v, lo + step)

    st.session_state[state_key] = (lo, hi)

def _compute_rooms_metrics(inputs_local, res_local, assign_local):
    """Returns (rooms_filled, regular_pairs, capacity). Works with new/old solver outputs."""
    pairs_local = res_local.get("pairs", {})
    adcom_singles_local = res_local.get("adcom_singles")
    slot_ids_local = [s.id for s in inputs_local.slots]
    cap_map_local = inputs_local.max_pairs_per_slot
    iv_by_id_local = {iv.id: iv for iv in inputs_local.interviewers}

    if adcom_singles_local is not None:
        rooms_filled = int(sum(int(pairs_local.get(t, 0)) + int(adcom_singles_local.get(t, 0)) for t in slot_ids_local))
        reg_pairs = int(sum(int(pairs_local.get(t, 0)) for t in slot_ids_local))
    else:
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
    Prepare a DataFrame for st.dataframe() to avoid Arrow/casting errors.
    - Coerces float-like columns
    - Casts integer-like columns to Int64 when safe
    - If any value exceeds int64 bounds, converts that entire column to string
    - Replaces ±inf with NaN
    - Optionally limits rows
    - Truncates long Status cells
    """
    if df is None or df.empty:
        return df

    safe = df.copy()

    if max_rows is not None and len(safe) > max_rows:
        safe = safe.head(max_rows)

    INT_CANDIDATES = [
        "Scenario #", "Rooms Filled", "Reg Pairs", "Capacity",
        "Filled", "Run #",
        "Rooms Used", "Used_Rooms", "Unused_Rooms",
    ]
    FLOAT_COLS = [
        "Percent Filled", "Objective",
        "Pct of Capacity (%)", "Share of Used (%)",
        "Unused_%",
    ]

    # 1) Floats
    for c in FLOAT_COLS:
        if c in safe.columns:
            s = pd.to_numeric(safe[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], pd.NA)
            safe[c] = s.astype("Float64")

    # 2) Ints (with overflow guard)
    MAX_I64 = np.iinfo(np.int64).max

    def _coerce_intish(col: pd.Series) -> pd.Series:
        v = pd.to_numeric(col, errors="coerce")  # may be Int64/float/object of big-ints
        v = v.replace([np.inf, -np.inf], pd.NA)

        # Detect any value outside int64 bounds
        big = v.dropna()
        try:
            exceeds = (big.abs() > MAX_I64).any()
        except Exception:
            exceeds = True  # if weird dtype, force safe path

        if exceeds:
            # Convert entire column to strings to keep Arrow happy
            return col.astype(str)

        # If integral-like, keep as Int64 (nullable)
        # If not integral, fall back to Float64 to be safe
        v_float = pd.to_numeric(v, errors="coerce").astype("Float64")
        is_integral = v_float.dropna().apply(lambda x: float(x).is_integer()).all()
        if is_integral:
            try:
                return v.astype("Int64")
            except Exception:
                return v_float
        return v_float

    for c in INT_CANDIDATES:
        if c in safe.columns:
            safe[c] = _coerce_intish(safe[c])

    if "Status" in safe.columns:
        safe["Status"] = safe["Status"].astype(str).str.slice(0, 240)

    return safe

def _to_int_or_none(x):
    try:
        return int(x) if pd.notna(x) else None
    except Exception:
        return None

# ---------------------------
# App init
# ---------------------------
st.set_page_config(page_title="Scheduler", layout="wide")
st.title("🎯 Step 2 - Interview Scheduler")

st.subheader("Instructions")
st.markdown(
    """
        Run the Scheduler

        1. Complete **Step 1 - Build the Workbook** to automatically pass the excel file to Step 2

        2. Review Data Preview. The app will display summary statistics for the file.

        3. Set limits. In the left sidebar, adjust **Scheduler Limits**.

        4. Run. **Click Run Scheduler**.

        5. Iterate. You can run the scheduler multiple times, changing group constraints as needed.

        6. Compare results. The result for each run is saved in **Scheduler Results** section.
    """
)

if "needs_rerun" not in st.session_state:
    st.session_state["needs_rerun"] = False
if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# ---------------------------
# Sidebar (collapsible groups)
# ---------------------------
with st.sidebar:
    # 1) Assignment Limits
    with st.expander("Scheduler Limits", expanded=True):
        st.markdown("**Select Group Constraints**")
        reg_max_daily = st.number_input("Regular MAX per day", 0, 24, 2, key="reg_max_daily", on_change=_mark_dirty)
        reg_max_total = st.number_input("Regular MAX total", 0, 999, 7, key="reg_max_total", on_change=_mark_dirty)
        reg_min_total = st.number_input("Regular MIN total", 0, 999, 5, key="reg_min_total", on_change=_mark_dirty)
        sen_max_daily = st.number_input("Adcom MAX per day", 0, 24, 2, key="sen_max_daily", on_change=_mark_dirty)
        sen_max_total = st.number_input("Adcom MAX total", 0, 999, 5, key="sen_max_total", on_change=_mark_dirty)
        sen_min_total = st.number_input("Adcom MIN total", 0, 999, 0, key="sen_min_total", on_change=_mark_dirty)

    # 2) Settings
    with st.expander("Settings", expanded=False):
        st.markdown("**Solver runtime & adjacency**")
        time_limit = st.number_input("Time limit (s)", 10, 1800, 120, key="time_limit", on_change=_mark_dirty, disabled=True)
        threads = st.number_input("Threads (0=auto)", 0, 64, 0, key="threads", on_change=_mark_dirty, disabled=True)
        b2b_mode = st.selectbox("Back-to-back", ["soft", "hard", "off"], index=0, key="b2b_mode", on_change=_mark_dirty)
        observer_extra = st.number_input("Observer extra per slot", 0, 10, 0, key="observer_extra", on_change=_mark_dirty, disabled=True)
        adjacency_grace = st.number_input("Adjacency grace (min)", 0, 30, 0, key="adjacency_grace", on_change=_mark_dirty, disabled=True)

        # NEW: randomness control for de-biasing ties/seed order
        random_seed_input = st.number_input(
            "Random seed (0 = fixed jitter)",
            min_value=0, max_value=1_000_000, value=int(st.session_state.get("random_seed", 0)),
            key="random_seed",
            on_change=_mark_dirty,
        )

        with st.expander("Objective Weights", expanded=False):
            w_pairs = st.number_input("Weight: pairs", 1000, 5_000_000, 1_000_000, step=1000, key="w_pairs", on_change=_mark_dirty, disabled=True)
            w_fill = st.number_input("Weight: fill (Regulars)", 10, 100_000, 1000, step=10, key="w_fill", on_change=_mark_dirty, disabled=True)
            w_b2b = st.number_input("Penalty: back-to-back", 0, 1000, 1, key="w_b2b", on_change=_mark_dirty, disabled=True)

        with st.expander("Scarcity Priority", expanded=False):
            scarcity_bonus = st.number_input("Scarcity bonus (per missing Regular)", 0, 100, 5, key="scarcity_bonus", on_change=_mark_dirty, disabled=True)
            w_fill_adcom = st.number_input("Weight: fill (Adcom)", 0, 100_000, 500, step=10, key="w_fill_adcom", on_change=_mark_dirty, disabled=True)

        with st.expander("Global day caps (optional)", expanded=False):
            st.caption('JSON mapping of day -> cap, e.g. {"2025-10-01": 120}')
            day_caps_text = st.text_area("Day caps JSON", value="", key="day_caps_text", on_change=_mark_dirty, disabled=True)

        with st.expander("Legacy parsing options", expanded=False):
            year = st.number_input("Calendar year", 2000, 2100, 2025, key="year", on_change=_mark_dirty, disabled=True)
            slot_minutes = st.number_input("Slot length (minutes)", 5, 240, 120, key="slot_minutes", on_change=_mark_dirty, disabled=True)

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
        w_day = 3
        w_total = 10

        # Initialize defaults once (no-op if keys already set)
        _init_range_state("reg_max_daily_range", 0, 10, int(reg_max_daily), width=w_day,  step=int(granularity))
        _init_range_state("reg_max_total_range", 0, 10, int(reg_max_total), width=w_total, step=int(granularity))
        _init_range_state("reg_min_total_range", 0, 10, int(reg_min_total), width=w_total, step=int(granularity))

        _init_range_state("sen_max_daily_range", 0, 10, int(sen_max_daily), width=w_day,  step=int(granularity))
        _init_range_state("sen_max_total_range", 0, 10, int(sen_max_total), width=w_total, step=int(granularity))
        _init_range_state("sen_min_total_range", 0, 10, int(sen_min_total), width=w_total, step=int(granularity))

        # Render sliders (values come from session_state, and will persist)
        st.markdown("**Regulars**")
        reg_max_daily_min, reg_max_daily_max = st.slider(
            "Regular max/day", 0, 10, st.session_state["reg_max_daily_range"],
            step=int(granularity), key="reg_max_daily_range", help="Drag handles to set the inclusive min/max."
        )
        reg_max_total_min, reg_max_total_max = st.slider(
            "Regular max total", 0, 10, st.session_state["reg_max_total_range"],
            step=int(granularity), key="reg_max_total_range"
        )
        reg_min_total_min, reg_min_total_max = st.slider(
            "Regular min total", 0, 10, st.session_state["reg_min_total_range"],
            step=int(granularity), key="reg_min_total_range"
        )

        st.markdown("**Adcoms**")
        sen_max_daily_min, sen_max_daily_max = st.slider(
            "Adcom max/day", 0, 10, st.session_state["sen_max_daily_range"],
            step=int(granularity), key="sen_max_daily_range"
        )
        sen_max_total_min, sen_max_total_max = st.slider(
            "Adcom max total", 0, 10, st.session_state["sen_max_total_range"],
            step=int(granularity), key="sen_max_total_range"
        )
        sen_min_total_min, sen_min_total_max = st.slider(
            "Adcom min total", 0, 10, st.session_state["sen_min_total_range"],
            step=int(granularity), key="sen_min_total_range"
        )

        # Mirror per-field step vars so the rest of the code uses them
        reg_max_daily_step = reg_max_total_step = reg_min_total_step = int(granularity)
        sen_max_daily_step = sen_max_total_step = sen_min_total_step = int(granularity)

        # Per-scenario config
        scan_time_limit = st.number_input(
            "Time limit per scenario (s)", 5, 900, min(60, int(time_limit)),
            key="scan_time_limit", disabled=True
        )
        max_scenarios_warn = st.number_input(
            "Warn if scenarios exceed", 1, 500, 50, key="max_scenarios_warn", disabled=True
        )

        # Scenario count estimate
        _est = (
            len(_build_range(reg_max_daily_min, reg_max_daily_max, reg_max_daily_step)) *
            len(_build_range(reg_max_total_min, reg_max_total_max, reg_max_total_step)) *
            len(_build_range(reg_min_total_min, reg_min_total_max, reg_min_total_step)) *
            len(_build_range(sen_max_daily_min, sen_max_daily_max, sen_max_daily_step)) *
            len(_build_range(sen_max_total_min, sen_max_total_max, sen_max_total_step)) *
            len(_build_range(sen_min_total_min, sen_min_total_max, sen_min_total_step))
        )
        st.caption(f"Estimated scenarios: **{_est:,}**")

        run_autoscan = st.button("Run auto-scan now", type="secondary", key="run_autoscan_btn")

# Build Settings from sidebar values
cfg = Settings(
    time_limit_s=float(time_limit),
    threads=int(threads),
    back_to_back_mode=b2b_mode,
    observer_extra_per_slot=int(observer_extra),
    w_pairs=int(w_pairs),
    w_fill=int(w_fill),
    w_b2b=int(w_b2b),
    adjacency_grace_min=int(adjacency_grace),
    scarcity_bonus=int(scarcity_bonus),
    w_fill_adcom=int(w_fill_adcom),
    random_seed=int(st.session_state.get("random_seed", 0)),  # ← safe access
)
if day_caps_text.strip():
    try:
        cfg.day_caps = {str(k): int(v) for k, v in json.loads(day_caps_text).items()}
    except Exception as e:
        st.warning(f"Ignoring day caps: {e}")

# ---------------------------
# Main content
# ---------------------------
st.markdown("### 1) Choose your workbook source")

has_step1 = "formatted_xlsx" in st.session_state
source = st.radio(
    "Source",
    ["Use file from Step 1", "Upload manually"],
    index=(0 if has_step1 else 1),
    horizontal=True,
)

up = None

if source == "Use file from Step 1":
    if not has_step1:
        st.warning("No formatted workbook found in this session. Please complete Step 1 first.")
        st.page_link("pages/01_Builder.py", label="← Go to Step 1", icon="⬅️")
        st.stop()
    # Use the bytes produced in Step 1
    up = io.BytesIO(st.session_state["formatted_xlsx"])
    st.success("Using workbook from Step 1 (formatted.xlsx).")

else:
    # Fall back to manual upload (keeps your on_change behavior)
    up = st.file_uploader(
        "Upload your original workbook (tabs: Max_Pairs_Per_Slot, Master_Availability_Sheet, Adcom_Availability)",
        type=["xlsx"], key="up", on_change=_on_upload_change
    )
    if not up:
        if st.session_state.get("needs_rerun") and st.session_state.get("last_results"):
            st.info("Settings changed since last run. Upload a workbook and re-run the scheduler.")
        st.stop()

# Detect legacy format
try:
    xls_preview = pd.ExcelFile(up)
    sheets = set(xls_preview.sheet_names)
    use_legacy = {"Max_Pairs_Per_Slot", "Master_Availability_Sheet", "Adcom_Availability"}.issubset(sheets)
except Exception as e:
    st.error(f"Cannot open workbook: {e}")
    st.stop()

if not use_legacy:
    st.error("Workbook doesn't match the format.")
    st.stop()

st.success("Detected correct workbook format ✔️")

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
inputs = Inp(
    interviewers=inputs.interviewers,
    slots=slots2,
    max_pairs_per_slot=inputs.max_pairs_per_slot
)

# Ensure we always have the inputs used for the currently displayed results
if "inputs_for_results" not in st.session_state:
    st.session_state["inputs_for_results"] = inputs

# Preview
st.markdown("### 2) Data Preview")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Interviewers", len(inputs.interviewers))
with c2:
    st.metric("Slots", len(inputs.slots))
with c3:
    st.metric("Slots with capacity", sum(1 for v in inputs.max_pairs_per_slot.values() if v > 0))

# --- Tabs inside a single expander ---
with st.expander("Summary Tables", expanded=False):
    tab_people, tab_slots = st.tabs(["People", "Slots"])

    with tab_people:
        df_people = pd.DataFrame([{
            "id": iv.id, "name": iv.name, "kind": iv.kind,
            "pre_assigned": iv.pre_assigned,
            "min_total": iv.min_total, "max_daily": iv.max_daily, "max_total": iv.max_total,
            "avail_count": len(iv.available_slots)
        } for iv in inputs.interviewers])
        st.dataframe(_arrow_safe_scan_df(df_people), width='stretch')
        st.download_button(
            "⬇️ Download people (CSV)",
            df_people.to_csv(index=False).encode("utf-8"),
            file_name="people.csv",
            mime="text/csv",
            key="dl_people_csv",
        )

    with tab_slots:
        df_slots = pd.DataFrame([{
            "slot_id": s.id, "start": s.start, "end": s.end,
            "day": s.day_key, "cap_pairs": inputs.max_pairs_per_slot.get(s.id, 0),
            "adjacent": list(s.adjacent_forward)
        } for s in inputs.slots])

        # small UI to hide zero-capacity slots
        hide_zero = st.checkbox("Hide zero-capacity slots", value=True)
        view = df_slots[df_slots["cap_pairs"].fillna(0) > 0] if hide_zero else df_slots

        st.dataframe(_arrow_safe_scan_df(view), width='stretch')
        st.download_button(
            "⬇️ Download slots (CSV)",
            view.to_csv(index=False).encode("utf-8"),
            file_name="slots.csv",
            mime="text/csv",
            key="dl_slots_csv",
        )

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

        # Build adjacency (same grace as current cfg)
        nexts_i = build_adjacency(inputs_i.slots, grace_min=cfg.adjacency_grace_min)
        slots2_i = [
            Slot(id=s.id, start=s.start, end=s.end, day_key=s.day_key,
                 adjacent_forward=frozenset(nexts_i.get(s.id, tuple())))
            for s in inputs_i.slots
        ]
        inputs_i = Inp(
            interviewers=inputs_i.interviewers,
            slots=slots2_i,
            max_pairs_per_slot=inputs_i.max_pairs_per_slot
        )

        # Shortened time limit per scenario to keep scans practical
        cfg_scan = Settings(
            time_limit_s=float(scan_time_limit),
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
            random_seed=int(st.session_state.get("random_seed", 0)),
        )

        # Seed + solve
        try:
            hint_i = greedy_seed(inputs_i, seed=getattr(cfg_scan, "random_seed", 0))
            res_i = solve_weighted(inputs_i, cfg_scan, hint=hint_i)
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
        status = (res_i.get("status") or "UNKNOWN").upper()
        is_solution = status in {"OPTIMAL", "FEASIBLE", "SAT", "INTEGER", "FEASIBLE/SOLUTION_FOUND"}

        assign_i = res_i.get("assign", {})
        rooms_filled, reg_pairs, capacity = _compute_rooms_metrics(inputs_i, res_i, assign_i)

        # Only compute these if a solution exists
        pct = 100.0 * rooms_filled / capacity if (is_solution and capacity) else None
        objective = float(res_i.get("objective", 0.0)) if is_solution else None

        # For unsolved rows, clear misleading counts
        if not is_solution:
            res_i["assign"] = {}
            res_i["pairs"] = {}
            res_i["adcom_singles"] = {}

        row = {
            "Scenario #": idx,
            "Status": status,
            "Rooms Filled": rooms_filled,
            "Reg Pairs": reg_pairs,
            "Capacity": capacity,
            "Percent Filled": None if pct is None else round(pct, 1),
            "Objective": (None if objective is None or not np.isfinite(objective) else round(float(objective), 0)),
            "reg_max/day": r_md, "reg_max_total": r_mt, "reg_min_total": r_mn,
            "adcom_max/day": s_md, "adcom_max_total": s_mt, "adcom_min_total": s_mn,
        }
        results_rows.append(row)

        # Track best (lexicographic: Percent Filled, reg_pairs, objective)
        if is_solution:
            key = (
                (pct if pct is not None else -1.0),
                (reg_pairs if reg_pairs is not None else -1),
                (objective if objective is not None else -1),
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
        st.dataframe(df_scan_view, width='stretch')

        # CSV download of ALL scenarios
        csv_bytes = df_scan_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download all auto-scan results (CSV)",
            csv_bytes,
            file_name="autoscan_results.csv",
            mime="text/csv",
            key="dl_autoscan_csv",
        )

        if best is not None:
            best_row = best[2]
            st.success(
                f"Best scenario → % Filled: {best_row['Percent Filled']} | "
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
        try:
            file_bytes = up.getvalue()
        except Exception:
            file_bytes = None

        for _, row in df_scan_top.iterrows():
            required_keys = [
                "reg_max/day","reg_max_total","reg_min_total",
                "adcom_max/day","adcom_max_total","adcom_min_total",
            ]
            # Show line without a Run button if any defaults are missing/not-a-number
            if any(pd.isna(row[k]) for k in required_keys):
                scen_label = _to_int_or_none(row["Scenario #"])
                scen_label = scen_label if scen_label is not None else "—"
                st.write(
                    f"**Scenario #{scen_label}** — Status: {row['Status']} • "
                    f"Rooms: {row['Rooms Filled']} • Reg Pairs: {row['Reg Pairs']} • "
                    f"Obj: {row['Objective']} • % Filled: {row['Percent Filled']}"
                )
                continue

            scn_id = _to_int_or_none(row["Scenario #"])
            if scn_id is None:
                # Fallback label if somehow missing
                scn_id = int(len(results_rows))  # arbitrary but stable-ish

            cols = st.columns([6, 1])
            with cols[0]:
                st.write(
                    f"**Scenario #{scn_id}** — Status: {row['Status']} • "
                    f"Rooms: {row['Rooms Filled']} • Reg Pairs: {row['Reg Pairs']} • Obj: {row['Objective']} • % Filled: {row['Percent Filled']}  \n"
                    f"Defaults → Reg d/t/m: {int(row['reg_max/day'])}/{int(row['reg_max_total'])}/{int(row['reg_min_total'])} • "
                    f"Adcom d/t/m: {int(row['adcom_max/day'])}/{int(row['adcom_max_total'])}/{int(row['adcom_min_total'])}"
                )
            with cols[1]:
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
                        nexts_run = build_adjacency(inputs_run.slots, grace_min=cfg.adjacency_grace_min)
                        slots2_run = [
                            Slot(
                                id=s.id, start=s.start, end=s.end, day_key=s.day_key,
                                adjacent_forward=frozenset(nexts_run.get(s.id, tuple()))
                            ) for s in inputs_run.slots
                        ]
                        inputs_run = Inp(
                            interviewers=inputs_run.interviewers,
                            slots=slots2_run,
                            max_pairs_per_slot=inputs_run.max_pairs_per_slot
                        )

                        with st.spinner(f"Running scheduler for scenario #{scn_id}…"):
                            hint_run = greedy_seed(inputs_run, seed=getattr(cfg, "random_seed", 0))
                            res_run = solve_weighted(inputs_run, cfg, hint=hint_run)

                        st.session_state["last_results"] = {"res": res_run, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                        st.session_state["needs_rerun"] = False
                        st.session_state["inputs_for_results"] = inputs_run

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
st.markdown("### 3) Start Scheduling")

# Run button FIRST so we can clear stale state in the same render
run_clicked = st.button("Run scheduler", type="primary")

if run_clicked:
    with st.spinner("Solving with CP-SAT…"):
        hint = greedy_seed(inputs, seed=getattr(cfg, "random_seed", 0))
        res = solve_weighted(inputs, cfg, hint=hint)
    st.session_state["last_results"] = {"res": res, "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    st.session_state["needs_rerun"] = False
    st.session_state["inputs_for_results"] = inputs  # ensure results correspond to these inputs

# Show stale banner only if still stale *after* handling the click
show_stale = bool(st.session_state.get("needs_rerun") and st.session_state.get("last_results") and not run_clicked)
if show_stale:
    st.warning("Settings changed since the last run. Results below are stale — click **Run scheduler** to refresh.")

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

# Use the inputs that correspond to the current results (scenario-run or normal)
inputs_view = st.session_state.get("inputs_for_results", inputs)

st.subheader("Results")
st.write(f"Status: **{res['status']}** | Objective: **{res['objective']:.0f}**")

# === 🔽 PROMINENT EXCEL DOWNLOAD AT THE TOP ===
excel_path = "schedule_report.xlsx"
try:
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

# ----- Overview dashboard -----
iv_by_id = {iv.id: iv for iv in inputs_view.interviewers}
slot_ids = [s.id for s in inputs_view.slots]
cap_map = inputs_view.max_pairs_per_slot

# Rooms used per Date_Time (pairs + adcom singles)
rooms_used_by_t: dict[str, int] = {}
if adcom_singles is not None:
    for t in slot_ids:
        rooms_used_by_t[t] = int(pairs.get(t, 0)) + int(adcom_singles.get(t, 0))
else:
    # Fallback: compute from raw assignments; filter to valid ids/slots
    valid_ids = set(iv_by_id.keys())
    valid_slots = set(slot_ids)
    reg_people_by_t = {t: 0 for t in slot_ids}
    adcom_people_by_t = {t: 0 for t in slot_ids}
    for (i, t), v in assign.items():
        if not v:
            continue
        if (i not in valid_ids) or (t not in valid_slots):
            continue
        if iv_by_id[i].kind == "Regular":
            reg_people_by_t[t] += 1
        elif iv_by_id[i].kind == "Senior":
            adcom_people_by_t[t] += 1
    for t in slot_ids:
        rooms_used_by_t[t] = (reg_people_by_t[t] // 2) + adcom_people_by_t[t]

total_rooms_used = int(sum(rooms_used_by_t.values()))
total_capacity = int(sum(int(cap_map.get(t, 0)) for t in slot_ids))
pct_filled = 0.0 if total_capacity == 0 else 100.0 * total_rooms_used / total_capacity

# --- Regular vs Adcom dashboard breakdown ---
if adcom_singles is not None:
    reg_rooms_used = int(sum(pairs.get(t, 0) for t in slot_ids))
    adcom_rooms_used = int(sum(adcom_singles.get(t, 0) for t in slot_ids))
else:
    reg_rooms_used = int(sum((reg_people_by_t.get(t, 0) // 2) for t in slot_ids))
    adcom_rooms_used = int(sum(adcom_people_by_t.get(t, 0) for t in slot_ids))

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
    st.metric("Adcom (rooms)", f"{adcom_rooms_used}/{total_capacity}",
              help="Rooms occupied by Adcom singles (1 person per room).")
    st.progress(min(max(adcom_pct_capacity/100.0, 0.0), 1.0),
                text=f"{adcom_share_used:.1f}% of used • {adcom_pct_capacity:.1f}% of capacity")

# Optional: summary table
with st.expander("Regular vs Adcom Assignments"):
    df_group = pd.DataFrame([
        {"Group": "Regular", "Rooms Used": reg_rooms_used,
         "Share of Used (%)": round(reg_share_used, 1),
         "Pct of Capacity (%)": round(reg_pct_capacity, 1)},
        {"Group": "Adcom", "Rooms Used": adcom_rooms_used,
         "Share of Used (%)": round(adcom_share_used, 1),
         "Pct of Capacity (%)": round(adcom_pct_capacity, 1)},
    ])
    st.dataframe(_arrow_safe_scan_df(df_group), width='stretch')

# Persist run history with DEFAULT LIMITS snapshot
if run_clicked:
    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.run_history.append({
        "timestamp": ts_now,
        "filled": total_rooms_used,
        "capacity": total_capacity,
        "pct": pct_filled,
        # Store default limits used for this run
        "defaults": {
            "reg_max_daily": int(reg_max_daily),
            "reg_max_total": int(reg_max_total),
            "reg_min_total": int(reg_min_total),
            "sen_max_daily": int(sen_max_daily),
            "sen_max_total": int(sen_max_total),
            "sen_min_total": int(sen_min_total),
        }
    })
    st.session_state.run_history = st.session_state.run_history[-50:]  # keep last 50

# Top-line metrics
prev = st.session_state.run_history[-2] if len(st.session_state.run_history) >= 2 else None
delta_filled = None if not prev else total_rooms_used - prev["filled"]
delta_pct = None if not prev else pct_filled - prev["pct"]

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Rooms filled", value=total_rooms_used,
              delta=(None if delta_filled is None else f"{delta_filled:+d}"))
with m2:
    st.metric("Total rooms (capacity)", value=total_capacity)
with m3:
    st.metric("Percent filled", value=f"{pct_filled:.1f}%",
              delta=(None if delta_pct is None else f"{delta_pct:+.1f}%"))

# Current progress
st.progress(min(max(pct_filled / 100.0, 0.0), 1.0))

# ---------- Run history table (with Default limits) ----------
if st.session_state.run_history:
    rows = []
    for idx, item in enumerate(st.session_state.run_history, start=1):
        d = item.get("defaults", {})
        rows.append({
            "Run #": idx,
            "Timestamp": item["timestamp"],
            "Filled": item["filled"],
            "Capacity": item["capacity"],
            "Percent Filled": round(item["pct"], 1),
            "Reg Max/Day": d.get("reg_max_daily"),
            "Reg Max Total": d.get("reg_max_total"),
            "Reg Min Total": d.get("reg_min_total"),
            "Adcom Max/Day": d.get("sen_max_daily"),
            "Adcom Max Total": d.get("sen_max_total"),
            "Adcom Min Total": d.get("sen_min_total"),
        })
    df_hist = pd.DataFrame(rows)
    # Sort newest first by Timestamp
    try:
        df_hist["Timestamp_dt"] = pd.to_datetime(df_hist["Timestamp"])
        df_hist = df_hist.sort_values("Timestamp_dt", ascending=False).drop(columns=["Timestamp_dt"])
    except Exception:
        df_hist = df_hist.sort_values("Run #", ascending=False)
else:
    # ensure df_hist exists before plotting
    df_hist = pd.DataFrame(columns=["Run #", "Timestamp", "Filled", "Capacity", "Percent Filled"])

st.markdown("### Scheduler Results")

# --- Prep tidy run-history data ---
df_plot = (
    df_hist.copy()
    .assign(
        **{
            "Run #": pd.to_numeric(df_hist["Run #"], errors="coerce"),
            "Percent Filled": pd.to_numeric(df_hist["Percent Filled"], errors="coerce"),
        }
    )
    .dropna(subset=["Run #", "Percent Filled"])
    .sort_values("Run #")
    .reset_index(drop=True)
)

if df_plot.empty:
    st.info("No runs yet — hit **Run scheduler** to start building your progress history.")
else:
    # Personal-best tracking
    df_plot["Best So Far"] = df_plot["Percent Filled"].cummax().round(1)
    df_plot["Is PB"] = df_plot["Percent Filled"].eq(df_plot["Best So Far"])
    df_plot["Δ % (pts)"] = df_plot["Percent Filled"].diff().round(1)

    current_pct = float(df_plot.iloc[-1]["Percent Filled"])
    prev_best = float(df_plot["Percent Filled"][:-1].max()) if len(df_plot) > 1 else -float("inf")
    new_pb = current_pct > prev_best

    # Celebrate a new personal best
    last_attempt = int(df_plot.iloc[-1]["Run #"])
    last_pct_10 = int(round(float(df_plot.iloc[-1]["Percent Filled"]) * 10))  # tenths to avoid float noise
    pb_key = (last_attempt, last_pct_10)

    if new_pb and st.session_state.get("celebrated_pb_key") != pb_key:
        st.toast(f"New personal best: {df_plot.iloc[-1]['Percent Filled']:.1f}% filled! 🎉", icon="🎯")
        st.session_state["celebrated_pb_key"] = pb_key

    # ==== 1) Personal-best ladder (current vs best-so-far) ====
    st.markdown("#### Results")
    base = alt.Chart(df_plot).properties(
        width="container",
        height=240,
    )

    line_current = base.mark_line(point=True).encode(
        x=alt.X("Run #:O", title="Attempt #"),  # ordinal → whole-number ticks
        y=alt.Y("Percent Filled:Q", title="% filled", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("Run #:Q", title="Attempt #"),
            alt.Tooltip("Timestamp:N"),
            alt.Tooltip("Percent Filled:Q", format=".1f", title="% filled"),
            alt.Tooltip("Δ % (pts):Q", format="+.1f"),
            alt.Tooltip("Filled:Q"),
            alt.Tooltip("Capacity:Q"),
        ],
    )

    step_best = base.mark_line(interpolate="step-after", strokeDash=[6, 4]).encode(
        x=alt.X("Run #:O", title="Attempt #"),
        y=alt.Y("Best So Far:Q", title="% filled"),
        tooltip=[alt.Tooltip("Run #:Q"), alt.Tooltip("Best So Far:Q", format=".1f")],
    )

    confetti = base.transform_filter(alt.datum["Is PB"] == True).mark_text(
        text="🎉", dy=-12, size=16
    ).encode(
        x="Run #:O",
        y="Best So Far:Q",
    )

    st.altair_chart((step_best + line_current + confetti))

    # (optional) keep the raw table tucked away
    with st.expander("Show Run History Table"):
        st.dataframe(_arrow_safe_scan_df(df_hist), width='stretch')

with st.expander("Time Slot Results"):
    df_slots_summary = pd.DataFrame([
        {"Date_Time": t, "Used_Rooms": rooms_used_by_t[t], "Capacity": int(cap_map.get(t, 0))}
        for t in slot_ids
    ])

    # Add unused rooms (never negative) and optional percent
    df_slots_summary["Unused_Rooms"] = (df_slots_summary["Capacity"] - df_slots_summary["Used_Rooms"]).clip(lower=0)
    df_slots_summary["Unused_%"] = np.where(
        df_slots_summary["Capacity"] > 0,
        100.0 * df_slots_summary["Unused_Rooms"] / df_slots_summary["Capacity"],
        np.nan
    )

    # Controls
    col1, col2 = st.columns([1, 3])
    with col1:
        only_unused = st.checkbox("Show only non-full slots", value=True)
    with col2:
        st.caption("Rows with unused rooms are highlighted and sorted to the top.")

    # Filter (optional)
    view = df_slots_summary.copy()
    if only_unused:
        view = view[view["Unused_Rooms"] > 0]

    # Sort: unused first, then Date_Time
    view = view.sort_values(["Unused_Rooms", "Date_Time"], ascending=[False, True]).reset_index(drop=True)

    # Clean types for Arrow, then style
    safe = _arrow_safe_scan_df(view, max_rows=None)

    def _hl_unused(row):
        # soft amber background for rows with any unused rooms
        if row.get("Unused_Rooms", 0) > 0:
            return ["background-color: rgba(255, 193, 7, 0.25)"] * len(row)
        return [""] * len(row)

    styled = (
        safe.style
            .apply(_hl_unused, axis=1)
            .format({"Unused_%": "{:.1f}%"})
    )

    st.dataframe(styled, width='stretch')

# ----------------
# Detailed tables
# ----------------
by_slot = collections.defaultdict(list)
for (i, t), v in assign.items():
    if v:
        by_slot[t].append(i)
slot_df = pd.DataFrame([
    {"slot_id": t, "assigned": ", ".join(sorted(v)), "#people": len(v), "pairs": pairs.get(t, 0)}
    for t, v in by_slot.items()
]).sort_values(["slot_id"]) if by_slot else pd.DataFrame(columns=["slot_id","assigned","#people","pairs"])

by_i = collections.defaultdict(list)
for (i, t), v in assign.items():
    if v:
        by_i[i].append(t)
iv_df = pd.DataFrame([
    {"interviewer": i, "count_model": len(ts), "slots": ", ".join(sorted(ts))}
    for i, ts in by_i.items()
]).sort_values(["interviewer"]) if by_i else pd.DataFrame(columns=["interviewer","count_model","slots"])

# --- one collapsed section with tabs ---
with st.expander("Interview Assignment Details", expanded=False):
    tabs = st.tabs(["By slot", "By interviewer"])
    with tabs[0]:
        st.dataframe(_arrow_safe_scan_df(slot_df), width='stretch')
        # Optional: CSV export for this view
        st.download_button(
            "⬇️ Download 'By slot' CSV",
            slot_df.to_csv(index=False).encode("utf-8"),
            file_name="assignments_by_slot.csv",
            mime="text/csv",
            key="dl_by_slot_csv",
        )
    with tabs[1]:
        st.dataframe(_arrow_safe_scan_df(iv_df), width='stretch')
        # Optional: CSV export for this view
        st.download_button(
            "⬇️ Download 'By interviewer' CSV",
            iv_df.to_csv(index=False).encode("utf-8"),
            file_name="assignments_by_interviewer.csv",
            mime="text/csv",
            key="dl_by_interviewer_csv",
        )
