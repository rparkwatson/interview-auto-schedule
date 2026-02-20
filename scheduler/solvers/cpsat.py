from __future__ import annotations
from typing import Dict, Tuple
from collections import defaultdict
import random
from ortools.sat.python import cp_model
from ..domain import Inputs
from ..config import Settings

SolveResult = Dict[str, object]


class _Vars:
    def __init__(self):
        # Decision vars
        self.x: Dict[Tuple[str, str], cp_model.IntVar] = {}      # assign interviewer i to time t (0/1)
        self.P: Dict[str, cp_model.IntVar] = {}                  # number of Regular PAIRS at time t (rooms used by pairs)
        self.A: Dict[str, cp_model.IntVar] = {}                  # number of Adcom SINGLES at time t (rooms used by adcom)
        # Soft back-to-back indicators (optional)
        self.b2b: Dict[Tuple[str, str, str], cp_model.IntVar] = {}


def _kind(iv) -> str:
    """Normalize interviewer kind for robust classification."""
    return (getattr(iv, "kind", "") or "").strip().lower()


def _build_model(inputs: Inputs, cfg: Settings) -> tuple[cp_model.CpModel, _Vars]:
    m = cp_model.CpModel()
    V = _Vars()

    # --- Build stable, seeded orders to avoid alphabetical bias in variable creation ---
    # Preferred field is random_seed; keep backward-compatible fallback to seed.
    seed_val = int(getattr(cfg, "random_seed", getattr(cfg, "seed", 0)) or 0)
    rng = random.Random(seed_val)

    # Base (deterministic) lists
    I = [iv.id for iv in inputs.interviewers]
    T = [s.id for s in inputs.slots]

    # Deterministic shuffle (same seed -> same order; not alphabetical)
    rng.shuffle(I)
    rng.shuffle(T)

    regs = {iv.id for iv in inputs.interviewers if _kind(iv) == "regular"}
    seniors = {iv.id for iv in inputs.interviewers if _kind(iv) == "senior"}
    observers = {iv.id for iv in inputs.interviewers if _kind(iv) == "observer"}

    # Vars
    for i in I:
        for t in T:
            V.x[(i, t)] = m.NewBoolVar(f"x_{i}_{t}")
    for t in T:
        cap = int(inputs.max_pairs_per_slot.get(t, 0))
        V.P[t] = m.NewIntVar(0, cap, f"P_{t}")  # number of Regular pairs in this time
        V.A[t] = m.NewIntVar(0, cap, f"A_{t}")  # number of Adcom singles in this time

    # Availability
    for iv in inputs.interviewers:
        for t in T:
            if t not in iv.available_slots:
                m.Add(V.x[(iv.id, t)] == 0)

    # Slot-level counts and capacity
    for t in T:
        cap = int(inputs.max_pairs_per_slot.get(t, 0))
        # Exactly 2 Regulars per Regular pair; exactly 1 Adcom per Adcom single
        m.Add(sum(V.x[(i, t)] for i in regs) == 2 * V.P[t])
        m.Add(sum(V.x[(i, t)] for i in seniors) == V.A[t])

        # Room capacity (pairs + adcom singles must fit available rooms)
        m.Add(V.P[t] + V.A[t] <= cap)

        # Observers may be allowed beyond rooms by a configured slack
        if getattr(cfg, "observer_extra_per_slot", -1) >= 0:
            m.Add(sum(V.x[(i, t)] for i in observers) <= cfg.observer_extra_per_slot)

    # Per-interviewer totals and per-day caps (include pre_assigned in totals)
    pre_asg = {iv.id: iv.pre_assigned for iv in inputs.interviewers}
    day_slots = defaultdict(list)
    for s in inputs.slots:
        day_slots[s.day_key].append(s.id)

    for iv in inputs.interviewers:
        total_i = sum(V.x[(iv.id, t)] for t in T)
        # Max total
        m.Add(total_i + pre_asg[iv.id] <= iv.max_total)
        # Min total (now for ALL interviewers, including Adcom)
        if iv.min_total > 0:
            m.Add(total_i + pre_asg[iv.id] >= iv.min_total)
        # Per-day cap
        for day, ts in day_slots.items():
            m.Add(sum(V.x[(iv.id, t)] for t in ts) <= iv.max_daily)

    # Optional global day caps (across everyone)
    if getattr(cfg, "day_caps", None):
        for day, cap in cfg.day_caps.items():
            if day in day_slots:
                m.Add(sum(V.x[(i, t)] for i in I for t in day_slots[day]) <= int(cap))

    # Back-to-back adjacency
    if getattr(cfg, "back_to_back_mode", "off") != "off":
        fwd = {s.id: tuple(s.adjacent_forward) for s in inputs.slots}
        if cfg.back_to_back_mode == "hard":
            for i in I:
                for t in T:
                    for nxt in fwd[t]:
                        m.Add(V.x[(i, t)] + V.x[(i, nxt)] <= 1)
        else:  # "soft"
            for i in I:
                for t in T:
                    for nxt in fwd[t]:
                        v = m.NewBoolVar(f"b2b_{i}_{t}_{nxt}")
                        V.b2b[(i, t, nxt)] = v
                        m.Add(v >= V.x[(i, t)] + V.x[(i, nxt)] - 1)

    # ---------- Objective ----------
    # Primary: maximize Regular pairs (rooms with pairs)
    total_pairs = sum(V.P[t] for t in T)
    # Secondary: maximize Adcom singles (fills unused rooms)
    total_adcom = sum(V.A[t] for t in T)

    # Scarcity-weighted fill for Regulars (prefer slots with fewer feasible Regulars),
    # while still respecting min totals as hard constraints.
    reg_avail_by_t: Dict[str, int] = {}
    for t in T:
        cnt = 0
        for iv in inputs.interviewers:
            if iv.id in regs and t in iv.available_slots:
                cnt += 1
        reg_avail_by_t[t] = cnt
    max_reg_avail = max(reg_avail_by_t.values()) if reg_avail_by_t else 1

    fill_reg_terms = []
    for t in T:
        scarcity = max_reg_avail - reg_avail_by_t[t]
        slot_w = cfg.w_fill * (1 + cfg.scarcity_bonus * scarcity)
        for i in regs:
            fill_reg_terms.append(slot_w * V.x[(i, t)])
    total_fill_reg = sum(fill_reg_terms) if fill_reg_terms else 0

    total_b2b = sum(V.b2b.values()) if V.b2b else 0

    # Fairness: penalize spread in total assigned load (including pre-assigned)
    # within each major interviewer group.
    def _spread_penalty(ids: set[str], label: str) -> cp_model.LinearExpr:
        if len(ids) <= 1:
            return 0

        # Keep conservative bounds for robustness.
        ub = max(int(iv.max_total) for iv in inputs.interviewers if iv.id in ids)
        max_v = m.NewIntVar(0, ub, f"max_load_{label}")
        min_v = m.NewIntVar(0, ub, f"min_load_{label}")

        for iv in inputs.interviewers:
            if iv.id not in ids:
                continue
            load_i = sum(V.x[(iv.id, t)] for t in T) + int(iv.pre_assigned)
            m.Add(max_v >= load_i)
            m.Add(min_v <= load_i)

        return max_v - min_v

    reg_spread = _spread_penalty(regs, "reg")
    senior_spread = _spread_penalty(seniors, "senior")

    # --- Seeded microscopic jitter to break name-order ties without changing priorities ---
    # Keep the jitter coefficient tiny; allow override via cfg.w_tiebreak if desired.
    w_tiebreak = float(getattr(cfg, "w_tiebreak", 1e-6))
    iv_jit = {i: rng.random() for i in I}
    slot_jit = {t: rng.random() for t in T}
    jitter_obj = sum((iv_jit[i] + slot_jit[t]) * V.x[(i, t)] for i in I for t in T)

    m.Maximize(
        cfg.w_pairs * total_pairs
        + cfg.w_fill_adcom * total_adcom
        + total_fill_reg
        - cfg.w_b2b * total_b2b
        - cfg.w_fair_reg_spread * reg_spread
        - cfg.w_fair_senior_spread * senior_spread
        + w_tiebreak * jitter_obj
    )

    return m, V


def solve_weighted(inputs: Inputs, cfg: Settings, hint: Dict[tuple[str, str], int] | None = None) -> SolveResult:
    m, V = _build_model(inputs, cfg)

    solver = cp_model.CpSolver()

    # Seed solver's internal randomness to make runs deterministic for a given seed
    seed_val = int(getattr(cfg, "random_seed", getattr(cfg, "seed", 0)) or 0)
    solver.parameters.random_seed = seed_val

    # Warm-start hint (robust to OR-Tools version)
    if hint:
        try:
            solver.SetHint([V.x[k] for k in hint.keys()], [int(v) for v in hint.values()])
        except Exception:
            pass

    if getattr(cfg, "threads", None):
        solver.parameters.num_search_workers = cfg.threads
    solver.parameters.max_time_in_seconds = cfg.time_limit_s

    res = solver.Solve(m)
    return _extract_solution(inputs, cfg, solver, V, res)


def _extract_solution(inputs: Inputs, cfg: Settings, solver: cp_model.CpSolver, V: _Vars, res: int) -> SolveResult:
    status = solver.StatusName(res)
    I = [iv.id for iv in inputs.interviewers]
    T = [s.id for s in inputs.slots]

    assign = {(i, t): int(solver.Value(V.x[(i, t)])) for i in I for t in T}
    pairs = {t: int(solver.Value(V.P[t])) for t in T}         # number of Regular pairs per slot
    adcom = {t: int(solver.Value(V.A[t])) for t in T}         # number of Adcom singles per slot
    b2b = {k: int(solver.Value(v)) for k, v in V.b2b.items()} if V.b2b else {}

    try:
        best_bound = solver.BestObjectiveBound()
        obj = solver.ObjectiveValue()
        gap = None if obj == 0 else (best_bound - obj) / max(1.0, abs(obj))
    except Exception:
        obj, gap = 0.0, None

    return {
        "status": status,
        "objective": obj,
        "assign": assign,
        "pairs": pairs,
        "adcom_singles": adcom,
        "b2b": b2b,
        "gap": gap,
    }
