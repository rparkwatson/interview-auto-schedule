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
        # Soft spacing indicators (optional)
        self.b2b: Dict[Tuple[str, str, str], cp_model.IntVar] = {}
        self.run3: Dict[Tuple[str, str, str, str], cp_model.IntVar] = {}


class _ObjectiveTerms:
    def __init__(self):
        self.total_pairs: cp_model.LinearExpr = 0
        self.total_adcom: cp_model.LinearExpr = 0
        self.total_fill_reg: cp_model.LinearExpr = 0
        self.total_b2b: cp_model.LinearExpr = 0
        self.total_run3: cp_model.LinearExpr = 0
        self.reg_spread: cp_model.LinearExpr = 0
        self.senior_spread: cp_model.LinearExpr = 0
        self.jitter_obj: cp_model.LinearExpr = 0


def _kind(iv) -> str:
    """Normalize interviewer kind for robust classification."""
    return (getattr(iv, "kind", "") or "").strip().lower()


def _build_model(inputs: Inputs, cfg: Settings) -> tuple[cp_model.CpModel, _Vars, _ObjectiveTerms]:
    m = cp_model.CpModel()
    V = _Vars()
    O = _ObjectiveTerms()

    # --- Build stable, seeded orders to avoid alphabetical bias in variable creation ---
    seed_val = int(getattr(cfg, "random_seed", getattr(cfg, "seed", 0)) or 0)
    rng = random.Random(seed_val)

    I = [iv.id for iv in inputs.interviewers]
    T = [s.id for s in inputs.slots]

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
        V.P[t] = m.NewIntVar(0, cap, f"P_{t}")
        V.A[t] = m.NewIntVar(0, cap, f"A_{t}")

    # Availability
    for iv in inputs.interviewers:
        for t in T:
            if t not in iv.available_slots:
                m.Add(V.x[(iv.id, t)] == 0)

    # Slot-level counts and capacity
    for t in T:
        cap = int(inputs.max_pairs_per_slot.get(t, 0))
        m.Add(sum(V.x[(i, t)] for i in regs) == 2 * V.P[t])
        m.Add(sum(V.x[(i, t)] for i in seniors) == V.A[t])
        m.Add(V.P[t] + V.A[t] <= cap)

        if getattr(cfg, "observer_extra_per_slot", -1) >= 0:
            m.Add(sum(V.x[(i, t)] for i in observers) <= cfg.observer_extra_per_slot)

    # Per-interviewer totals and per-day caps
    pre_asg = {iv.id: iv.pre_assigned for iv in inputs.interviewers}
    day_slots = defaultdict(list)
    for s in inputs.slots:
        day_slots[s.day_key].append(s.id)

    for iv in inputs.interviewers:
        total_i = sum(V.x[(iv.id, t)] for t in T)
        m.Add(total_i + pre_asg[iv.id] <= iv.max_total)
        if iv.min_total > 0:
            m.Add(total_i + pre_asg[iv.id] >= iv.min_total)
        for day, ts in day_slots.items():
            m.Add(sum(V.x[(iv.id, t)] for t in ts) <= iv.max_daily)

    if getattr(cfg, "day_caps", None):
        for day, cap in cfg.day_caps.items():
            if day in day_slots:
                m.Add(sum(V.x[(i, t)] for i in I for t in day_slots[day]) <= int(cap))

    fwd = {s.id: tuple(s.adjacent_forward) for s in inputs.slots}

    # Back-to-back adjacency
    if getattr(cfg, "back_to_back_mode", "off") != "off":
        if cfg.back_to_back_mode == "hard":
            for i in I:
                for t in T:
                    for nxt in fwd[t]:
                        m.Add(V.x[(i, t)] + V.x[(i, nxt)] <= 1)
        else:  # soft
            for i in I:
                for t in T:
                    for nxt in fwd[t]:
                        v = m.NewBoolVar(f"b2b_{i}_{t}_{nxt}")
                        V.b2b[(i, t, nxt)] = v
                        m.Add(v >= V.x[(i, t)] + V.x[(i, nxt)] - 1)

    # No-three-in-a-row constraints
    if getattr(cfg, "no_three_in_row_mode", "off") != "off":
        for i in I:
            for t in T:
                for nxt in fwd[t]:
                    for nxt2 in fwd.get(nxt, tuple()):
                        if cfg.no_three_in_row_mode == "hard":
                            m.Add(V.x[(i, t)] + V.x[(i, nxt)] + V.x[(i, nxt2)] <= 2)
                        else:
                            r = m.NewBoolVar(f"run3_{i}_{t}_{nxt}_{nxt2}")
                            V.run3[(i, t, nxt, nxt2)] = r
                            m.Add(r >= V.x[(i, t)] + V.x[(i, nxt)] + V.x[(i, nxt2)] - 2)

    # Objective terms
    O.total_pairs = sum(V.P[t] for t in T)
    O.total_adcom = sum(V.A[t] for t in T)

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
    O.total_fill_reg = sum(fill_reg_terms) if fill_reg_terms else 0

    O.total_b2b = sum(V.b2b.values()) if V.b2b else 0
    O.total_run3 = sum(V.run3.values()) if V.run3 else 0

    def _spread_penalty(ids: set[str], label: str) -> cp_model.LinearExpr:
        if len(ids) <= 1:
            return 0
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

    O.reg_spread = _spread_penalty(regs, "reg")
    O.senior_spread = _spread_penalty(seniors, "senior")

    w_tiebreak = float(getattr(cfg, "w_tiebreak", 1e-6))
    iv_jit = {i: rng.random() for i in I}
    slot_jit = {t: rng.random() for t in T}
    O.jitter_obj = sum((iv_jit[i] + slot_jit[t]) * V.x[(i, t)] for i in I for t in T) * w_tiebreak

    return m, V, O


def _set_weighted_objective(m: cp_model.CpModel, cfg: Settings, O: _ObjectiveTerms) -> None:
    m.Maximize(
        cfg.w_pairs * O.total_pairs
        + cfg.w_fill_adcom * O.total_adcom
        + O.total_fill_reg
        - cfg.w_b2b * O.total_b2b
        - cfg.w_run3 * O.total_run3
        - cfg.w_fair_reg_spread * O.reg_spread
        - cfg.w_fair_senior_spread * O.senior_spread
        + O.jitter_obj
    )


def _new_solver(cfg: Settings) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    seed_val = int(getattr(cfg, "random_seed", getattr(cfg, "seed", 0)) or 0)
    solver.parameters.random_seed = seed_val
    if getattr(cfg, "threads", None):
        solver.parameters.num_search_workers = cfg.threads
    solver.parameters.max_time_in_seconds = cfg.time_limit_s
    return solver


def _safe_value(solver: cp_model.CpSolver, expr: cp_model.LinearExpr) -> int:
    return int(round(float(solver.Value(expr))))


def solve_weighted(inputs: Inputs, cfg: Settings, hint: Dict[tuple[str, str], int] | None = None) -> SolveResult:
    m, V, O = _build_model(inputs, cfg)
    strategy = getattr(cfg, "objective_strategy", "lexicographic")

    if strategy == "weighted":
        _set_weighted_objective(m, cfg, O)
        solver = _new_solver(cfg)
        if hint:
            try:
                solver.SetHint([V.x[k] for k in hint.keys()], [int(v) for v in hint.values()])
            except Exception:
                pass
        res = solver.Solve(m)
        return _extract_solution(inputs, cfg, solver, V, res)

    # Lexicographic multi-pass
    # 1) maximize regular pairs
    m.Maximize(O.total_pairs)
    solver1 = _new_solver(cfg)
    if hint:
        try:
            solver1.SetHint([V.x[k] for k in hint.keys()], [int(v) for v in hint.values()])
        except Exception:
            pass
    res1 = solver1.Solve(m)
    if res1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _extract_solution(inputs, cfg, solver1, V, res1)

    best_pairs = _safe_value(solver1, O.total_pairs)
    m.Add(O.total_pairs == best_pairs)

    # 2) maximize secondary fill while preserving pairs optimum
    m.Maximize(cfg.w_fill_adcom * O.total_adcom + O.total_fill_reg)
    solver2 = _new_solver(cfg)
    try:
        solver2.SetHint([V.x[(i, t)] for i, t in V.x.keys()], [int(solver1.Value(V.x[(i, t)])) for i, t in V.x.keys()])
    except Exception:
        pass
    res2 = solver2.Solve(m)
    if res2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return _extract_solution(inputs, cfg, solver2, V, res2)

    best_secondary = _safe_value(solver2, cfg.w_fill_adcom * O.total_adcom + O.total_fill_reg)
    m.Add(cfg.w_fill_adcom * O.total_adcom + O.total_fill_reg == best_secondary)

    # 3) minimize spacing/fairness (as maximize negative penalty) with tiny seeded jitter tie-break
    m.Maximize(
        - cfg.w_b2b * O.total_b2b
        - cfg.w_run3 * O.total_run3
        - cfg.w_fair_reg_spread * O.reg_spread
        - cfg.w_fair_senior_spread * O.senior_spread
        + O.jitter_obj
    )
    solver3 = _new_solver(cfg)
    try:
        solver3.SetHint([V.x[(i, t)] for i, t in V.x.keys()], [int(solver2.Value(V.x[(i, t)])) for i, t in V.x.keys()])
    except Exception:
        pass
    res3 = solver3.Solve(m)
    return _extract_solution(inputs, cfg, solver3, V, res3)


def _extract_solution(inputs: Inputs, cfg: Settings, solver: cp_model.CpSolver, V: _Vars, res: int) -> SolveResult:
    status = solver.StatusName(res)
    I = [iv.id for iv in inputs.interviewers]
    T = [s.id for s in inputs.slots]

    feasible = res in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assign = {(i, t): int(solver.Value(V.x[(i, t)])) for i in I for t in T} if feasible else {}
    pairs = {t: int(solver.Value(V.P[t])) for t in T} if feasible else {}
    adcom = {t: int(solver.Value(V.A[t])) for t in T} if feasible else {}
    b2b = {k: int(solver.Value(v)) for k, v in V.b2b.items()} if (feasible and V.b2b) else {}
    run3 = {k: int(solver.Value(v)) for k, v in V.run3.items()} if (feasible and V.run3) else {}

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
        "run3": run3,
        "gap": gap,
    }
