from __future__ import annotations
from typing import Dict, List, Set
from ..domain import Inputs

def greedy_seed(inputs: Inputs) -> Dict[tuple[str, str], int]:
    regs: Set[str] = {iv.id for iv in inputs.interviewers if iv.kind == "Regular"}
    feas_by_t: Dict[str, List[str]] = {
        s.id: [iv.id for iv in inputs.interviewers if s.id in iv.available_slots]
        for s in inputs.slots
    }
    feas_reg_by_t: Dict[str, List[str]] = {
        t: [i for i in feas if i in regs] for t, feas in feas_by_t.items()
    }
    ordered_slots = sorted(inputs.slots, key=lambda s: (len(feas_reg_by_t[s.id]), len(feas_by_t[s.id])))
    used_total = {iv.id: 0 for iv in inputs.interviewers}
    max_total = {iv.id: iv.max_total for iv in inputs.interviewers}
    hint: Dict[tuple[str, str], int] = {}
    for s in ordered_slots:
        cap_pairs = inputs.max_pairs_per_slot.get(s.id, 0)
        if cap_pairs <= 0:
            continue
        feas_regs = [i for i in feas_reg_by_t[s.id] if used_total[i] < max_total[i]]
        take = min(cap_pairs*2, len(feas_regs))
        for i in feas_regs[:take]:
            hint[(i, s.id)] = 1
            used_total[i] += 1
    return hint
