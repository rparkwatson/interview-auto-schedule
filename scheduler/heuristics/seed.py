from __future__ import annotations
from typing import Dict, List, Set, Optional
import random
from ..domain import Inputs

def greedy_seed(inputs: Inputs, seed: Optional[int] = None) -> Dict[tuple[str, str], int]:
    """
    Build a warm-start assignment that avoids hidden alphabetical bias.

    - Process scarcer slots first (fewest Regular-feasible people).
    - Break slot-order ties with a seeded random tiebreaker.
    - Within each slot, pick Regulars with the smallest used_total so far;
      break ties with a seeded random tiebreaker.
    - Deterministic by default (seed=0); pass a seed to vary outcomes.
    """
    rng = random.Random(0 if seed is None else int(seed))

    # Normalize group labels just in case ("Senior" vs "Adcom", case, etc.)
    def _kind(iv) -> str:
        return (iv.kind or "").strip().lower()

    regs: Set[str] = {iv.id for iv in inputs.interviewers if _kind(iv) == "regular"}

    # Feasibility maps
    feas_by_t: Dict[str, List[str]] = {
        s.id: [iv.id for iv in inputs.interviewers if s.id in iv.available_slots]
        for s in inputs.slots
    }
    feas_reg_by_t: Dict[str, List[str]] = {
        t: [i for i in feas if i in regs] for t, feas in feas_by_t.items()
    }

    # Stable (seeded) tie-breakers to prevent name-order bias
    slot_jit = {s.id: rng.random() for s in inputs.slots}
    iv_jit   = {iv.id: rng.random() for iv in inputs.interviewers}

    # Scarce slots first; tie-break with jitter so equal-scarcity slots don't fall back to A–Z
    ordered_slots = sorted(
        inputs.slots,
        key=lambda s: (len(feas_reg_by_t[s.id]), len(feas_by_t[s.id]), slot_jit[s.id])
    )

    # Track how many times each interviewer is used (to spread load a bit)
    used_total = {iv.id: 0 for iv in inputs.interviewers}
    # Guard if max_total can be None
    MAX_BIG = 10**9
    max_total = {iv.id: (iv.max_total if getattr(iv, "max_total", None) is not None else MAX_BIG)
                 for iv in inputs.interviewers}

    hint: Dict[tuple[str, str], int] = {}

    for s in ordered_slots:
        cap_pairs = int(inputs.max_pairs_per_slot.get(s.id, 0) or 0)
        if cap_pairs <= 0:
            continue

        # Regulars feasible for this slot and still under their total cap
        feas_regs = [i for i in feas_reg_by_t[s.id] if used_total[i] < max_total[i]]
        if not feas_regs:
            continue

        # Order Regulars by (used_total asc, random tie-break), not by name
        feas_regs.sort(key=lambda i: (used_total[i], iv_jit[i]))

        take = min(cap_pairs * 2, len(feas_regs))
        for i in feas_regs[:take]:
            hint[(i, s.id)] = 1
            used_total[i] += 1

    return hint
