from __future__ import annotations
from typing import Iterable, Dict, List, Tuple
from ..domain import Slot

def build_adjacency(
    slots: Iterable[Slot],
    grace_min: int = 0,
    lookahead_slots: int = 1,
) -> Dict[str, Tuple[str, ...]]:
    by_day: Dict[str, List[Slot]] = {}
    for s in slots:
        by_day.setdefault(s.day_key, []).append(s)
    for day in by_day:
        by_day[day].sort(key=lambda s: (s.start, s.end))
    nexts: Dict[str, Tuple[str, ...]] = {}
    for day, seq in by_day.items():
        for i, s in enumerate(seq):
            fwd: list[str] = []
            for j in range(i + 1, min(len(seq), i + 1 + int(max(1, lookahead_slots)))):
                n = seq[j]
                gap = int((n.start - s.end).total_seconds() // 60)
                if gap == 0 or (grace_min and 0 <= gap <= grace_min):
                    fwd.append(n.id)
                else:
                    break
            nexts[s.id] = tuple(fwd)
    return nexts
