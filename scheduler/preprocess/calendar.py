from __future__ import annotations
from typing import Iterable, Dict, List, Tuple
from ..domain import Slot

def build_adjacency(slots: Iterable[Slot], grace_min: int = 0) -> Dict[str, Tuple[str, ...]]:
    by_day: Dict[str, List[Slot]] = {}
    for s in slots:
        by_day.setdefault(s.day_key, []).append(s)
    for day in by_day:
        by_day[day].sort(key=lambda s: (s.start, s.end))
    nexts: Dict[str, Tuple[str, ...]] = {}
    for day, seq in by_day.items():
        for i, s in enumerate(seq[:-1]):
            n = seq[i+1]
            gap = int((n.start - s.end).total_seconds() // 60)
            if gap == 0 or (grace_min and 0 <= gap <= grace_min):
                nexts[s.id] = (n.id,)
            else:
                nexts[s.id] = tuple()
        if seq:
            nexts[seq[-1].id] = tuple()
    return nexts
