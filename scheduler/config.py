from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Optional

class Settings(BaseModel):
    # Solve controls
    time_limit_s: float = Field(120, ge=5)      # wall clock seconds
    threads: int = Field(0, ge=0)               # 0 = auto (all cores)

    # Policy toggles
    back_to_back_mode: str = Field("soft", pattern="^(soft|hard|off)$")
    no_three_in_row_mode: str = Field("off", pattern="^(soft|hard|off)$")
    objective_strategy: str = Field("lexicographic", pattern="^(weighted|lexicographic)$")
    observer_extra_per_slot: int = 0            # allow observers beyond pair capacity

    # Objective weights
    w_pairs: int = 1_000_000
    w_fill: int = 1_000
    w_b2b: int = 1
    w_run3: int = 5
    adjacency_grace_min: int = 0
    adjacency_lookahead_slots: int = Field(1, ge=1, le=4)

    # Optional global day caps: {"YYYY-MM-DD": max_assignments_that_day}
    day_caps: Optional[Dict[str, int]] = None

    # Scarcity + Adcom weighting (used by v4+ objective)
    scarcity_bonus: int = 5                     # bonus per missing Regular on a slot
    w_fill_adcom: int = 500

    # Randomness / tiebreak controls
    random_seed: int = Field(0, ge=0)
    w_tiebreak: float = Field(1e-6, ge=0.0)

    # Fairness controls (penalize spread in total assigned load by interviewer kind)
    w_fair_reg_spread: int = Field(250, ge=0)
    w_fair_senior_spread: int = Field(100, ge=0)
