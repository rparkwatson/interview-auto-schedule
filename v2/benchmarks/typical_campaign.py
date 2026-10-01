"""Repeatable synthetic scale check. Run from the repository root as a module."""

import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from zoneinfo import ZoneInfo

from v2.interview_scheduler_v2 import (
    Interviewer, InterviewerGroup, SchedulerConfig, SchedulingProblem, Slot,
)
from v2.interview_scheduler_v2.optimization import solve


def campaign() -> SchedulingProblem:
    first = datetime(2026, 3, 1, 8, tzinfo=ZoneInfo("America/New_York"))
    slots = []
    for index in range(150):
        start = first + timedelta(days=index // 5, hours=2 * (index % 5))
        slots.append(Slot(f"slot-{index:03}", start, start + timedelta(minutes=90), 3, 3))
    people = tuple(
        Interviewer.create(
            name=f"Synthetic Person {index:02}",
            group=InterviewerGroup.STUDENT if index < 40 else InterviewerGroup.ADCOM,
            available_slot_ids=[
                slot.id for position, slot in enumerate(slots)
                if (index * 7 + position * 11) % 5 != 0
            ],
        )
        for index in range(65)
    )
    return SchedulingProblem(people, tuple(slots))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()
    problem = campaign()
    config = SchedulerConfig(time_limit_seconds=args.time_limit)
    result = solve(problem, scenario="Synthetic 65 people / 150 periods", config=config)
    print(json.dumps({
        "status": result.status.value,
        "interviewers": len(problem.interviewers),
        "slots": len(problem.slots),
        "assignments": len(result.assignments),
        "minimum_shortfalls": len(result.minimum_shortfalls),
        "target_deficit": sum(item.target_deficit for item in result.slot_summaries),
        "back_to_back_pairs": sum(item.back_to_back_pairs for item in result.interviewer_summaries),
        "wall_time_seconds": round(result.wall_time_seconds, 3),
        "time_limit_seconds": args.time_limit,
        "stages": json.loads(result.settings["solver_stages_json"]),
        "input_sha256": result.settings["input_sha256"],
    }, indent=2))
    assert result.succeeded, result.message
    people = {person.id: person for person in problem.interviewers}
    counts = Counter(item.slot_id for item in result.assignments)
    assert all(counts[slot.id] <= slot.capacity for slot in problem.slots)
    assert all(item.slot_id in people[item.interviewer_id].available_slot_ids
               for item in result.assignments)
    for item in result.interviewer_summaries:
        assert item.minimum <= item.cumulative_total <= item.maximum
        assert item.maximum_assigned_on_day <= item.max_per_day


if __name__ == "__main__":
    main()
