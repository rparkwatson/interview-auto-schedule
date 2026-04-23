
from __future__ import annotations
import os
import csv
import random
from io import StringIO
from typing import Dict, Tuple, List
from datetime import datetime
import pandas as pd
from ..domain import Inputs, Interviewer

def make_assignment_csv(inputs: Inputs, assign: Dict[tuple[str, str], int]) -> str:
    out = StringIO()
    w = csv.writer(out)
    w.writerow(["slot_id", "interviewer_id", "assigned"])
    for (i, t), v in assign.items():
        if v:
            w.writerow([t, i, 1])
    return out.getvalue()

def make_ics(inputs: Inputs, assign: Dict[tuple[str, str], int], tz: str = "UTC") -> str:
    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y%m%dT%H%M%S")
    lines = ["BEGIN:VCALENDAR","VERSION:2.0","PRODID:-//SchedulerApp//EN"]
    slot_by_id = {s.id: s for s in inputs.slots}
    for (i, t), v in assign.items():
        if not v:
            continue
        s = slot_by_id[t]
        uid = f"{i}-{t}@scheduler"
        lines += ["BEGIN:VEVENT",
                  f"UID:{uid}",
                  f"DTSTART;TZID={tz}:{fmt(s.start)}",
                  f"DTEND;TZID={tz}:{fmt(s.end)}",
                  f"SUMMARY:Interview {t} — {i}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\n".join(lines)

# ---- Excel report ----

def _time_key_from_label(label: str) -> str:
    if "-" in label:
        return label.split("-", 1)[1]
    return label

def _date_key_from_label(label: str) -> str:
    if "-" in label:
        return label.split("-", 1)[0]
    return label

def _sort_key(label: str) -> Tuple:
    date = _date_key_from_label(label)
    t = _time_key_from_label(label)
    order = ["8AM", "10:30AM", "1PM", "330PM", "6PM"]
    t_norm = t.replace(":", "").upper()
    map_norm = {"1030AM": "10:30AM", "330PM": "330PM"}
    t_key = map_norm.get(t_norm, t)
    try:
        idx = order.index(t_key)
    except ValueError:
        idx = len(order) + (0 if t_key == t else 1)
    try:
        mm, dd = date.split("/")
        mm = f"{int(mm):02d}"; dd = f"{int(dd):02d}"
        date_key = f"{mm}-{dd}"
    except Exception:
        date_key = date
    return (date_key, idx, t_key)

def make_excel_report(
    inputs: Inputs,
    assign: Dict[Tuple[str, str], int],
    *,
    path: str,
    pair_order_seed: int | None = None
) -> str:
    slot_ids = [s.id for s in inputs.slots]
    iv_by_id: Dict[str, Interviewer] = {iv.id: iv for iv in inputs.interviewers}
    cap_pairs = inputs.max_pairs_per_slot

    # Per-slot assignments split by type
    regs_by_t: Dict[str, List[str]] = {t: [] for t in slot_ids}
    adcom_by_t: Dict[str, List[str]] = {t: [] for t in slot_ids}
    for (i, t), v in assign.items():
        if not v:
            continue
        iv = iv_by_id[i]
        if iv.kind == "Regular":
            regs_by_t[t].append(iv.name)
        elif iv.kind == "Senior":
            adcom_by_t[t].append(iv.name)
        else:
            # Observers are excluded from exported pairing/adcom sheets
            pass

    # Build stable pair tuples per slot from regular assignments, with optional seeded
    # 50/50 intra-pair order swap.
    rng = random.Random(pair_order_seed)
    pairs_by_t: Dict[str, List[Tuple[str, str]]] = {t: [] for t in slot_ids}
    for t in slot_ids:
        regs = regs_by_t[t]  # preserve assignment/insertion order
        for k in range(0, len(regs) - 1, 2):
            a, b = regs[k], regs[k + 1]
            if rng.random() < 0.5:
                a, b = b, a
            pairs_by_t[t].append((a, b))

    # 1) Regular_Interviewers (two rows per pair)
    regular_rows = []
    for t in slot_ids:
        for pair_idx, (a, b) in enumerate(pairs_by_t[t], start=1):
            regular_rows.append({"Date_Time": t, "Pair": pair_idx, "Interview": a, "Interviewer_Type": "Regular"})
            regular_rows.append({"Date_Time": t, "Pair": pair_idx, "Interview": b, "Interviewer_Type": "Regular"})
    df_regular = pd.DataFrame(regular_rows)
    if not df_regular.empty:
        df_regular = df_regular.sort_values(
            by=["Date_Time","Pair"],
            key=lambda col: col.map(_sort_key) if col.name=="Date_Time" else col
    )

    # 2) Adcom_Staff (one row per Adcom assignment; multiple per Date_Time allowed)
    adcom_rows = []
    for t in slot_ids:
        for name in sorted(adcom_by_t[t]):
            adcom_rows.append({"Date_Time": t, "Pair": 1, "Interview": name, "Interviewer_Type": "Adcom"})
    df_adcom = pd.DataFrame(adcom_rows)
    if not df_adcom.empty:
        df_adcom = df_adcom.sort_values(by=["Date_Time","Pair"], key=lambda col: col.map(_sort_key) if col.name=="Date_Time" else col)

    # 3) Unmet_Minimums
    unmet = []
    assigned_count = {iv.id: 0 for iv in inputs.interviewers}
    for (i, t), v in assign.items():
        if v:
            assigned_count[i] += 1
    for iv in inputs.interviewers:
        total_with_pre = assigned_count[iv.id] + iv.pre_assigned
        if iv.min_total and total_with_pre < iv.min_total:
            unmet.append({"Interviewer_Name": iv.name})
    df_unmet = pd.DataFrame(unmet)

    # 4) Total_Assignments
    totals_rows = []
    for iv in inputs.interviewers:
        asg = assigned_count[iv.id]
        total = asg + iv.pre_assigned
        totals_rows.append({
            "Interviewer_Name": iv.name,
            "Interviewer_Type": "Adcom" if iv.kind == "Senior" else ("Regular" if iv.kind == "Regular" else iv.kind),
            "Pre_Assigned_Count": iv.pre_assigned,
            "Total_Assignments": total
        })
    df_totals = pd.DataFrame(totals_rows)

    # 5) Slot_Summary (pairs + adcom singles)
    slots_rows = []
    for t in slot_ids:
        max_slot = int(cap_pairs.get(t, 0))  # directly from upload (Max_Pairs)
        pair_count = len(regs_by_t[t]) // 2
        adcom_count = len(adcom_by_t[t])
        assigned_count_slots = pair_count + adcom_count
        remaining = max_slot - assigned_count_slots
        slots_rows.append({
            "Date_Time": t,
            "Max_Slot": max_slot,
            "Assigned_Count": assigned_count_slots,
            "Remaining_Slots": remaining
        })
    df_slots = pd.DataFrame(slots_rows)
    if not df_slots.empty:
        df_slots = df_slots.sort_values(by=["Date_Time"], key=lambda col: col.map(_sort_key))

    # 6) Assigned_Pairs (mix pairs and adcom singles)
    # Create enough columns to show all rooms: max of (Max_Pairs) across times, but at least 6.
    max_slots_cols = max(max(cap_pairs.values()), 6) if cap_pairs else 6
    col_names = ["Date_Time"] + [f"Slot {i}" for i in range(1, max_slots_cols+1)]
    ap_rows = []
    for t in slot_ids:
        row = {k: "" for k in col_names}
        row["Date_Time"] = t

        # Build ordered room list: pairs first (A & B), then Adcom singles
        pairs = [f"{a} & {b}" for a, b in pairs_by_t[t]]
        adcoms = sorted(adcom_by_t[t])

        rooms = pairs + adcoms  # pairs in Slot 1..p, then adcom singles
        for idx, label in enumerate(rooms[:max_slots_cols], start=1):
            row[f"Slot {idx}"] = label
        ap_rows.append(row)
    df_ap = pd.DataFrame(ap_rows, columns=col_names)
    if not df_ap.empty:
        df_ap = df_ap.sort_values(by=["Date_Time"], key=lambda col: col.map(_sort_key))

    # Consistency check: pair labels reconstructed from Regular_Interviewers must
    # match Assigned_Pairs pair slot labels for each time slot.
    regular_pair_labels_by_t: Dict[str, List[str]] = {t: [] for t in slot_ids}
    for t in slot_ids:
        rows_t = [r for r in regular_rows if r["Date_Time"] == t]
        by_pair: Dict[int, List[str]] = {}
        for r in rows_t:
            by_pair.setdefault(r["Pair"], []).append(r["Interview"])
        for pair_idx in sorted(by_pair):
            names = by_pair[pair_idx]
            if len(names) == 2:
                regular_pair_labels_by_t[t].append(f"{names[0]} & {names[1]}")
    assigned_pair_labels_by_t: Dict[str, List[str]] = {
        t: [f"{a} & {b}" for a, b in pairs_by_t[t]]
        for t in slot_ids
    }
    for t in slot_ids:
        assert regular_pair_labels_by_t[t] == assigned_pair_labels_by_t[t], (
            f"Pair label mismatch at {t}: "
            f"{regular_pair_labels_by_t[t]} != {assigned_pair_labels_by_t[t]}"
        )

    # Write Excel
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_regular.to_excel(writer, sheet_name="Regular_Interviewers", index=False)
        df_adcom.to_excel(writer, sheet_name="Adcom_Staff", index=False)
        df_unmet.to_excel(writer, sheet_name="Unmet_Minimums", index=False)
        df_totals.to_excel(writer, sheet_name="Total_Assignments", index=False)
        df_slots.to_excel(writer, sheet_name="Slot_Summary", index=False)
        df_ap.to_excel(writer, sheet_name="Assigned_Pairs", index=False)
    return path
