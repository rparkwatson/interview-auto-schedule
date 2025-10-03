
from __future__ import annotations
import re
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta
from typing import Dict, List
from ..domain import Interviewer, Slot, Inputs

KIND_OK = {"Regular", "Senior", "Observer"}

def read_inputs_from_excel(file_like) -> Inputs:
    xls = pd.ExcelFile(file_like)
    if set(xls.sheet_names) >= {"People","Availability","Slots"}:
        return _read_simple(xls)
    if set(xls.sheet_names) & {"Max_Pairs_Per_Slot","Master_Availability_Sheet","Adcom_Availability"}:
        raise RuntimeError("This workbook looks like the legacy format. Use read_inputs_from_legacy() instead.")
    raise ValueError("Unrecognized workbook format.")

def read_inputs_from_legacy(file_like, *, year:int, slot_minutes:int,
                            regular_sheet:str="Master_Availability_Sheet",
                            senior_sheet:str="Adcom_Availability",
                            max_pairs_sheet:str="Max_Pairs_Per_Slot",
                            defaults:dict|None=None) -> Inputs:
    xls = pd.ExcelFile(file_like)
    if max_pairs_sheet not in xls.sheet_names:
        raise ValueError(f"Missing sheet: {max_pairs_sheet}")
    if regular_sheet not in xls.sheet_names:
        raise ValueError(f"Missing sheet: {regular_sheet}")
    if senior_sheet not in xls.sheet_names:
        raise ValueError(f"Missing sheet: {senior_sheet}")

    mpps = pd.read_excel(xls, max_pairs_sheet).rename(columns=str)
    mpps.columns = [c.strip() for c in mpps.columns]
    if not {"Date_Time","Max_Pairs"}.issubset(set(mpps.columns)):
        raise ValueError("Max_Pairs_Per_Slot must have columns Date_Time, Max_Pairs")

    dt_rows = [str(x).strip() for x in mpps["Date_Time"].tolist()]
    cap_rows = [int(x) for x in mpps["Max_Pairs"].fillna(0).tolist()]
    slots: list[Slot] = []
    max_pairs: dict[str,int] = {}
    for dt_label, cap in zip(dt_rows, cap_rows):
        sid = dt_label
        start = _parse_dt_label(dt_label, year)
        end = start + timedelta(minutes=int(slot_minutes))
        day_key = start.date().isoformat()
        slots.append(Slot(id=sid, start=start, end=end, day_key=day_key, adjacent_forward=frozenset()))
        max_pairs[sid] = int(cap)

    reg_df = pd.read_excel(xls, regular_sheet).rename(columns=str)
    sen_df = pd.read_excel(xls, senior_sheet).rename(columns=str)
    for df in (reg_df, sen_df):
        df.columns = [c.strip() for c in df.columns]

    avail_cols = [c for c in reg_df.columns if c not in {"Interviewer_Name","Pre_Assigned_Count"}]
    missing = [c for c in avail_cols if c not in max_pairs]
    if missing:
        raise ValueError(f"Availability columns not in Max_Pairs_Per_Slot: {missing[:5]}{'...' if len(missing)>5 else ''}")

    defaults = defaults or {}
    reg_max_daily = int(defaults.get("reg_max_daily", 4))
    reg_max_total = int(defaults.get("reg_max_total", 7))
    reg_min_total = int(defaults.get("reg_min_total", 0))
    sen_max_daily = int(defaults.get("senior_max_daily", 4))
    sen_max_total = int(defaults.get("senior_max_total", 7))
    sen_min_total = int(defaults.get("senior_min_total", 0))  # NEW: default Adcom minimum

    ivs: list[Interviewer] = []
    def _avail_slots(row) -> frozenset[str]:
        return frozenset([c for c in avail_cols if str(row.get(c,0)).strip() not in {"0","0.0","","nan","NaN"}])

    for _, r in reg_df.iterrows():
        name = str(r.get("Interviewer_Name","?")).strip()
        pre = int(r.get("Pre_Assigned_Count", 0) or 0)
        ivs.append(Interviewer(
            id=name, name=name, kind="Regular",
            max_daily=reg_max_daily, max_total=reg_max_total,
            min_total=reg_min_total, pre_assigned=pre,
            available_slots=_avail_slots(r)
        ))

    for _, r in sen_df.iterrows():
        name = str(r.get("Interviewer_Name","?")).strip()
        pre = int(r.get("Pre_Assigned_Count", 0) or 0)
        ivs.append(Interviewer(
            id=name, name=name, kind="Senior",
            max_daily=sen_max_daily, max_total=sen_max_total,
            min_total=sen_min_total, pre_assigned=pre,   # NEW: apply min_total to Seniors
            available_slots=_avail_slots(r)
        ))

    return Inputs(interviewers=ivs, slots=slots, max_pairs_per_slot=max_pairs)

# helpers
def _parse_dt_label(label: str, year:int) -> datetime:
    label = label.strip()
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})-(\d{1,2})(\d{2})?(AM|PM)", label, flags=re.I)
    if not m:
        try:
            return parser.parse(f"{label} {year}")
        except Exception:
            raise ValueError(f"Unrecognized Date_Time label: {label}")
    month = int(m.group(1)); day = int(m.group(2))
    hour = int(m.group(3)); minute = int(m.group(4) or 0)
    ampm = m.group(5).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    if ampm == "AM" and hour == 12:
        hour = 0
    return datetime(year, month, day, hour, minute)

def _read_simple(xls: pd.ExcelFile) -> Inputs:
    people = pd.read_excel(xls, "People").fillna(0)
    avail = pd.read_excel(xls, "Availability")
    slots = pd.read_excel(xls, "Slots")
    people["kind"] = people["kind"].astype(str)
    bad = set(people["kind"]) - KIND_OK
    if bad:
        raise ValueError(f"Unknown kinds: {bad}")
    # slots
    slot_objs: list[Slot] = []
    for _, r in slots.iterrows():
        sid = str(r["slot_id"]).strip()
        start = parser.parse(str(r["start"]))
        end = parser.parse(str(r["end"]))
        day_key = str(r.get("day", start.date()))
        slot_objs.append(Slot(id=sid, start=start, end=end, day_key=str(day_key), adjacent_forward=frozenset()))
    # availability
    avail = avail.astype({"id": str, "slot_id": str})
    avail_map = avail.groupby("id")["slot_id"].apply(lambda s: frozenset(map(str, s))).to_dict()
    def _slots_for(iid: str):
        return avail_map.get(str(iid), frozenset())
    # people
    ivs: list[Interviewer] = []
    for _, r in people.iterrows():
        ivs.append(Interviewer(
            id=str(r["id"]).strip(),
            name=str(r.get("name", r["id"])),
            kind=str(r["kind"]).strip(),
            max_daily=int(r.get("max_daily", 9999)),
            max_total=int(r.get("max_total", 9999)),
            available_slots=_slots_for(str(r["id"])),
            min_total=int(r.get("min_total", 0)),
            pre_assigned=int(r.get("pre_assigned", 0)),
        ))
    max_pairs = {str(r["slot_id"]): int(r.get("max_pairs", 0)) for _, r in slots.iterrows()}
    return Inputs(interviewers=ivs, slots=slot_objs, max_pairs_per_slot=max_pairs)
