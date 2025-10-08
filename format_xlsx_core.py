# format_xlsx_core.py
import io
import re
import logging
from typing import List, Tuple, Dict, Any, Iterable

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# For sorting time portions in a day
TIME_SLOTS_ORDER = ["8AM", "1030AM", "1PM", "330PM", "6PM"]

# Sheets to ignore
EXCLUDED_SHEETS = {"AF Names", "IGNORE Robin", "CHECK OFF WHEN DONE", "EXAMPLE"}

# Normalize excluded names to lowercase once
EXCLUDED_INTERVIEWER_NAMES = {n.strip().lower() for n in {"aaaaa"}}

# ------------------------------- Helpers --------------------------------------
def load_workbook_from_filelike(file_like) -> pd.ExcelFile:
    """Accepts bytes or a file-like and returns a Pandas ExcelFile."""
    if isinstance(file_like, (bytes, bytearray)):
        file_like = io.BytesIO(file_like)
    return pd.ExcelFile(file_like)

def format_sheet_time(sheet_name: str) -> str:
    raw = re.sub(r"[\s:]", "", sheet_name).upper()
    time_map = {
        "800AM": "8AM", "8AM": "8AM",
        "1030AM": "1030AM",
        "100PM": "1PM", "1PM": "1PM",
        "330PM": "330PM",
        "600PM": "6PM", "6PM": "6PM",
    }
    return time_map.get(raw, raw)

def extract_date(col_name: str) -> str:
    match = re.search(r"\((.*?)\)", col_name)
    if not match:
        raise ValueError(f"Date not found in column name: {col_name}")
    return match.group(1).strip()

def sort_date_time_slots(slots: List[Dict[str, Any]]) -> None:
    """Sorts in-place by (month, day, TIME_SLOTS_ORDER index)."""
    def key_fn(row: Dict[str, Any]):
        dt = row["Date_Time"]
        parts = dt.split("-", 1)
        if len(parts) != 2:
            return (9999, 9999, len(TIME_SLOTS_ORDER))
        date_str, time_str = parts
        try:
            month, day = map(int, date_str.split("/"))
        except Exception:
            return (9999, 9999, len(TIME_SLOTS_ORDER))
        try:
            i = TIME_SLOTS_ORDER.index(time_str)
        except ValueError:
            i = len(TIME_SLOTS_ORDER)
        return (month, day, i)
    slots.sort(key=key_fn)

# -------------------------- Core transformation --------------------------------
def extract_date_time_slots(
    workbook: pd.ExcelFile, header_row: int = 3
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Returns:
      - master_availability: [{Date_Time, Available_Interviewers: [names]}]
      - max_interviews_slots: [{Date_Time, Max_Pairs}]
      - all_interviewers: sorted list of unique names
    """
    master_availability = []
    max_interviews_slots = []
    all_interviewers = set()

    for sheet_name in workbook.sheet_names:
        if sheet_name in EXCLUDED_SHEETS:
            logging.info(f"Skipping excluded sheet: {sheet_name}")
            continue

        time_part = format_sheet_time(sheet_name)

        try:
            sheet = workbook.parse(sheet_name, header=header_row)
        except Exception as e:
            logging.warning(f"Failed parsing sheet '{sheet_name}': {e}")
            continue

        date_columns = [c for c in sheet.columns if isinstance(c, str) and "(" in c]
        if not date_columns:
            logging.warning(f"No date columns found in sheet '{sheet_name}'. Skipping.")
            continue

        for date_col in date_columns:
            try:
                date_part = extract_date(date_col)
            except ValueError as e:
                logging.warning(e)
                continue

            date_time = f"{date_part}-{time_part}"
            max_interviews_slots.append({"Date_Time": date_time, "Max_Pairs": None})

            names = (
                sheet[date_col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.title()
                .tolist()
            )

            # filter out excluded names (case-insensitive)
            names = [n for n in names if n.strip().lower() not in EXCLUDED_INTERVIEWER_NAMES]

            all_interviewers.update(names)
            master_availability.append(
                {"Date_Time": date_time, "Available_Interviewers": names}
            )

    sort_date_time_slots(max_interviews_slots)
    return master_availability, max_interviews_slots, sorted(all_interviewers)

def create_master_df(
    all_interviewers: Iterable[str],
    max_interviews_slots: List[Dict[str, Any]],
    master_availability: List[Dict[str, Any]],
) -> pd.DataFrame:
    # Safety: re-filter and sort
    all_interviewers = sorted(
        n for n in all_interviewers
        if str(n).strip().lower() not in EXCLUDED_INTERVIEWER_NAMES
    )

    cols = [s["Date_Time"] for s in max_interviews_slots]
    master_df = pd.DataFrame(0, index=list(all_interviewers), columns=cols, dtype="int8")
    master_df.index.name = "Interviewer_Name"

    for entry in master_availability:
        dt = entry["Date_Time"]
        names = [n for n in entry.get("Available_Interviewers", []) if str(n).strip().lower() not in EXCLUDED_INTERVIEWER_NAMES]
        if not names:
            continue
        # Guard against any names not present in index (just in case)
        valid = [n for n in names if n in master_df.index]
        if valid:
            master_df.loc[valid, dt] = 1

    master_df = master_df.reset_index()
    master_df.insert(1, "Pre_Assigned_Count", 0)
    return master_df

def create_program_info_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Program_Name": "Interview Scheduler Data Sheet"},
            {"Version": "1.0"},
            {"Note": "This workbook contains availability data for interviewers and Adcom. "
                     "This info will be uploaded for scheduling."},
            {"Note": "Date_Time formatting MUST match across all sheets in this document."},
            {"Note": "Ensure 'Max_Pairs_Per_Slot' matches the columns in the availability sheets."},
            {"Note": "Pre_Assigned_Count is how many have already been hosted in prior rounds."},
        ]
    )

def validate_data(master_df: pd.DataFrame, max_df: pd.DataFrame, adcom_df: pd.DataFrame) -> None:
    if master_df.isnull().values.any():
        logging.warning("Master Availability contains nulls.")
    if "Max_Pairs" in max_df.columns and max_df["Max_Pairs"].isnull().any():
        logging.warning("Some Max_Pairs values are not set.")
    if adcom_df.isnull().values.any():
        logging.warning("Adcom Availability contains nulls.")

def _export_to_excel_bytes(
    program_info_df: pd.DataFrame,
    master_df: pd.DataFrame,
    max_df: pd.DataFrame,
    adcom_df: pd.DataFrame,
) -> bytes:
    """Write all 4 sheets to an in-memory .xlsx and return bytes."""
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        program_info_df.to_excel(writer, sheet_name="Program_Info", index=False)
        max_df.to_excel(writer, sheet_name="Max_Pairs_Per_Slot", index=False)
        master_df.to_excel(writer, sheet_name="Master_Availability_Sheet", index=False)
        adcom_df.to_excel(writer, sheet_name="Adcom_Availability", index=False)

        workbook = writer.book
        # Auto-fit columns by header length only (simple, safe)
        for sheet_name, df in {
            "Program_Info": program_info_df,
            "Max_Pairs_Per_Slot": max_df,
            "Master_Availability_Sheet": master_df,
            "Adcom_Availability": adcom_df,
        }.items():
            ws = writer.sheets[sheet_name]
            for c_idx, col in enumerate(df.columns):
                ws.set_column(c_idx, c_idx, max(10, len(str(col)) + 2))

        # Highlight column B in Master & Adcom
        highlight = workbook.add_format({"bg_color": "#C6EFCE"})
        writer.sheets["Master_Availability_Sheet"].set_column("B:B", None, highlight)
        writer.sheets["Adcom_Availability"].set_column("B:B", None, highlight)

    return out.getvalue()

# --------------------------- Public entry points --------------------------------
def parse_primary_and_adcom(
    primary_file_like, adcom_file_like, header_row: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (master_df, max_pairs_df, adcom_df) without writing files.
    """
    # Primary
    p_wb = load_workbook_from_filelike(primary_file_like)
    p_master_avail, p_max_slots, p_all = extract_date_time_slots(p_wb, header_row)
    master_df = create_master_df(p_all, p_max_slots, p_master_avail)
    max_df = pd.DataFrame(p_max_slots)

    # Adcom
    a_wb = load_workbook_from_filelike(adcom_file_like)
    a_master_avail, a_max_slots, a_all = extract_date_time_slots(a_wb, header_row)
    adcom_df = create_master_df(a_all, a_max_slots, a_master_avail)

    return master_df, max_df, adcom_df

def build_formatted_workbook_bytes(
    primary_file_like,
    adcom_file_like,
    header_row: int = 3,
    max_pairs_overrides: Dict[str, int] | None = None,
) -> Tuple[bytes, Dict[str, pd.DataFrame]]:
    """
    High-level: parse → (optionally) apply Max_Pairs overrides → export → return bytes.
    Returns: (xlsx_bytes, {"program_info_df":..., "master_df":..., "max_df":..., "adcom_df":...})
    """
    master_df, max_df, adcom_df = parse_primary_and_adcom(primary_file_like, adcom_file_like, header_row)

    # Ensure Max_Pairs exists and is clean integer without downcast warnings
    if "Max_Pairs" not in max_df.columns:
        max_df["Max_Pairs"] = 0
    max_df["Max_Pairs"] = (
        pd.to_numeric(max_df["Max_Pairs"], errors="coerce")
          .fillna(0)
          .astype("int64")
    )

    # Apply any per-slot overrides
    if max_pairs_overrides:
        # Expect overrides keyed by Date_Time
        max_df = max_df.set_index("Date_Time", drop=False)
        for dt, val in max_pairs_overrides.items():
            if dt in max_df.index:
                try:
                    max_df.at[dt, "Max_Pairs"] = int(val)
                except Exception:
                    pass
        max_df = max_df.reset_index(drop=True)

    program_info_df = create_program_info_df()
    validate_data(master_df, max_df, adcom_df)
    xlsx_bytes = _export_to_excel_bytes(program_info_df, master_df, max_df, adcom_df)

    return xlsx_bytes, {
        "program_info_df": program_info_df,
        "master_df": master_df,
        "max_df": max_df,
        "adcom_df": adcom_df,
    }
