"""Google Sheets -> clean pandas DataFrame.

Streamlit-free on purpose: ``clean_dataframe`` is a pure function that is easy to
test, and caching lives in the UI layer (``app.py``).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from gspread.utils import ValueRenderOption

from .config import SHEET_WORKSHEET, SHEETS_SCOPES, get_secret

# Column types, taken from the data dictionary. schema.py adds the descriptions.
DATE_COLUMNS = ("Date",)
TEXT_COLUMNS = ("Set_ID", "Gym_Day", "Exersize", "Type", "Comment")
INT_COLUMNS = ("Set_Number", "Reps")
# RPE is float: the dictionary calls it an integer but uses half-steps (9.5, 8.5).
FLOAT_COLUMNS = ("Weight", "RPE", "RPE_Norm_Weight", "Body_Weight")

EXPECTED_COLUMNS = (
    "Date", "Set_ID", "Gym_Day", "Exersize", "Type", "Set_Number",
    "Weight", "Reps", "RPE", "RPE_Norm_Weight", "Body_Weight", "Comment",
)

# Google Sheets serial dates (days since 1899-12-30); sane range ~1954-2119.
_SERIAL_MIN, _SERIAL_MAX = 20_000, 80_000


class DataLoadError(Exception):
    """A problem loading or validating the sheet, with a user-readable message."""


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------
def _service_account_info() -> dict[str, Any]:
    info = get_secret("gcp_service_account")
    if not info:
        raise DataLoadError("Missing secret 'gcp_service_account' (the service account JSON as a TOML table).")
    info = dict(info)
    key = info.get("private_key")
    if isinstance(key, str):
        info["private_key"] = key.replace("\\n", "\n")  # tolerate escaped newlines
    return info


def get_client() -> gspread.Client:
    info = _service_account_info()
    try:
        creds = Credentials.from_service_account_info(info, scopes=SHEETS_SCOPES)
    except (ValueError, KeyError) as exc:
        raise DataLoadError(f"The 'gcp_service_account' secret is malformed: {exc}") from exc
    return gspread.authorize(creds)


def fetch_rows() -> list[list[Any]]:
    """Raw cell values (header row first). Dates arrive as serial numbers, numbers as numbers."""
    url = get_secret("GOOGLE_SHEET_URL")
    if not url:
        raise DataLoadError("Missing secret 'GOOGLE_SHEET_URL'.")
    email = _service_account_info().get("client_email", "the service account")
    client = get_client()
    try:
        worksheet = client.open_by_url(url).worksheet(SHEET_WORKSHEET)
        return worksheet.get_all_values(value_render_option=ValueRenderOption.unformatted)
    except gspread.SpreadsheetNotFound as exc:
        raise DataLoadError(
            f"Spreadsheet not found. Check GOOGLE_SHEET_URL and share the sheet (Viewer is enough) with {email}."
        ) from exc
    except gspread.WorksheetNotFound as exc:
        raise DataLoadError(f"The spreadsheet has no tab named '{SHEET_WORKSHEET}' (names are case-sensitive).") from exc
    except gspread.exceptions.APIError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 403:
            raise DataLoadError(f"Permission denied by Google. Share the sheet with {email} and enable the Sheets API.") from exc
        if status == 429:
            raise DataLoadError("Google Sheets rate limit hit. Wait a minute and refresh.") from exc
        raise DataLoadError(f"Google Sheets API error ({status}): {exc}") from exc


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _clean_text(series: pd.Series) -> pd.Series:
    blank = series.map(_is_blank)
    return series.astype(str).str.strip().mask(blank)


def _parse_dates(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.between(_SERIAL_MIN, _SERIAL_MAX))
    from_serial = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
    text = series.where(numeric.isna()).map(lambda v: None if _is_blank(v) else str(v).strip())
    from_text = pd.to_datetime(text, errors="coerce", format="mixed")
    return from_serial.fillna(from_text).dt.normalize()


def _clean_extra(series: pd.Series) -> pd.Series:
    """Best-effort typing for columns we don't know about (e.g. newly added feature columns)."""
    blank = series.map(_is_blank)
    present = series[~blank]
    if present.empty:
        return _clean_text(series)
    if all(isinstance(v, bool) for v in present):
        out = series.mask(blank).astype("boolean")
        return out if blank.any() else out.astype(bool)
    numeric = pd.to_numeric(series.mask(blank), errors="coerce")
    if numeric[~blank].notna().all():
        return numeric
    return _clean_text(series)


def clean_dataframe(rows: list[list[Any]]) -> pd.DataFrame:
    """Turn raw sheet rows (header first) into a typed DataFrame."""
    if not rows:
        raise DataLoadError("The sheet is empty.")

    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    keep = [i for i, h in enumerate(header) if h]  # ignore columns with no header
    names = [header[i] for i in keep]

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise DataLoadError(f"Duplicate column headers: {', '.join(duplicates)}.")
    missing = [c for c in EXPECTED_COLUMNS if c not in names]
    if missing:
        raise DataLoadError(f"Missing expected columns: {', '.join(missing)}. Found: {', '.join(names)}.")

    width = len(header)
    body = [(list(r) + [""] * width)[:width] for r in rows[1:]]  # pad / trim ragged rows
    df = pd.DataFrame([[r[i] for i in keep] for r in body], columns=names, dtype=object)

    df = df[~df.map(_is_blank).all(axis=1)]  # fully blank rows
    n_before = len(df)

    df["Date"] = _parse_dates(df["Date"])
    df = df[df["Date"].notna()]
    dropped = n_before - len(df)

    for col in TEXT_COLUMNS:
        df[col] = _clean_text(df[col])
    for col in FLOAT_COLUMNS:
        numeric = pd.to_numeric(df[col].mask(df[col].map(_is_blank)).map(_strip_commas), errors="coerce")
        df[col] = numeric.astype("float64")
    for col in INT_COLUMNS:
        numeric = pd.to_numeric(df[col].mask(df[col].map(_is_blank)).map(_strip_commas), errors="coerce")
        df[col] = numeric.round().astype("Int64")
    for col in names:
        if col not in EXPECTED_COLUMNS:
            df[col] = _clean_extra(df[col])

    if df.empty:
        raise DataLoadError("The sheet has a header but no rows with a valid Date.")

    ordered = list(EXPECTED_COLUMNS) + [c for c in names if c not in EXPECTED_COLUMNS]
    df = df[ordered].sort_values("Date", kind="stable").reset_index(drop=True)
    df.attrs["dropped_rows"] = dropped
    df.attrs["extra_columns"] = [c for c in names if c not in EXPECTED_COLUMNS]
    return df


def _strip_commas(value: Any) -> Any:
    return value.replace(",", "") if isinstance(value, str) else value


def load_dataframe() -> pd.DataFrame:
    df = clean_dataframe(fetch_rows())
    df.attrs["loaded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df
