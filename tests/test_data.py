import pandas as pd
import pytest

from workout_chat.data import CORE_COLUMNS, DataLoadError, clean_dataframe

HEADER = list(CORE_COLUMNS)


def row(date="2025-03-14", set_id="S1", day="Push", ex="Bench Press", typ="Hypertrophy",
        n=1, w=135, reps=8, rpe=8.5, norm=170.2, bw=180.4, comment="felt good"):
    return [date, set_id, day, ex, typ, n, w, reps, rpe, norm, bw, comment]


def serial(ts: str) -> int:
    return (pd.Timestamp(ts) - pd.Timestamp("1899-12-30")).days


def test_types_and_values():
    df = clean_dataframe([HEADER, row(rpe=9.5)])
    assert list(df.columns) == HEADER
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert df["RPE"].iloc[0] == 9.5  # half-steps survive
    assert str(df["Reps"].dtype) == "Int64"
    assert df["Weight"].dtype.kind == "f"


def test_serial_and_text_dates():
    df = clean_dataframe([
        HEADER,
        row(date=serial("2025-03-14"), set_id="a"),
        row(date="2025-03-15", set_id="b"),
        row(date="Mar 16, 2025", set_id="c"),
    ])
    assert df["Date"].tolist() == [pd.Timestamp("2025-03-14"), pd.Timestamp("2025-03-15"), pd.Timestamp("2025-03-16")]


def test_blank_rows_dropped_and_ragged_rows_padded():
    ragged = ["2025-03-14", "S2", "Pull", "Row", "Strength", 1, 100, 5]  # trailing cells missing
    df = clean_dataframe([HEADER, row(), ["", "", "", "", "", "", "", "", "", "", "", ""], ragged])
    assert len(df) == 2
    assert df["RPE"].isna().iloc[1]


def test_header_whitespace_and_blank_header_column():
    header = [" Date "] + HEADER[1:] + [""]
    df = clean_dataframe([header, row() + ["junk"]])
    assert list(df.columns) == HEADER


def test_numeric_strings_with_commas():
    df = clean_dataframe([HEADER, row(w="1,025", reps="8")])
    assert df["Weight"].iloc[0] == 1025
    assert df["Reps"].iloc[0] == 8


def test_blank_comment_is_missing_not_string():
    df = clean_dataframe([HEADER, row(comment="   ")])
    assert df["Comment"].isna().iloc[0]


def test_missing_column_raises_with_name():
    header = [h for h in HEADER if h != "RPE"]
    with pytest.raises(DataLoadError, match="RPE"):
        clean_dataframe([header, ["x"] * len(header)])


def test_duplicate_header_raises():
    with pytest.raises(DataLoadError, match="Duplicate"):
        clean_dataframe([HEADER + ["Reps"], row() + [1]])


def test_extra_columns_kept_and_typed():
    header = HEADER + ["Shoulder_Pain", "Mood_Score", "Notes"]
    df = clean_dataframe([header, row() + [True, 4, "ok"], row(set_id="S2") + [False, "", "fine"]])
    assert df.attrs["extra_columns"] == ["Shoulder_Pain", "Mood_Score", "Notes"]
    assert df["Shoulder_Pain"].dtype == bool
    assert df["Mood_Score"].dtype.kind == "f"
    assert df["Notes"].tolist() == ["ok", "fine"]


def test_unparseable_date_rows_dropped_and_counted():
    df = clean_dataframe([HEADER, row(), row(date="not a date", set_id="bad")])
    assert len(df) == 1
    assert df.attrs["dropped_rows"] == 1


def test_sorted_by_date_stable_within_day():
    df = clean_dataframe([
        HEADER,
        row(date="2025-03-15", set_id="late"),
        row(date="2025-03-14", set_id="first"),
        row(date="2025-03-14", set_id="second"),
    ])
    assert df["Set_ID"].tolist() == ["first", "second", "late"]


def test_header_only_raises():
    with pytest.raises(DataLoadError):
        clean_dataframe([HEADER])
    with pytest.raises(DataLoadError):
        clean_dataframe([])
