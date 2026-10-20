"""Live check: does the real Google Sheet load and clean correctly?

Run from the repo root (Codespace or local) with .streamlit/secrets.toml in place:
    python scripts/check_sheet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from workout_chat.config import missing_secrets  # noqa: E402
from workout_chat.data import FEATURE_COLUMNS, DataLoadError, load_dataframe  # noqa: E402
from workout_chat.schema import WORKOUT_LEVEL_COLUMNS  # noqa: E402


def main() -> int:
    missing = missing_secrets()
    if missing:
        print(f"Missing secrets: {', '.join(missing)}\nSee .streamlit/secrets.toml.example")
        return 1
    try:
        df = load_dataframe()
    except DataLoadError as exc:
        print(f"LOAD FAILED: {exc}")
        return 1

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"Rows: {len(df)}   Columns: {len(df.columns)}")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Dropped rows (no valid Date): {df.attrs['dropped_rows']}")
    print(f"Extra columns beyond the data dictionary: {df.attrs['extra_columns'] or 'none'}")
    print(f"Feature columns missing from the sheet: {df.attrs['missing_feature_columns'] or 'none'}")
    print(f"Rows with every feature blank (comment not processed yet): {df.attrs['unprocessed_rows']}")
    print(f"Invalid feature values turned into NA (column: count): {df.attrs['feature_value_issues'] or 'none'}")
    print("\nDtypes:\n", df.dtypes.to_string())
    print("\nNulls per column:\n", df.isna().sum().to_string())
    for col in ("Gym_Day", "Type"):
        print(f"\n{col} values:", sorted(df[col].dropna().unique()))
    print("\nFeature value counts (<NA> = blank or 'NA'):")
    for col in (c for c in FEATURE_COLUMNS if c in df):
        counts = df[col].value_counts(dropna=False).sort_index()
        print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("\nWorkout-level consistency (each workout = one Date, same value on every set):")
    problems = 0
    for col in (c for c in WORKOUT_LEVEL_COLUMNS if c in df):
        bad = int((df.groupby("Date")[col].nunique(dropna=True) > 1).sum())
        if bad:
            problems += 1
            print(f"  WARNING: {col} differs between sets of the same workout on {bad} date(s)")
    if not problems:
        print("  OK")
    print(f"\nDistinct Exercise values: {df['Exercise'].nunique()}")
    print("\nSample (Comment masked):\n", df.head(5).assign(Comment="[masked]").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
