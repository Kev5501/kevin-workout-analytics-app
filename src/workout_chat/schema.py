"""Data dictionary, schema context for the LLMs, and the opening chat greeting.

Nothing here calls an LLM. Everything is derived from the cleaned DataFrame plus
the static data dictionary, so it costs no quota and is fully deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from .data import EXPECTED_COLUMNS

DF_NAME = "df"                  # variable name for generated pandas code
SQL_TABLE_NAME = "workouts"     # table name for generated SQL (same columns as df)
COMMENT_MASK = "[masked]"
MASKED_COLUMNS = ("Comment",)   # never shown to a model in sample rows
CATEGORICAL_COLUMNS = ("Gym_Day", "Type", "Exersize")  # exact values are listed for the coder

ROW_MEANING = "Each row is one set performed in the gym."


@dataclass(frozen=True)
class ColumnDoc:
    name: str
    kind: str    # friendly type, shown to the user
    short: str   # one-liner for the greeting table
    detail: str  # full description, shown to the LLMs


DATA_DICTIONARY: tuple[ColumnDoc, ...] = (
    ColumnDoc("Date", "date", "Date the set was performed",
              "The date the set was performed on."),
    ColumnDoc("Set_ID", "text", "Unique ID of the set",
              "The unique identifier for each set."),
    ColumnDoc("Gym_Day", "text", "Muscle group trained that day",
              "The muscles targeted in the gym session in which the set was performed. 'Lower' and 'Legs' "
              "have different names but are the same gym day (split: Push, Pull, Legs, Rest, Upper, Lower, Rest)."),
    ColumnDoc("Exersize", "text", "Exercise name",
              "The name of the exercise performed. NOTE: the column name is spelled 'Exersize' in the data."),
    ColumnDoc("Type", "text", "Hypertrophy or Strength",
              "Whether the intended outcome of the exercise was Hypertrophy (muscle growth) or Strength gain."),
    ColumnDoc("Set_Number", "integer", "Set number within that exercise today",
              "The set number for a given exercise within the workout. 1 means the first set of that exercise that day."),
    ColumnDoc("Weight", "number (lbs)", "Weight used, in lbs",
              "Weight in lbs used for the set. Dumbbell exercises record the weight per dumbbell; every other "
              "exercise records the total weight."),
    ColumnDoc("Reps", "integer", "Repetitions completed",
              "How many repetitions were done in the set."),
    ColumnDoc("RPE", "number (1-10)", "Rate of perceived exertion",
              "Rate of Perceived Exertion on a 1-10 scale (half steps used). 10 = could do 0 more reps; "
              "9.5 = maybe 1 more rep or slightly more weight; 9 = 1 more rep; 8.5 = definitely 1 more rep, "
              "maybe 2; and so on down the scale."),
    ColumnDoc("RPE_Norm_Weight", "number (lbs)", "Estimated one-rep max for the set",
              "Estimated one-rep max (lbs) for the exercise, computed from the set's weight, reps and RPE. "
              "Lets strength trends be compared fairly across different rep ranges."),
    ColumnDoc("Body_Weight", "number (lbs)", "Body weight that day, in lbs",
              "Body weight in lbs on the day the set was performed."),
    ColumnDoc("Comment", "text", "Notes on the whole workout",
              "Free-text notes about the workout as a whole, NOT per set: every set from the same workout "
              "shares the same comment. Anything not captured by the other columns lives here (injuries, "
              "mood or sentiment, form notes, quality of pump, etc.). Sample rows show this column masked."),
)

_DOC_BY_NAME = {d.name: d for d in DATA_DICTIONARY}


def _dtype_label(series: pd.Series) -> str:
    return str(series.dtype)


def _extra_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _DOC_BY_NAME]


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _fmt_date(ts: pd.Timestamp) -> str:
    return f"{ts:%b} {ts.day}, {ts.year}"  # avoids %-d, which is not portable to Windows


# --------------------------------------------------------------------------
# Context sent to the coder / rewriter / synthesizer
# --------------------------------------------------------------------------
def sample_rows(df: pd.DataFrame, n: int = 5) -> str:
    """The most recent ``n`` rows as CSV, with masked columns hidden."""
    sample = df.tail(n).copy()
    sample["Date"] = sample["Date"].dt.strftime("%Y-%m-%d")
    for col in MASKED_COLUMNS:
        if col in sample:
            sample[col] = sample[col].where(sample[col].isna(), COMMENT_MASK)
    return sample.to_csv(index=False).strip()


def _allowed_values(df: pd.DataFrame, col: str, max_values: int) -> str:
    counts = df[col].value_counts()  # most frequent first
    shown = [str(v) for v in counts.index[:max_values]]
    hidden = len(counts) - len(shown)
    note = f" (+{hidden} more not shown)" if hidden > 0 else ""
    return f"{col} ({len(counts)} distinct, most frequent first): {json.dumps(shown)}{note}"


def build_schema_context(df: pd.DataFrame, n_samples: int = 5, max_values: int = 150) -> str:
    """Everything a code-writing model needs to know about the data. Comment text is never included."""
    lines = [
        ROW_MEANING,
        f"Python: a pandas DataFrame named `{DF_NAME}`. SQL: a table named `{SQL_TABLE_NAME}` with the same columns.",
        f"Rows: {len(df)} | Workouts (distinct dates): {df['Date'].nunique()} | "
        f"Dates: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}",
        "",
        "Columns (name (pandas dtype): meaning):",
    ]
    for col in df.columns:
        doc = _DOC_BY_NAME.get(col)
        lines.append(f"- {col} ({_dtype_label(df[col])}): {doc.detail if doc else '(no description available)'}")

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        lines += ["", "Missing values: " + ", ".join(f"{c}: {n}" for c, n in nulls.items())]

    lines += ["", "Exact values (use these spellings when filtering):"]
    lines += [_allowed_values(df, c, max_values) for c in CATEGORICAL_COLUMNS if c in df]
    lines += ["", f"Sample rows (most recent {min(n_samples, len(df))}; Comment masked):", sample_rows(df, n_samples)]
    return "\n".join(lines)


def router_domain_description() -> str:
    """Short, data-free description used inside the router's typed-question instructions."""
    names = ", ".join(c for c in EXPECTED_COLUMNS if c not in MASKED_COLUMNS)
    return (
        "A personal strength-training log of individual gym sets. Each set records the date, gym day "
        "(push/pull/legs/upper/lower), exercise, hypertrophy-vs-strength type, set number, weight, reps, RPE, "
        f"estimated one-rep max and body weight, plus free-text notes per workout (injuries, mood, form, pump). "
        f"Columns: {names}, Comment."
    )


# --------------------------------------------------------------------------
# Opening message shown before the user types anything
# --------------------------------------------------------------------------
def build_greeting(df: pd.DataFrame) -> str:
    first, last = df["Date"].min(), df["Date"].max()
    top = df["Exersize"].value_counts().index[0] if df["Exersize"].notna().any() else None
    trend_q = (
        f"How has Kevin's estimated one-rep max on {top} trended over the last 6 months?"
        if top else "How has Kevin's estimated one-rep max trended over the last 3 months?"
    )

    rows = ["| Column | Type | What it holds |", "|---|---|---|"]
    for col in df.columns:
        doc = _DOC_BY_NAME.get(col)
        if doc:
            rows.append(f"| `{col}` | {doc.kind} | {doc.short} |")
        else:
            rows.append(f"| `{col}` | {_dtype_label(df[col])} | Additional column |")

    examples = [
        trend_q,
        "Which exercise did Kevin do the most total volume on last month?",
        "How much weight did Kevin bench press most recently?",
        "Draw a line chart of Kevin's body weight over the past year.",
        "What days did Kevin mention an injury or pain in his notes?",
    ]
    return "\n".join([
        f"Hi! I'm connected to Kevin's workout log: **{_plural(len(df), 'set')}** across "
        f"**{_plural(df['Date'].nunique(), 'workout')}**, "
        f"from **{_fmt_date(first)}** to **{_fmt_date(last)}**. Each row is one set.",
        "",
        "\n".join(rows),
        "",
        "**Try asking:**",
        *[f"- {q}" for q in examples],
        "",
        "Follow-up questions work, since I remember this conversation. I can add a chart "
        "when it helps, and only answer questions about Kevin's workout data.",
    ])
