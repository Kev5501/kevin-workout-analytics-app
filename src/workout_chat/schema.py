"""Data dictionary, schema context for the LLMs, and the opening chat greeting.

Nothing here calls an LLM. Everything is derived from the cleaned DataFrame plus
the static data dictionary, so it costs no quota and is fully deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from .data import EXPECTED_COLUMNS, FEATURE_COLUMNS

DF_NAME = "df"                  # variable name for generated pandas code
SQL_TABLE_NAME = "workouts"     # table name for generated SQL (same columns as df)
COMMENT_MASK = "[masked]"
MASKED_COLUMNS = ("Comment",)   # never shown to a model in sample rows
CATEGORICAL_COLUMNS = ("Gym_Day", "Type", "Exercise")  # exact values are listed for the coder
# Values repeated on every set of a workout (Comment and the features derived from it).
WORKOUT_LEVEL_COLUMNS = ("Comment", *FEATURE_COLUMNS)

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
    ColumnDoc("Exercise", "text", "Exercise name",
              "The name of the exercise performed."),
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
    ColumnDoc("workout_rating", "number (1-5)", "Overall rating of the workout, based on the comment",
              "Overall rating of the workout, inferred from the lifter's comment: 1 = very bad, 2 = bad, "
              "3 = okay, 4 = good, 5 = very good. Missing (null) if the comment had too little information "
              "to rate the workout or has not been processed yet. Never compare it to the string 'NA'."),
    ColumnDoc("pump_rating", "number (1-3)", "Rating of the pump, based on the comment",
              "How good the lifter's pump was, inferred from the comment: 1 = below average, 2 = average, "
              "3 = above average. Missing (null) if the comment had too little information to rate the pump "
              "or has not been processed yet. Never compare it to the string 'NA'."),
    ColumnDoc("injury_ind", "yes/no (1/0)", "Had pain, a tweak or an injury",
              "1 if the lifter had pain, a tweak, an injury or joint/muscle discomfort beyond normal training "
              "soreness, else 0. Missing (null) if the comment has not been processed yet."),
    ColumnDoc("illness_ind", "yes/no (1/0)", "Was ill",
              "1 if the lifter was ill in any way during the workout, else 0. "
              "Missing (null) if the comment has not been processed yet."),
    ColumnDoc("girlfriend_ind", "yes/no (1/0)", "Girlfriend was present",
              "1 if the lifter's girlfriend was present during the workout, else 0. "
              "Missing (null) if the comment has not been processed yet."),
    ColumnDoc("fatigue_ind", "yes/no (1/0)", "Felt fatigued or not recovered",
              "1 if the lifter felt tired, fatigued, not recovered, or lacked sleep for the workout, else 0. "
              "Missing (null) if the comment has not been processed yet."),
    ColumnDoc("vacation_ind", "yes/no (1/0)", "Was on vacation",
              "1 if the lifter was on any sort of vacation for the workout, else 0. "
              "Missing (null) if the comment has not been processed yet."),
    ColumnDoc("form_change_ind", "yes/no (1/0)", "Changed form or technique",
              "1 if the lifter had any change in form, technique or execution of exercises for the workout, else 0. "
              "Missing (null) if the comment has not been processed yet."),
    ColumnDoc("forgot_headphones_ind", "yes/no (1/0)", "Forgot his headphones",
              "1 if the lifter forgot or did not have his headphones for the workout, else 0. "
              "Missing (null) if the comment has not been processed yet."),
    ColumnDoc("other_extenuating_circumstances_ind", "yes/no (1/0)", "Other extenuating circumstances",
              "1 if there were extenuating circumstances for the workout unrelated to injury, illness, fatigue, "
              "vacation, form change or forgetting headphones, else 0. "
              "Missing (null) if the comment has not been processed yet."),
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


def _missing_value_lines(df: pd.DataFrame, workout_level: list[str]) -> list[str]:
    """Where values are missing, and how the code model should treat them."""
    per_set = [c for c in df.columns if c not in workout_level]
    set_nulls = df[per_set].isna().sum()
    set_nulls = set_nulls[set_nulls > 0]
    workouts = df.drop_duplicates("Date")
    wk_nulls = workouts[workout_level].isna().sum() if workout_level else pd.Series(dtype=int)
    wk_nulls = wk_nulls[wk_nulls > 0]
    if set_nulls.empty and wk_nulls.empty:
        return []

    lines = [""]
    if not set_nulls.empty:
        lines.append("Missing values: " + ", ".join(f"{c}: {n}" for c, n in set_nulls.items()) + " (counted in sets)")
    if not wk_nulls.empty:
        lines.append(
            f"Missing values in workout-level columns, counted in workouts (of {len(workouts)}): "
            + ", ".join(f"{c}: {n}" for c, n in wk_nulls.items())
        )
    lines += [
        "Handling missing values (null = unknown, NOT 0 and NOT 'no'):",
        "- Never drop or filter rows for nulls in a column the question does not use. A question about bench press "
        "weight keeps every row even if workout_rating is null.",
        "- When a question does use a column, ignore that column's nulls for that calculation only. pandas "
        "mean/sum/count and SQL AVG/COUNT(col) already skip nulls, and comparisons such as == 1 or = 0 leave unknowns out. "
        "To filter explicitly, use .notna() or IS NOT NULL.",
        "- Never fillna(0), astype(int) or astype(bool) these columns, and never use the total number of workouts or sets "
        "as the denominator for a rate. Divide by the number of workouts that have a value (mean() or COUNT(col) does this).",
        "- Tell the user how many workouts had a value for the column(s) used, out of the total.",
    ]
    return lines


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

    workout_level = [c for c in WORKOUT_LEVEL_COLUMNS if c in df.columns]
    lines += [
        "",
        f"Workout-level columns ({', '.join(workout_level)}): every set in a workout repeats the same value, and "
        "one workout = a unique Date and Gym_Day combination. To count or average across WORKOUTS, first keep one row per unique Date/Gym_Day combination "
        f"(pandas: {DF_NAME}.drop_duplicates(subset=['Date','Gym_Day']); SQL: SELECT DISTINCT Date, Gym_Day <column> FROM {SQL_TABLE_NAME}). "
        "Using every row instead would weight each workout by its number of sets.",
    ]

    lines += _missing_value_lines(df, workout_level)

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
    top = df["Exercise"].value_counts().index[0] if df["Exercise"].notna().any() else None
    trend_q = (
        f"How has Kevin's estimated one-rep max on {top} trended over the last 3 months?"
        if top else "How has my estimated one-rep max trended over the last 3 months?"
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
        "Follow-up questions work, since I remember this conversation. I'll show the code I ran, add a chart "
        "when it helps, and only answer questions about this workout data.",
    ])