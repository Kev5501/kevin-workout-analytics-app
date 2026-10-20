"""Missing / 'NA' feature values must only matter to questions that use that column."""
import duckdb
import pytest

from conftest import make_row
from workout_chat.data import BINARY_COLUMNS, CORE_COLUMNS, FEATURE_COLUMNS, clean_dataframe
from workout_chat.schema import build_schema_context

HEADER = list(CORE_COLUMNS) + list(FEATURE_COLUMNS)


def raw(date, weight, rating, injury, sets=2):
    pad = [""] * (len(BINARY_COLUMNS) - 1)
    return [make_row(date=date, set_id=f"{date}-{i}", w=weight) + [rating, "", injury] + pad for i in range(sets)]


@pytest.fixture
def df():
    """4 workouts x 2 sets: rated / 'NA' / never processed / rated but injury unknown."""
    rows = [HEADER]
    for r in (raw("2025-03-01", 100, 4, 1), raw("2025-03-02", 200, "NA", 0), raw("2025-03-03", 300, "", ""),
              raw("2025-03-04", 400, 2, "")):
        rows += r
    return clean_dataframe(rows)


def test_loader_keeps_every_row_whatever_is_missing(df):
    assert len(df) == 8


def test_unrelated_question_uses_every_row(df):
    assert df["Weight"].mean() == 250.0  # all four workouts, including unprocessed ones


def test_feature_question_skips_only_that_columns_nulls(df):
    workouts = df.drop_duplicates("Date")
    assert workouts["workout_rating"].mean() == 3.0
    assert workouts["workout_rating"].count() == 2
    assert int((workouts["injury_ind"] == 1).sum()) == 1
    assert int((workouts["injury_ind"] != 1).sum()) == 1      # unknown is not 'no'
    assert workouts["injury_ind"].mean() == 0.5               # denominator = known workouts


def test_sql_behaves_the_same(df):
    con = duckdb.connect()
    con.register("workouts", df)
    assert con.sql("SELECT AVG(Weight) FROM workouts").fetchone()[0] == 250.0
    assert con.sql("SELECT AVG(workout_rating) FROM (SELECT DISTINCT Date, workout_rating FROM workouts)").fetchone()[0] == 3.0
    assert con.sql("SELECT COUNT(injury_ind) FROM (SELECT DISTINCT Date, injury_ind FROM workouts)").fetchone()[0] == 2


# ---- what the code model is told ----------------------------------------------
def test_context_counts_workout_level_nulls_in_workouts_not_sets(df):
    ctx = build_schema_context(df)
    assert "counted in workouts (of 4)" in ctx
    assert "workout_rating: 2" in ctx            # 2 workouts missing a rating (not 4 sets)
    assert "injury_ind: 2" in ctx


def test_context_states_the_rules_for_missing_values(df):
    ctx = build_schema_context(df)
    assert "null = unknown, NOT 0 and NOT 'no'" in ctx
    assert "Never drop or filter rows for nulls in a column the question does not use" in ctx
    assert "Never fillna(0)" in ctx and "denominator" in ctx
    assert "how many workouts had a value" in ctx


def test_set_level_nulls_are_counted_in_sets():
    rows = [list(CORE_COLUMNS), make_row(rpe=""), make_row(set_id="S2", rpe=""), make_row(set_id="S3")]
    ctx = build_schema_context(clean_dataframe(rows))
    assert "RPE: 2 (counted in sets)" in ctx


def test_no_missing_value_section_when_nothing_is_missing():
    rows = [list(CORE_COLUMNS), make_row(), make_row(set_id="S2")]
    ctx = build_schema_context(clean_dataframe(rows))
    assert "Handling missing values" not in ctx and "Missing values" not in ctx
