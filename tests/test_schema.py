import re

import pandas as pd

from conftest import HEADER, make_row
from workout_chat import schema
from workout_chat.data import EXPECTED_COLUMNS, FEATURE_COLUMNS, clean_dataframe
from workout_chat.schema import (
    COMMENT_MASK, DATA_DICTIONARY, build_greeting, build_schema_context, router_domain_description,
)


def test_dictionary_covers_every_expected_column_in_order():
    assert tuple(d.name for d in DATA_DICTIONARY) == EXPECTED_COLUMNS


def test_dictionary_never_tells_the_model_to_look_for_na_strings():
    """Cleaned features are null, not the string 'NA'; the text must say so."""
    for doc in DATA_DICTIONARY:
        if doc.name in FEATURE_COLUMNS:
            assert "null" in doc.detail.lower(), doc.name


# ---- schema context ---------------------------------------------------------
def test_context_never_contains_comment_text(sample_df):
    ctx = build_schema_context(sample_df)
    assert "SECRET" not in ctx and "great pump" not in ctx
    assert COMMENT_MASK in ctx


def test_context_basics(sample_df):
    ctx = build_schema_context(sample_df)
    assert "Rows: 6" in ctx and "2025-03-05 to 2025-03-14" in ctx
    assert f"`{schema.DF_NAME}`" in ctx and f"`{schema.SQL_TABLE_NAME}`" in ctx
    assert "Exercise" in ctx  # dictionary mentions the spelling


def test_context_lists_exact_values_most_frequent_first(sample_df):
    ctx = build_schema_context(sample_df)
    assert ctx.index('"Bench Press"') < ctx.index('"Squat"')  # 4 sets vs 2
    assert ctx.index('"Push"') < ctx.index('"Legs"')


def test_context_reports_missing_values(sample_df):
    assert "Missing values:" in build_schema_context(sample_df)
    assert "RPE: 1" in build_schema_context(sample_df)


def test_context_samples_are_most_recent_rows_only(sample_df):
    ctx = build_schema_context(sample_df, n_samples=2)
    assert "OLDEST" not in ctx
    assert "S5" in ctx and "S6" in ctx


def test_context_caps_long_value_lists():
    rows = [HEADER] + [make_row(set_id=f"S{i}", ex=f"Exercise {i}") for i in range(5)]
    ctx = build_schema_context(clean_dataframe(rows), max_values=2)
    assert "5 distinct" in ctx and "(+3 more not shown)" in ctx


def test_context_includes_extra_columns_with_dtype():
    rows = [HEADER + ["Shoulder_Pain"], make_row() + [True], make_row(set_id="S2") + [False]]
    ctx = build_schema_context(clean_dataframe(rows))
    assert "Shoulder_Pain (bool)" in ctx and "no description available" in ctx


def test_context_still_works_when_comments_are_all_empty():
    rows = [HEADER, make_row(comment=""), make_row(set_id="S2", comment="")]
    assert "Rows: 2" in build_schema_context(clean_dataframe(rows))


# ---- router blurb -----------------------------------------------------------
def test_router_description_is_short_and_data_free():
    text = router_domain_description()
    assert len(text) < 900
    for col in ("Exercise", "RPE", "Weight", "Comment"):
        assert col in text


# ---- greeting ---------------------------------------------------------------
def test_greeting_lists_every_column_and_stats(sample_df):
    g = build_greeting(sample_df)
    for col in sample_df.columns:
        assert f"`{col}`" in g
    assert "6 sets" in g and "3 workouts" in g
    assert "Mar 5, 2025" in g and "Mar 14, 2025" in g  # no leading zero on the day


def test_greeting_uses_most_frequent_exercise_in_example(sample_df):
    assert "estimated one-rep max on Bench Press" in build_greeting(sample_df)


def test_greeting_never_leaks_comment_text(sample_df):
    g = build_greeting(sample_df)
    assert "SECRET" not in g and "great pump" not in g


def test_greeting_flags_extra_columns():
    rows = [HEADER + ["Mood_Score"], make_row() + [4]]
    assert re.search(r"`Mood_Score` \| \w+ \| Additional column", build_greeting(clean_dataframe(rows)))


def test_greeting_singular_wording():
    g = build_greeting(clean_dataframe([HEADER, make_row()]))
    assert "1 set**" in g and "1 workout**" in g
