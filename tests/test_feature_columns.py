import pandas as pd
import pytest

from conftest import make_row
from workout_chat.data import (
    BINARY_COLUMNS, CORE_COLUMNS, EXPECTED_COLUMNS, FEATURE_COLUMNS, ORDINAL_COLUMNS, clean_dataframe,
)
from workout_chat.schema import build_schema_context

HEADER = list(EXPECTED_COLUMNS)


def row(rating="", pump="", binaries=None, **kw):
    """One raw row with every documented column. ``binaries`` can be a value (for all 8) or a list."""
    if not isinstance(binaries, list):
        binaries = [binaries if binaries is not None else ""] * len(BINARY_COLUMNS)
    return make_row(**kw) + [rating, pump] + binaries


def feature_df(*rows):
    return clean_dataframe([HEADER, *rows])


def test_feature_columns_are_nullable_integers():
    df = feature_df(row(rating=4, pump=2, binaries=[1, 0, 0, 1, 0, 0, 0, 1]))
    for col in FEATURE_COLUMNS:
        assert str(df[col].dtype) == "Int64", col
    assert df["workout_rating"].iloc[0] == 4 and df["injury_ind"].iloc[0] == 1


def test_blank_and_NA_strings_both_become_missing():
    df = feature_df(
        row(rating="NA", pump="n/a", binaries="", set_id="a"),
        row(rating="", pump="", binaries="NA", set_id="b"),
        row(rating=3, pump=1, binaries=0, set_id="c"),
    )
    assert df["workout_rating"].isna().tolist() == [True, True, False]
    assert df["pump_rating"].isna().tolist() == [True, True, False]
    assert df["injury_ind"].isna().tolist() == [True, True, False]
    assert df.attrs["feature_value_issues"] == {}  # NA / blank are normal, not problems


def test_aggregations_skip_missing_values():
    df = feature_df(row(rating=4, set_id="a"), row(rating="NA", set_id="b"), row(rating=2, set_id="c"))
    assert df["workout_rating"].mean() == 3.0
    assert df["workout_rating"].count() == 2


def test_numeric_text_and_booleans_are_accepted():
    df = feature_df(row(rating="4", pump=2.0, binaries=True, set_id="a"), row(rating=5, binaries="false", set_id="b"))
    assert df["workout_rating"].tolist() == [4, 5]
    assert df["pump_rating"].iloc[0] == 2
    assert df["injury_ind"].tolist() == [1, 0]


@pytest.mark.parametrize("col,bad", [
    ("rating", 0), ("rating", 6), ("rating", 3.5), ("rating", "great"),
    ("pump", 4), ("pump", 0),
])
def test_out_of_range_or_invalid_ordinals_become_missing_and_are_counted(col, bad):
    kwargs = {col: bad}
    df = feature_df(row(**kwargs))
    name = "workout_rating" if col == "rating" else "pump_rating"
    assert df[name].isna().iloc[0]
    assert df.attrs["feature_value_issues"] == {name: 1}


def test_invalid_binary_values_become_missing_and_are_counted():
    df = feature_df(row(binaries=[2, "maybe", 0, 1, 0, 0, 0, 0]))
    assert df["injury_ind"].isna().iloc[0] and df["illness_ind"].isna().iloc[0]
    assert df.attrs["feature_value_issues"] == {"injury_ind": 1, "illness_ind": 1}


def test_unprocessed_rows_are_rows_with_every_feature_blank():
    df = feature_df(
        row(set_id="blank"),                                   # nothing processed
        row(rating="NA", set_id="insufficient"),               # processed: agent said NA
        row(binaries=0, set_id="flags-only"),
    )
    assert df.attrs["unprocessed_rows"] == 1


def test_feature_columns_are_optional():
    """A sheet without the feature columns still loads; the app just has fewer columns."""
    df = clean_dataframe([list(CORE_COLUMNS), make_row()])
    assert list(df.columns) == list(CORE_COLUMNS)
    assert df.attrs["missing_feature_columns"] == list(FEATURE_COLUMNS)
    assert df.attrs["unprocessed_rows"] == 0


def test_partial_feature_columns_work():
    header = list(CORE_COLUMNS) + ["workout_rating", "injury_ind"]
    df = clean_dataframe([header, make_row() + [4, 1]])
    assert df["workout_rating"].iloc[0] == 4 and df["injury_ind"].iloc[0] == 1
    assert "pump_rating" in df.attrs["missing_feature_columns"]


def test_column_order_follows_the_dictionary_even_if_sheet_order_differs():
    header = list(CORE_COLUMNS) + ["injury_ind", "workout_rating"]
    df = clean_dataframe([header, make_row() + [1, 4]])
    assert list(df.columns)[-2:] == ["workout_rating", "injury_ind"]


def test_core_columns_are_still_required():
    header = [c for c in HEADER if c != "Reps"]
    with pytest.raises(Exception, match="Reps"):
        clean_dataframe([header, ["x"] * len(header)])


def test_schema_context_tells_the_model_features_are_workout_level():
    ctx = build_schema_context(feature_df(row(rating=4, set_id="a"), row(rating=4, set_id="b")))
    assert "Workout-level columns (Comment, workout_rating" in ctx
    assert "drop_duplicates('Date')" in ctx and "SELECT DISTINCT Date" in ctx
    assert "workout_rating (Int64)" in ctx


def test_schema_context_only_lists_workout_level_columns_that_exist():
    ctx = build_schema_context(clean_dataframe([list(CORE_COLUMNS), make_row()]))
    assert "Workout-level columns (Comment)" in ctx


def test_every_ordinal_range_is_documented_in_the_dictionary():
    from workout_chat.schema import DATA_DICTIONARY
    docs = {d.name: d for d in DATA_DICTIONARY}
    for col, (low, high) in ORDINAL_COLUMNS.items():
        assert f"({low}-{high})" in docs[col].kind
