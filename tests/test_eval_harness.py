"""Tests for the evaluation harness itself (no network, no models)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import run_eval  # noqa: E402
from run_eval import Expect, TRUTH, check_answer, load_cases, number_present, numbers_in  # noqa: E402

from conftest import HEADER, make_row  # noqa: E402
from workout_chat.data import EXPECTED_COLUMNS, clean_dataframe  # noqa: E402


def test_there_are_17_well_formed_cases():
    cases = load_cases()
    assert len(cases) == 17
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == len(ids)
    for c in cases:
        assert c["route"] in run_eval.ROUTES and c["question"].strip()
        if c["route"] in ("sql",):
            assert c["truth"] in TRUTH, c["id"]


def test_every_truth_function_is_used_and_every_use_exists():
    used = {c["truth"] for c in load_cases() if c.get("truth")}
    assert used == set(TRUTH)


def test_follow_up_cases_share_a_thread_and_come_in_order():
    cases = [c for c in load_cases() if c.get("thread") == "heaviest"]
    assert [c["id"] for c in cases] == ["heaviest_weight", "heaviest_exercise"]


@pytest.mark.parametrize("text,expected", [
    ("Kevin lifted 1,025.5 lbs", [1025.5]),
    ("Total: 6 sets. RPE 8.5.", [6.0, 8.5]),
    ("It was 8.", [8.0]),
    ("no numbers", []),
])
def test_numbers_in(text, expected):
    assert numbers_in(text) == expected


def test_number_matching_tolerates_rounding_but_not_errors():
    assert number_present(8.4666, "average RPE is 8.47")
    assert number_present(8.4666, "average RPE is 8.5")
    assert number_present(12345, "about 12,340")
    assert not number_present(8.4666, "average RPE is 7")
    assert not number_present(6, "he did 60 sets")


def test_check_answer_reports_each_problem():
    assert check_answer(Expect([6], ["Squat"]), "6 sets, mostly squat") == []
    problems = check_answer(Expect([6], ["Squat"]), "7 sets of Bench")
    assert problems == ["missing number 6", "missing 'Squat'"]


@pytest.fixture
def df():
    header = list(EXPECTED_COLUMNS)
    pad = [""] * 10  # feature columns not processed yet (blank)
    flags_injury = [4, 2, 1, 0, 0, 0, 0, 0, 0, 0]
    rows = [
        make_row("2025-03-05", "a", w=100, rpe=8, bw=180, ex="Bench Press", typ="Strength") + pad,
        make_row("2025-03-05", "b", w=300, rpe=9, bw=180, ex="Squat", typ="Strength", day="Legs") + pad,
        make_row("2025-03-12", "c", w=100, rpe=7, bw=181, ex="Bench Press", typ="Hypertrophy") + flags_injury,
        make_row("2025-03-12", "d", w=100, rpe="", bw=181, ex="Bench Press", typ="Hypertrophy") + flags_injury,
    ]
    return clean_dataframe([header, *rows])


def test_truth_functions_match_hand_calculated_values(df):
    assert TRUTH["total_sets"](df).numbers == [4]
    assert TRUTH["workout_count"](df).numbers == [2]
    assert TRUTH["top_exercise"](df).words == ["Bench Press"]
    assert TRUTH["heaviest_weight"](df).numbers == [300.0]
    assert TRUTH["heaviest_exercise"](df).words == ["Squat"]
    assert TRUTH["avg_rpe"](df).numbers == [8.0]                       # (8 + 9 + 7) / 3, the blank RPE is ignored
    assert TRUTH["total_volume"](df).numbers == [100 * 8 + 300 * 8 + 100 * 8 + 100 * 8]
    assert TRUTH["latest_body_weight"](df).numbers == [181.0]
    assert TRUTH["avg_workout_rating"](df).numbers == [4.0]            # per workout, not per set: only the 03-12 workout is rated
    assert TRUTH["injury_workouts"](df).numbers == [1]                 # one workout, even though two sets repeat the flag
    rpe = TRUTH["rpe_by_type"](df)
    assert sorted(rpe.words) == ["Hypertrophy", "Strength"] and sorted(rpe.numbers) == [7.0, 8.5]


def test_busiest_gym_day_counts_workouts_not_sets(df):
    # Push has 3 sets in 2 workouts (03-05 starts with a Push set); Legs has 1 set -> by workout, still Push
    assert TRUTH["busiest_gym_day"](df).words == ["Push"]


def test_cases_needing_missing_columns_are_declared():
    needs = {c["id"]: c["needs"] for c in load_cases() if c.get("needs")}
    assert needs == {"avg_workout_rating": ["workout_rating"], "injury_workouts": ["injury_ind"]}
