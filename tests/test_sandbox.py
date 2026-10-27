import json
import time

import pandas as pd
import pytest

from workout_chat import sandbox
from workout_chat.sandbox import run_python, run_sql, validate_python


# ---- SQL ---------------------------------------------------------------------
def test_sql_basic_aggregation(sample_df):
    res = run_sql('SELECT "Exercise", COUNT(*) AS n FROM workouts GROUP BY 1 ORDER BY n DESC', sample_df)
    assert res.ok and "Bench Press" in res.text and "4" in res.text


def test_sql_with_clause_and_dates(sample_df):
    res = run_sql("WITH d AS (SELECT DISTINCT Date FROM workouts) SELECT COUNT(*) AS workouts FROM d WHERE Date >= DATE '2025-03-10'", sample_df)
    assert res.ok and "2" in res.text


def test_sql_handles_nullable_features(sample_df):
    res = run_sql("SELECT AVG(Weight) AS w, COUNT(RPE) AS rpe_known FROM workouts", sample_df)
    assert res.ok and "rpe_known" in res.text


def test_sql_truncates_long_results(sample_df):
    res = run_sql("SELECT * FROM range(500)", sample_df, max_rows=5)
    assert res.ok and "showing the first 5" in res.text


def test_sql_zero_rows(sample_df):
    assert "0 rows" in run_sql("SELECT * FROM workouts WHERE Weight > 99999", sample_df).text


@pytest.mark.parametrize("bad", [
    "SELECT 1; SELECT 2",
    "DROP TABLE workouts",
    "INSERT INTO workouts SELECT * FROM workouts",
    "COPY (SELECT 1) TO 'x.csv'",
    "ATTACH 'x.db'",
    "SET enable_external_access=true",
    "PRAGMA database_list",
    "",
])
def test_sql_rejects_anything_but_one_select(sample_df, bad):
    res = run_sql(bad, sample_df)
    assert not res.ok and res.error


def test_sql_cannot_read_files(sample_df):
    res = run_sql("SELECT * FROM read_csv('/etc/passwd')", sample_df)
    assert not res.ok and ("disabled" in res.error.lower() or "permission" in res.error.lower())


def test_sql_error_message_is_useful_for_retry(sample_df):
    res = run_sql("SELECT Wieght FROM workouts", sample_df)
    assert not res.ok and "Weight" in res.error  # DuckDB suggests the right column


def test_sql_syntax_error(sample_df):
    assert "syntax" in run_sql("SELEC 1", sample_df).error.lower()


def test_sql_timeout(sample_df):
    start = time.time()
    res = run_sql("SELECT COUNT(*) FROM range(100000000000)", sample_df, timeout=1)
    assert not res.ok and "longer than" in res.error and time.time() - start < 8


# ---- Python: static checks (fast, no subprocess) -----------------------------------
@pytest.mark.parametrize("code", [
    "import os",
    "from os import path",
    "open('x')",
    "__import__('os')",
    "eval('1')",
    "exec('1')",
    "getattr(df, 'x')",
    "df.__class__",
    "df._mgr",
    "x = 'a__b'",
    "df.to_csv('x.csv')",
    "pd.read_csv('x.csv')",
    "pd.read_html('http://example.com')",
    "np.load('x.npy')",
    "pd.io.common",
    "df.eval('a+b')",
    "os.system('ls')",
    "async def f(): pass",
])
def test_validator_blocks(code):
    assert validate_python(code), code


@pytest.mark.parametrize("code", [
    "result = df['Weight'].mean()",
    "result = df.groupby('Exercise')['Weight'].max().reset_index()",
    "fig = px.line(df, x='Date', y='Weight')\nresult = 1",
    "result = df.drop_duplicates('Date')['workout_rating'].mean() if 'workout_rating' in df else None",
    "recent = df[df['Date'] >= pd.Timestamp.today() - pd.Timedelta(days=30)]\nresult = len(recent)",
])
def test_validator_allows_normal_analysis(code):
    assert validate_python(code) is None


def test_validator_reports_syntax_errors_with_line():
    assert "SyntaxError" in validate_python("result = (1 +\n")


def test_validator_rejects_oversized_code():
    assert "too long" in validate_python("x = 1\n" * 5000)


# ---- Python: real subprocess runs --------------------------------------------------
def test_python_scalar_result(sample_df):
    res = run_python("result = round(df['Weight'].mean(), 2)", sample_df)
    assert res.ok and res.text.strip() == str(round(sample_df["Weight"].mean(), 2))


def test_python_dataframe_result_and_print(sample_df):
    res = run_python("result = df.groupby('Exercise')['Weight'].max().reset_index()\nprint('done')", sample_df)
    assert res.ok and "Squat" in res.text and "Printed output" in res.text and "done" in res.text


def test_python_plotly_figure_returned_as_json(sample_df):
    code = "fig = px.line(df, x='Date', y='Weight', title='Weight')\nresult = 'ok'"
    res = run_python(code, sample_df)
    assert res.ok and res.figure_json
    assert json.loads(res.figure_json)["layout"]["title"]["text"] == "Weight"


def test_python_figure_assigned_to_result_is_still_a_chart(sample_df):
    res = run_python("result = px.bar(df, x='Exercise', y='Reps')", sample_df)
    assert res.ok and res.figure_json


def test_python_large_results_are_truncated(sample_df):
    res = run_python("result = pd.DataFrame({'n': range(500)})", sample_df)
    assert res.ok and "500 rows in total" in res.text


def test_python_runtime_error_reports_line(sample_df):
    res = run_python("x = 1\nresult = df['NoSuchColumn'].mean()", sample_df)
    assert not res.ok and "KeyError" in res.error and "line 2" in res.error


def test_python_requires_a_result(sample_df):
    res = run_python("x = 1", sample_df)
    assert not res.ok and "result" in res.error


def test_python_blocked_code_never_starts_a_process(sample_df):
    start = time.time()
    res = run_python("import os", sample_df)
    assert not res.ok and "Imports" in res.error and time.time() - start < 1


def test_python_infinite_loop_is_killed(sample_df):
    start = time.time()
    res = run_python("while True:\n    pass", sample_df, timeout=3)
    assert not res.ok and "longer than" in res.error and time.time() - start < 15


def test_python_cannot_change_the_callers_data(sample_df):
    before = sample_df.copy()
    res = run_python("df['Weight'] = 0\nresult = df['Weight'].sum()", sample_df)
    assert res.ok and res.text.strip() == "0"
    pd.testing.assert_frame_equal(sample_df, before)


def test_python_child_gets_no_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "super-secret")
    env = sandbox._child_env()
    assert not any("secret" in v for v in env.values()) and "GEMINI_API_KEY" not in env
