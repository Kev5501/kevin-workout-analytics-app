"""Try the sandbox on your real data"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workout_chat.data import load_dataframe  # noqa: E402
from workout_chat.sandbox import run_python, run_sql  # noqa: E402

df = load_dataframe()

print("1) SQL: top 5 exercises by number of sets")
print(run_sql("SELECT Exercise, COUNT(*) AS sets FROM workouts GROUP BY 1 ORDER BY 2 DESC LIMIT 5", df).text)

print("\n2) Python: heaviest weight")
print(run_python('result = df["Weight"].max()', df).text)

print("\n3) Python: a forbidden import should be blocked")
print(run_python("import os", df).error)

print("\n4) SQL: anything other than a single SELECT should be blocked")
print(run_sql("DROP TABLE workouts", df).error)