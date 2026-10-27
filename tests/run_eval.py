"""Live evaluation: real sheet, real models, real answers checked against numbers computed straight from the data.

    python tests/run_eval.py --router-only        # cheap: only the router (1 call per question, no code is generated)
    python tests/run_eval.py                      # everything (about 3-4 model calls per question)
    python tests/run_eval.py --ids total_sets,vague
    python tests/run_eval.py --pause 5            # wait 5 s between questions (free tiers have per-minute limits)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

QUESTIONS_FILE = ROOT / "tests" / "eval_questions.yaml"
ROUTES = {"sql", "python", "clarify", "off_topic"}


# --------------------------------------------------------------------------------------
# Ground truth. Each function takes the cleaned DataFrame and returns what a correct answer must contain.
# --------------------------------------------------------------------------------------
@dataclass
class Expect:
    numbers: list[float] = field(default_factory=list)   # each must appear in the answer (1% or 0.06 tolerance)
    words: list[str] = field(default_factory=list)        # each must appear in the answer (case-insensitive)


def _workouts(df: pd.DataFrame) -> pd.DataFrame:
    """Workout-level columns repeat on every set, so one row per workout date."""
    return df.drop_duplicates("Date")


def total_sets(df): return Expect(numbers=[len(df)])
def workout_count(df): return Expect(numbers=[df["Date"].nunique()])
def top_exercise(df): return Expect(words=[str(df["Exercise"].value_counts().index[0])])
def heaviest_weight(df): return Expect(numbers=[float(df["Weight"].max())])
def heaviest_exercise(df): return Expect(words=[str(df.loc[df["Weight"].idxmax(), "Exercise"])])
def avg_rpe(df): return Expect(numbers=[float(df["RPE"].mean())])
def total_volume(df): return Expect(numbers=[float((df["Weight"] * df["Reps"].astype("float64")).sum())])
def latest_body_weight(df):
    last = df[df["Date"] == df["Date"].max()]["Body_Weight"].dropna()
    return Expect(numbers=[float(last.iloc[-1])] if len(last) else [])
def busiest_gym_day(df): return Expect(words=[str(_workouts(df)["Gym_Day"].value_counts().index[0])])
def avg_workout_rating(df): return Expect(numbers=[float(_workouts(df)["workout_rating"].dropna().astype("float64").mean())])
def injury_workouts(df): return Expect(numbers=[int((_workouts(df)["injury_ind"] == 1).sum())])
def rpe_by_type(df):
    means = df.groupby("Type")["RPE"].mean().dropna()
    return Expect(numbers=[float(v) for v in means], words=[str(t) for t in means.index])


TRUTH = {f.__name__: f for f in (
    total_sets, workout_count, top_exercise, heaviest_weight, heaviest_exercise, avg_rpe, total_volume,
    latest_body_weight, busiest_gym_day, avg_workout_rating, injury_workouts, rpe_by_type,
)}


# --------------------------------------------------------------------------------------
# Checking
# --------------------------------------------------------------------------------------
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    out = []
    for raw in _NUMBER.findall(text):
        try:
            out.append(float(raw.replace(",", "").rstrip(".")))
        except ValueError:
            pass
    return out


def number_present(target: float, text: str) -> bool:
    return any(abs(n - target) <= max(abs(target) * 0.01, 0.06) for n in numbers_in(text))


def check_answer(expect: Expect, answer: str) -> list[str]:
    """Return a list of problems (empty = the answer is correct)."""
    problems = [f"missing number {n:g}" for n in expect.numbers if not number_present(n, answer)]
    problems += [f"missing '{w}'" for w in expect.words if w.lower() not in answer.lower()]
    return problems


def load_cases(path: Path = QUESTIONS_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------
def run_router_only(cases, df, pause):
    from langchain_core.messages import HumanMessage

    from workout_chat.nodes.router import make_route_node
    from workout_chat.tools import TOOLS, ToolContext

    node = make_route_node(ToolContext.from_df(df), TOOLS)
    results = []
    for case in cases:
        update = node({"messages": [HumanMessage(case["question"])], "question": case["question"]})
        problems = [] if update["route"] == case["route"] else [f"routed to '{update['route']}', expected '{case['route']}'"]
        results.append((case, problems, update.get("answer", "") if update["route"] == "error" else ""))
        time.sleep(pause)
    return results


def run_full(cases, df, pause):
    from langgraph.checkpoint.memory import InMemorySaver

    from workout_chat.graph import build_graph, run_turn

    graph = build_graph(df, checkpointer=InMemorySaver())
    threads: dict[str, str] = {}
    results = []
    for case in cases:
        thread = threads.setdefault(case.get("thread") or case["id"], uuid.uuid4().hex)
        res = run_turn(graph, thread, case["question"])
        problems = []
        if res.route != case["route"]:
            problems.append(f"routed to '{res.route}', expected '{case['route']}'")
        if case.get("chart") and not res.figure_json:
            problems.append("no chart returned")
        if case.get("truth") and res.route == case["route"]:
            problems += check_answer(TRUTH[case["truth"]](df), res.answer)
        results.append((case, problems, res.answer))
        time.sleep(pause)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-only", action="store_true")
    parser.add_argument("--ids", help="comma-separated case ids")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between questions (default 2)")
    args = parser.parse_args(argv)

    from workout_chat.config import missing_secrets
    from workout_chat.data import load_dataframe

    needed = [m for m in missing_secrets() if m not in ("APP_PASSWORD", "GUEST_PASSWORD")]
    if needed:
        print(f"Missing secrets: {', '.join(needed)}")
        return 1
    df = load_dataframe()

    cases = load_cases()
    if args.ids:
        wanted = set(args.ids.split(","))
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            print(f"Unknown ids: {', '.join(sorted(unknown))}")
            return 1
        cases = [c for c in cases if c["id"] in wanted]
    skipped = [c["id"] for c in cases if any(col not in df.columns for col in c.get("needs", []))]
    cases = [c for c in cases if c["id"] not in skipped]
    if args.router_only:
        cases = [c for c in cases if not c.get("thread") or c["id"] != "heaviest_exercise"]  # needs conversation history

    runner = run_router_only if args.router_only else run_full
    results = runner(cases, df, args.pause)

    failed = 0
    for case, problems, answer in results:
        print(f"{'PASS' if not problems else 'FAIL'}  {case['id']:<26} {case['route']}")
        for p in problems:
            print(f"      - {p}")
        if problems and answer:
            print(f"      answer: {answer[:300]!r}")
        failed += bool(problems)
    if skipped:
        print(f"\nSkipped (columns missing from the sheet): {', '.join(skipped)}")
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
