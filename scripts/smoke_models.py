"""Smoke-test the models in models.toml with a few tiny REAL requests.

    python scripts/smoke_models.py                      # every model used by any node
    python scripts/smoke_models.py --node router        # only models in the router chain
    python scripts/smoke_models.py --models gemini_flash,mercury_decide_free

Uses real API quota (a handful of requests per model). Models whose API key is not set are
skipped. Exit code is 1 if any model that has a key fails, so you can run it after editing
models.toml. Obvious-case classifier answers that look wrong are flagged as WARN, not FAIL.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from workout_chat import llm  # noqa: E402
from workout_chat.config import PROVIDER_SECRETS, ConfigError, get_model_config, has_secret  # noqa: E402
from workout_chat.decisions import (  # noqa: E402
    Choice, Noul, call_chat_decision_model, call_decision_model,
)
from workout_chat.llm import ModelCallError  # noqa: E402

QUESTIONS = {
    "route": Choice(
        "Which kind of work answers this message about Kevin's workout log?",
        {
            "sql": "A lookup, filter, count, sum or average over the workout table",
            "python": "Statistics, trends, advanced math or a chart that needs pandas code",
            "off_topic": "Not about Kevin's workouts or this dataset at all",
        },
    ),
    "needs_chart": Noul("Does the message ask for a chart or plot?", "A chart or plot is requested", "No chart requested"),
}
# (state, expected route)
CASES = [
    ("How many sets of squats did Kevin do last month?", "sql"),
    ("What is the capital of France?", "off_topic"),
    ("Give me a chart of Kevin's weight over the last year.","python"),
    ("How much did Kevin bench press last week?", "sql"),
    ("How can I get better squat form?", "off_topic"),
    ("What was the month over month difference in Kevins squat in the last two months?", "python")
]


def ms(start: float) -> str:
    return f"{(time.time() - start) * 1000:.0f} ms"


def check_answers(answers, expected: str) -> str:
    got = answers["route"].value
    note = "" if got == expected else f"  WARN expected route '{expected}'"
    return f"route={got} (confidence {answers['route'].confidence:.2f}), needs_chart P={answers['needs_chart'].value:.2f}{note}"


def test_model(spec, nodes: list[str], settings: dict) -> list[str]:
    """Run the checks for one model. Returns result lines; raises ModelCallError on failure."""
    lines = []
    if spec.kind == "decision":
        for state, expected in CASES:
            t0 = time.time()
            lines.append(f"decision call ok ({ms(t0)}): " + check_answers(call_decision_model(spec, state, QUESTIONS), expected))
        return lines
    t0 = time.time()
    reply = llm.call_chat_model(spec, [("human", "Reply with exactly the word: OK")], settings=settings)
    lines.append(f"chat call ok ({ms(t0)}): {llm.text_of(reply).strip()[:40]!r}")
    if "router" in nodes:  # this model may have to act as the router fallback
        state, expected = CASES[0]
        t0 = time.time()
        answers = call_chat_decision_model(spec, state, QUESTIONS, settings=settings)
        lines.append(f"structured decisions ok ({ms(t0)}): " + check_answers(answers, expected))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--node", help="only test models in this node's chain")
    parser.add_argument("--models", help="comma-separated model keys from models.toml")
    args = parser.parse_args()

    try:
        cfg = get_model_config()
    except (ConfigError, OSError) as exc:
        print(f"models.toml problem: {exc}")
        return 1

    nodes = [args.node] if args.node else list(cfg.nodes)
    unknown = [n for n in nodes if n not in cfg.nodes]
    if unknown:
        print(f"Unknown node(s): {', '.join(unknown)}. Known: {', '.join(cfg.nodes)}")
        return 1

    used: dict[str, list[str]] = {}
    for node in nodes:
        for key in cfg.nodes[node]:
            used.setdefault(key, []).append(node)
    if args.models:
        wanted = [k.strip() for k in args.models.split(",")]
        bad = [k for k in wanted if k not in cfg.models]
        if bad:
            print(f"Unknown model key(s): {', '.join(bad)}. Known: {', '.join(cfg.models)}")
            return 1
        used = {k: used.get(k, []) for k in wanted}

    failed = 0
    summary = []
    for key, node_list in used.items():
        spec = cfg.models[key]
        header = f"{key}  [{spec.provider}:{spec.model}, kind={spec.kind}]  nodes: {', '.join(node_list) or '-'}"
        secret = PROVIDER_SECRETS[spec.provider]
        print(header)
        if not has_secret(secret):
            print(f"  SKIP  secret {secret} is not set\n")
            summary.append((key, "SKIP"))
            continue
        try:
            for line in test_model(spec, node_list, cfg.settings):
                print(f"  PASS  {line}")
            summary.append((key, "PASS"))
        except ModelCallError as err:
            failed += 1
            print(f"  FAIL  {err.kind}: {err.detail}")
            summary.append((key, "FAIL"))
        print()

    print("Summary: " + ", ".join(f"{k}={s}" for k, s in summary))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
