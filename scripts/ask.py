"""Ask the real agent a question from the terminal (uses the real sheet and real models).

    python scripts/ask.py "How many sets of Bench Press has Kevin done?"
    python scripts/ask.py                       # interactive chat; type 'quit' to stop
Charts are saved to chart.html in the current folder.
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import plotly.io as pio  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from workout_chat.config import missing_secrets  # noqa: E402
from workout_chat.data import DataLoadError, load_dataframe  # noqa: E402
from workout_chat.graph import build_graph, run_turn  # noqa: E402


def show(result) -> None:
    print(f"\n[route: {result.route}]  [models: {result.models_used}]")
    if result.standalone_question:
        print(f"[understood as: {result.standalone_question}]")
    if result.code:
        print(f"\n--- {result.language} ---\n{result.code}\n---")
    if result.error:
        print(f"[detail: {result.error}]")
    if result.figure_json:
        pio.from_json(result.figure_json).write_html("chart.html")
        print("[chart saved to chart.html]")
    print(f"\n{result.answer}\n")


def main() -> int:
    needed = [m for m in missing_secrets() if m not in ("APP_PASSWORD", "GUEST_PASSWORD")]
    if needed:
        print(f"Missing secrets: {', '.join(needed)}")
        return 1
    try:
        df = load_dataframe()
    except DataLoadError as exc:
        print(f"LOAD FAILED: {exc}")
        return 1
    graph = build_graph(df, checkpointer=InMemorySaver())
    thread = uuid.uuid4().hex
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    if questions:
        show(run_turn(graph, thread, questions[0], on_step=lambda s: print(f"  ... {s}")))
        return 0
    print("Ask about Kevin's workouts ('quit' to stop).")
    while True:
        q = input("> ").strip()
        if q.lower() in ("quit", "exit", ""):
            return 0
        show(run_turn(graph, thread, q, on_step=lambda s: print(f"  ... {s}")))


if __name__ == "__main__":
    raise SystemExit(main())
