"""Child process that runs generated pandas/plotly code.

Started by ``sandbox.run_python`` as ``python -m workout_chat.python_worker``. It reads a pickled
payload from stdin, runs the (already validated) code with a restricted set of builtins, and
writes ONE JSON object to stdout. Never import this module from the app.
"""
from __future__ import annotations

import builtins
import contextlib
import datetime
import io
import json
import math
import os
import pickle
import pprint
import re
import sys
import tempfile
import traceback

MAX_FIGURE_BYTES = 1_500_000

_SAFE_BUILTIN_NAMES = (
    "abs all any bool dict divmod enumerate filter float format frozenset int isinstance issubclass iter "
    "len list map max min next pow print range repr reversed round set slice sorted str sum tuple zip "
    "chr ord callable hasattr "
    "ArithmeticError AttributeError Exception IndexError KeyError LookupError TypeError ValueError "
    "ZeroDivisionError True False None"
).split()


def _safe_builtins() -> dict:
    return {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}


def format_result(obj, max_rows: int = 40, max_chars: int = 6000) -> str:
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        text = obj.head(max_rows).to_string(index=False, max_colwidth=60)
        if len(obj) > max_rows:
            text += f"\n... ({len(obj)} rows in total, showing the first {max_rows})"
    elif isinstance(obj, pd.Series):
        text = obj.to_string(max_rows=max_rows)
        if len(obj) > max_rows:
            text += f"\n... ({len(obj)} values in total)"
    elif isinstance(obj, np.generic):
        text = str(obj.item())
    elif isinstance(obj, (dict, list, tuple, set)):
        text = pprint.pformat(obj, width=120, compact=True)
    else:
        text = str(obj)
    return text if len(text) <= max_chars else text[:max_chars] + "\n... [truncated]"


def _error_text(exc: BaseException) -> str:
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<generated>":
            line = frame.lineno
    where = f" (line {line})" if line else ""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}{where}"


def _is_plotly_figure(obj) -> bool:
    return hasattr(obj, "to_json") and type(obj).__module__.startswith("plotly")


def execute(payload: dict) -> dict:
    code, df = payload["code"], payload["df"]
    max_rows = payload.get("max_rows", 40)
    max_chars = payload.get("max_chars", 6000)
    os.chdir(tempfile.mkdtemp())

    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 200)
    namespace = {
        "__builtins__": _safe_builtins(),
        "df": df.copy(),
        "pd": pd, "np": np, "math": math, "datetime": datetime,
    }
    if re.search(r"\bpx\b", code):
        import plotly.express as px
        namespace["px"] = px
    if re.search(r"\bgo\b", code):
        import plotly.graph_objects as go
        namespace["go"] = go

    printed = io.StringIO()
    try:
        compiled = compile(code, "<generated>", "exec")
        with contextlib.redirect_stdout(printed):
            exec(compiled, namespace)  # noqa: S102 - validated upstream, restricted builtins, own process
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _error_text(exc)}

    result, fig = namespace.get("result"), namespace.get("fig")
    if fig is None and _is_plotly_figure(result):
        fig, result = result, None

    figure_json = None
    if fig is not None:
        if not _is_plotly_figure(fig):
            return {"ok": False, "error": "`fig` must be a Plotly figure (use px.* or go.Figure)."}
        figure_json = fig.to_json()
        if len(figure_json) > MAX_FIGURE_BYTES:
            return {"ok": False, "error": "The chart is too large; aggregate the data or plot fewer points."}

    printed_text = printed.getvalue().strip()
    if result is None and not printed_text and figure_json is None:
        return {"ok": False, "error": "The code produced no output: assign the answer to a variable named `result`."}

    parts = []
    if result is not None:
        parts.append(format_result(result, max_rows, max_chars))
    if printed_text:
        parts.append(("Printed output:\n" if parts else "") + printed_text[:max_chars])
    text = "\n".join(parts) if parts else "(A chart was created; no other result.)"
    return {"ok": True, "text": text, "figure_json": figure_json}


def main() -> None:
    out = sys.__stdout__
    try:
        payload = pickle.load(sys.stdin.buffer)
        response = execute(payload)
    except BaseException as exc:  # noqa: BLE001 - always answer with JSON
        response = {"ok": False, "error": _error_text(exc)}
    out.write(json.dumps(response))
    out.flush()


if __name__ == "__main__":
    main()
