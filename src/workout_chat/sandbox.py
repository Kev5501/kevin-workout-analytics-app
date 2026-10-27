"""Run model-written SQL and Python against the workout data, safely enough.

This is a set of guardrails, not a perfect jail. The data is Kevin's own, nothing here can reach
secrets or the network, and the real protections are layered:
  SQL    - one SELECT only, DuckDB with file/network access switched off and the config locked,
           a timeout, and a row cap.
  Python - a static check that blocks imports, file/network/IO methods and dunder tricks, then
           execution in a separate process with restricted builtins, a copy of the data, a scrubbed
           environment, a throwaway working directory and a hard timeout.
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from .schema import SQL_TABLE_NAME

SRC_DIR = Path(__file__).resolve().parents[1]
SQL_TIMEOUT_SECONDS = 10
PYTHON_TIMEOUT_SECONDS = 20
MAX_ROWS = 40          # rows shown to the answer-writing model
MAX_CHARS = 6000
MAX_CODE_CHARS = 8000


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    text: str = ""
    error: str | None = None
    figure_json: str | None = None


def _short(text: object, limit: int = 400) -> str:
    return " ".join(str(text).split())[:limit]


# --------------------------------------------------------------------------
# SQL (DuckDB)
# --------------------------------------------------------------------------
def run_sql(sql: str, df: pd.DataFrame, *, timeout: float = SQL_TIMEOUT_SECONDS, max_rows: int = MAX_ROWS) -> SandboxResult:
    if not sql.strip():
        return SandboxResult(False, error="No SQL was provided.")
    if len(sql) > MAX_CODE_CHARS:
        return SandboxResult(False, error="The SQL is too long.")
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        return SandboxResult(False, error=f"SQL syntax error: {_short(exc)}")
    if len(statements) != 1:
        return SandboxResult(False, error="Exactly one SQL statement is allowed.")
    if statements[0].type != duckdb.StatementType.SELECT or sql.lstrip().lower().startswith("pragma"):
        return SandboxResult(False, error="Only SELECT queries are allowed (read-only).")

    con = duckdb.connect(":memory:")
    timer = threading.Timer(timeout, con.interrupt)
    try:
        con.register(SQL_TABLE_NAME, df)
        con.execute("SET autoinstall_known_extensions=false")
        con.execute("SET autoload_known_extensions=false")
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
        timer.start()
        cursor = con.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description]
    except duckdb.InterruptException:
        return SandboxResult(False, error=f"The query took longer than {timeout:.0f} seconds.")
    except duckdb.Error as exc:
        return SandboxResult(False, error=_short(exc))
    finally:
        timer.cancel()
        con.close()

    if not rows:
        return SandboxResult(True, text="(The query returned 0 rows.)")
    shown = pd.DataFrame(rows[:max_rows], columns=columns)
    text = shown.to_string(index=False, max_colwidth=60)
    if len(rows) > max_rows:
        text += f"\n... (more rows exist; showing the first {max_rows})"
    return SandboxResult(True, text=text[:MAX_CHARS])


# --------------------------------------------------------------------------
# Python (separate process)
# --------------------------------------------------------------------------
BLOCKED_NAMES = frozenset(
    "open exec eval compile input breakpoint globals locals vars getattr setattr delattr dir help exit quit "
    "memoryview os sys subprocess importlib builtins socket pathlib shutil requests urllib pickle".split()
)
BLOCKED_ATTRS = frozenset(
    "to_csv to_excel to_pickle to_parquet to_feather to_hdf to_sql to_clipboard to_orc to_stata to_xml to_latex "
    "to_html to_markdown write_html write_image write_json show save savez savez_compressed savetxt load loadtxt "
    "genfromtxt fromfile tofile memmap system popen eval io".split()
)


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.error: str | None = None

    def fail(self, node: ast.AST, message: str) -> None:
        if self.error is None:
            self.error = f"{message} (line {getattr(node, 'lineno', '?')})"

    def visit_Import(self, node):  # noqa: N802
        self.fail(node, "Imports are not allowed; pd, np, px, go, math and datetime are already available")

    visit_ImportFrom = visit_Import  # noqa: N815

    def visit_Name(self, node):  # noqa: N802
        if node.id in BLOCKED_NAMES or node.id.startswith("__"):
            self.fail(node, f"`{node.id}` is not allowed")

    def visit_Attribute(self, node):  # noqa: N802
        if node.attr.startswith("_") or node.attr.startswith("read_") or node.attr in BLOCKED_ATTRS:
            self.fail(node, f"`.{node.attr}` is not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node):  # noqa: N802
        if isinstance(node.value, str) and "__" in node.value:
            self.fail(node, "Strings containing '__' are not allowed")

    def _no_async(self, node):
        self.fail(node, "async code is not allowed")

    visit_AsyncFunctionDef = visit_AsyncFor = visit_AsyncWith = visit_Await = _no_async  # noqa: N815


def validate_python(code: str) -> str | None:
    """Return an error message if the code must not run, else None."""
    if not code.strip():
        return "No Python code was provided."
    if len(code) > MAX_CODE_CHARS:
        return "The Python code is too long."
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    checker = _Checker()
    checker.visit(tree)
    return checker.error


def _child_env() -> dict[str, str]:
    keep = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "TEMP", "TMP", "TMPDIR", "HOME", "LANG", "LC_ALL")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(PYTHONPATH=str(SRC_DIR), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    return env  # note: no API keys or other secrets are passed on


def run_python(code: str, df: pd.DataFrame, *, timeout: float = PYTHON_TIMEOUT_SECONDS) -> SandboxResult:
    problem = validate_python(code)
    if problem:
        return SandboxResult(False, error=problem)
    payload = pickle.dumps({"code": code, "df": df, "max_rows": MAX_ROWS, "max_chars": MAX_CHARS})
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "workout_chat.python_worker"],
                input=payload, capture_output=True, timeout=timeout, env=_child_env(), cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"The code took longer than {timeout:.0f} seconds (infinite loop or too heavy).")
    try:
        reply = json.loads(proc.stdout.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        tail = _short(proc.stderr.decode("utf-8", "replace")[-300:])
        return SandboxResult(False, error=f"The code crashed the sandbox. {tail}")
    if not reply.get("ok"):
        return SandboxResult(False, error=reply.get("error", "Unknown error"))
    return SandboxResult(True, text=reply.get("text", ""), figure_json=reply.get("figure_json"))
