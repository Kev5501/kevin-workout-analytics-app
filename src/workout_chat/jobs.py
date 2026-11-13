"""Run chat turns in background threads, so an answer survives the page going away.

On a phone, locking the screen drops Streamlit's websocket; the page then reloads or reruns, and anything
running inside the script run is lost. Here the turn runs in a worker thread owned by the app process: it
saves its reply through ``on_done`` whatever happens to the page, and any later page (same session or a
reloaded one) can look the job up by conversation id to show progress.

Streamlit-free on purpose: ``work`` and ``on_done`` must not call ``st.*`` (there is no script context in a
worker thread).
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from .graph import GENERIC_ERROR, INITIAL_LABEL

logger = logging.getLogger(__name__)

RUNNING, DONE, FAILED = "running", "done", "failed"
KEEP_FINISHED_SECONDS = 600


@dataclass
class TurnJob:
    thread_id: str
    question: str
    status: str = RUNNING
    label: str = INITIAL_LABEL
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    message: Optional[dict] = None   # the assistant reply, once finished
    save_error: str = ""             # set when on_done failed (the page then shows ``message`` itself)

    @property
    def running(self) -> bool:
        return self.status == RUNNING


def _default_executor() -> Executor:
    return ThreadPoolExecutor(max_workers=4, thread_name_prefix="turn")


class JobRunner:
    """One running turn per conversation; finished jobs are remembered for a while for late pages."""

    def __init__(self, executor: Executor | None = None, *, keep_seconds: float = KEEP_FINISHED_SECONDS,
                 clock: Callable[[], float] = time.time):
        self._executor = executor or _default_executor()
        self._jobs: dict[str, TurnJob] = {}
        self._lock = threading.Lock()
        self._keep = keep_seconds
        self._clock = clock

    def get(self, thread_id: str) -> TurnJob | None:
        with self._lock:
            self._forget_old()
            return self._jobs.get(thread_id)

    def is_running(self, thread_id: str) -> bool:
        job = self.get(thread_id)
        return bool(job and job.running)

    def submit(
        self,
        thread_id: str,
        question: str,
        work: Callable[[Callable[[str], None]], dict],
        on_done: Callable[[dict], None],
    ) -> TurnJob:
        """Start ``work(on_step) -> assistant message`` in the background; ``on_done(message)`` persists it.

        Raises RuntimeError if this conversation already has a turn running.
        """
        with self._lock:
            self._forget_old()
            current = self._jobs.get(thread_id)
            if current and current.running:
                raise RuntimeError("A question is already being answered in this conversation")
            job = TurnJob(thread_id, question, started_at=self._clock())
            self._jobs[thread_id] = job
        self._executor.submit(self._run, job, work, on_done)
        return job

    def _run(self, job: TurnJob, work, on_done) -> None:
        def on_step(label: str) -> None:
            job.label = label

        status = DONE
        try:
            message = work(on_step)
        except Exception as exc:  # noqa: BLE001 - work is expected never to raise; be safe anyway
            logger.exception("Background turn failed")
            message = {"role": "assistant", "content": GENERIC_ERROR, "debug": f"detail: {type(exc).__name__}: {exc}"}
            status = FAILED
        job.message = message
        try:
            on_done(message)
        except Exception as exc:  # noqa: BLE001 - the page falls back to job.message
            logger.exception("Saving the reply failed")
            job.save_error = f"{type(exc).__name__}: {exc}"
        # Flip the status last, so a page that sees "finished" also finds the saved reply.
        job.finished_at = self._clock()
        job.status = status

    def _forget_old(self) -> None:
        now = self._clock()
        stale = [k for k, j in self._jobs.items() if j.finished_at is not None and now - j.finished_at > self._keep]
        for key in stale:
            del self._jobs[key]
